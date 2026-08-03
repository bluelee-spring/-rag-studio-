from __future__ import annotations

import gc
import importlib.util
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx

from ..config import settings
from ..models import (
    ModelConfigRequest,
    ModelPublicStatus,
    ModelTestResponse,
)


MODEL_WEIGHT_FILENAMES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
ADAPTER_WEIGHT_FILENAMES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
}
TOKENIZER_FILENAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
    "tokenizer.model",
}
MODEL_PATH_KEYS = {
    "base_model_name_or_path",
    "base_model_path",
    "model_name_or_path",
    "model_path",
    "model_source",
}
RUN_METADATA_FILENAMES = (
    "run_config.json",
    "training_config.json",
    "training_args.json",
    "metadata.json",
)


@dataclass(frozen=True)
class LocalArtifactResolution:
    requested_path: Path
    artifact_type: str
    model_path: Path
    adapter_path: Path | None
    base_model_source: str
    message: str


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:3]}{'*' * min(16, len(value) - 6)}{value[-3:]}"


def _api_endpoint(api_base: str, operation: str) -> str:
    """兼容填写 /v1、/chat/completions 或 /embeddings 三种形式。"""
    base = api_base.strip().rstrip("/")
    if not base:
        return ""
    suffixes = ("/chat/completions", "/embeddings")
    root = base
    for suffix in suffixes:
        if base.endswith(suffix):
            root = base[: -len(suffix)]
            break
    suffix = "/chat/completions" if operation == "chat" else "/embeddings"
    return f"{root}{suffix}"


def _clean_local_path(value: str) -> str:
    """允许学员直接粘贴带单引号或双引号的 Windows 路径。"""
    return value.strip().strip('"').strip("'").strip()


def _normalise_directory(path: Path) -> Path:
    if path.is_file() and path.name in (
        MODEL_WEIGHT_FILENAMES | ADAPTER_WEIGHT_FILENAMES
    ):
        return path.parent
    return path


def _has_tokenizer(path: Path) -> bool:
    return any((path / filename).exists() for filename in TOKENIZER_FILENAMES)


def _has_model_weights(path: Path) -> bool:
    if any((path / filename).exists() for filename in MODEL_WEIGHT_FILENAMES):
        return True
    if any(path.glob("model-*.safetensors")):
        return True
    return any(path.glob("pytorch_model-*.bin"))


def _has_adapter_weights(path: Path) -> bool:
    return any((path / filename).exists() for filename in ADAPTER_WEIGHT_FILENAMES)


def _full_model_problem(path: Path) -> str | None:
    if not path.exists():
        return "路径不存在于 FastAPI 后端所在计算机"
    if not path.is_dir():
        return "路径必须指向模型目录，而不是普通文件"
    if not (path / "config.json").exists():
        return "目录缺少 config.json"
    if not _has_tokenizer(path):
        return "目录缺少 Tokenizer 配置"
    if not _has_model_weights(path):
        return "目录缺少 Transformers 可读取的完整模型权重"
    return None


def _adapter_problem(path: Path) -> str | None:
    if not path.exists():
        return "LoRA 输出目录不存在于 FastAPI 后端所在计算机"
    if not path.is_dir():
        return "LoRA 输出路径必须是目录"
    if not (path / "adapter_config.json").exists():
        return "LoRA 输出目录缺少 adapter_config.json"
    if not _has_adapter_weights(path):
        return "LoRA 输出目录缺少 adapter_model 权重"
    return None


def _json_model_path_values(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in MODEL_PATH_KEYS and isinstance(nested, str) and nested.strip():
                yield nested.strip()
            yield from _json_model_path_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _json_model_path_values(nested)


class RuntimeModelManager:
    """当前后端进程的模型连接中心；密钥永不通过读取接口返回。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local_load_lock = threading.Lock()
        self._local_signature = ""
        self._local_model: Any | None = None
        self._local_tokenizer: Any | None = None
        self._local_loaded_device = ""
        self._local_loaded_dtype = ""
        self._provider = "environment"
        self._api_base = settings.llm_api_base
        self._api_key = settings.llm_api_key
        self._chat_model = settings.llm_model
        self._embedding_model = settings.embedding_model
        self._local_model_path = ""
        self._local_adapter_path = ""
        self._local_device = "auto"
        self._local_dtype = "auto"
        self._max_new_tokens = 512
        self._enable_planner = settings.enable_llm_planner
        self._enable_answer = settings.enable_llm_answer

    @property
    def planner_enabled(self) -> bool:
        return self._enable_planner and self.generation_ready

    @property
    def answer_enabled(self) -> bool:
        return self._enable_answer and self.generation_ready

    @property
    def generation_ready(self) -> bool:
        if self._provider in {"environment", "remote_api"}:
            return bool(self._api_base and self._api_key and self._chat_model)
        resolution, _ = self._resolve_local_artifact()
        if resolution is None:
            return False
        return not self._missing_local_dependencies(resolution)

    @property
    def embedding_ready(self) -> bool:
        return bool(self._api_base and self._api_key and self._embedding_model)

    @property
    def embedding_provider_name(self) -> str:
        return self._embedding_model if self.embedding_ready else ""

    def _candidate_base_paths(
        self,
        adapter_path: Path,
        configured_value: str,
    ) -> list[tuple[Path, str]]:
        candidates: list[tuple[Path, str]] = []
        seen: set[str] = set()

        def add(path: Path, source: str) -> None:
            normalised = _normalise_directory(path.expanduser())
            marker = str(normalised).casefold()
            if marker not in seen:
                seen.add(marker)
                candidates.append((normalised, source))

        raw_values: list[tuple[str, str]] = []
        if configured_value:
            raw_values.append(
                (configured_value, "adapter_config.json:base_model_name_or_path")
            )
        for filename in RUN_METADATA_FILENAMES:
            metadata_path = adapter_path / filename
            if not metadata_path.exists():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            raw_values.extend(
                (item, filename) for item in _json_model_path_values(payload)
            )

        for raw, source in raw_values:
            value = _clean_local_path(raw)
            path = Path(value)
            if path.is_absolute():
                add(path, source)
            else:
                add(adapter_path / path, source)
                add(adapter_path.parent / path, source)
                add(Path.cwd() / path, source)

        slash_normalised = configured_value.replace("\\", "/").rstrip("/")
        model_name = slash_normalised.rsplit("/", 1)[-1] if slash_normalised else ""
        if model_name:
            # LoRA Visual Lab 的常见结构：backend/outputs/run_xxx 与 backend/models/模型名。
            for parent in (adapter_path.parent, *adapter_path.parents):
                if parent.name.casefold() == "outputs":
                    add(
                        parent.parent / "models" / model_name,
                        "根据 outputs 与 models 的同级目录自动定位",
                    )
                    break
            for parent in list(adapter_path.parents)[:4]:
                add(
                    parent / "models" / model_name,
                    "根据微调工程目录结构自动定位",
                )
        return candidates

    def _resolve_adapter_base(
        self,
        adapter_path: Path,
    ) -> tuple[Path | None, str, str]:
        try:
            payload = json.loads(
                (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return None, "", f"无法读取 adapter_config.json：{exc}"
        configured_value = str(payload.get("base_model_name_or_path") or "").strip()
        for candidate, source in self._candidate_base_paths(
            adapter_path,
            configured_value,
        ):
            if _full_model_problem(candidate) is None:
                return candidate, source, ""
        display = configured_value or "未记录"
        return (
            None,
            "",
            "已识别为 LoRA Adapter，但没有找到对应的本地基础模型。"
            f"adapter_config.json 记录的基础模型为：{display}。"
            "请确认基础模型仍位于微调工程的 backend\\models 目录；"
            "也可以把基础模型填入主目录，并在高级输入框中填写此 Adapter 目录。",
        )

    def _resolve_local_artifact(
        self,
    ) -> tuple[LocalArtifactResolution | None, str]:
        if not self._local_model_path:
            return None, "尚未填写模型或微调输出目录"
        requested = _normalise_directory(
            Path(self._local_model_path).expanduser()
        )

        if self._local_adapter_path:
            adapter = _normalise_directory(
                Path(self._local_adapter_path).expanduser()
            )
            model_problem = _full_model_problem(requested)
            if model_problem:
                return None, f"基础模型无效：{model_problem}"
            adapter_problem = _adapter_problem(adapter)
            if adapter_problem:
                return None, adapter_problem
            return (
                LocalArtifactResolution(
                    requested_path=requested,
                    artifact_type="full_model_with_adapter",
                    model_path=requested,
                    adapter_path=adapter,
                    base_model_source="前端明确填写",
                    message="已识别为基础模型 + 独立 LoRA Adapter",
                ),
                "",
            )

        if requested.exists() and (requested / "adapter_config.json").exists():
            adapter_problem = _adapter_problem(requested)
            if adapter_problem:
                return None, adapter_problem
            model_path, source, problem = self._resolve_adapter_base(requested)
            if model_path is None:
                return None, problem
            return (
                LocalArtifactResolution(
                    requested_path=requested,
                    artifact_type="lora_adapter",
                    model_path=model_path,
                    adapter_path=requested,
                    base_model_source=source,
                    message="已识别为 LoRA 微调输出，并自动定位基础模型",
                ),
                "",
            )

        model_problem = _full_model_problem(requested)
        if model_problem:
            return None, f"模型目录无效：{model_problem}"
        return (
            LocalArtifactResolution(
                requested_path=requested,
                artifact_type="full_model",
                model_path=requested,
                adapter_path=None,
                base_model_source="选择的完整模型目录",
                message="已识别为完整 Hugging Face 模型或已合并模型",
            ),
            "",
        )

    @staticmethod
    def _missing_local_dependencies(
        resolution: LocalArtifactResolution,
    ) -> list[str]:
        required = ["torch", "transformers"]
        if resolution.adapter_path is not None:
            required.append("peft")
        return [
            name
            for name in required
            if importlib.util.find_spec(name) is None
        ]

    def public_status(self) -> ModelPublicStatus:
        notes: list[str] = []
        artifact_type = "unknown"
        resolved_model_path = ""
        resolved_adapter_path = ""
        if self._provider == "local_huggingface":
            resolution, problem = self._resolve_local_artifact()
            configured = resolution is not None
            if resolution is None:
                notes.append(problem)
            else:
                artifact_type = resolution.artifact_type
                resolved_model_path = str(resolution.model_path)
                resolved_adapter_path = (
                    str(resolution.adapter_path)
                    if resolution.adapter_path is not None
                    else ""
                )
                notes.append(resolution.message)
                if resolution.adapter_path is not None:
                    notes.append(
                        "基础模型："
                        + str(resolution.model_path)
                        + "（"
                        + resolution.base_model_source
                        + "）"
                    )
                missing = self._missing_local_dependencies(resolution)
                if missing:
                    notes.append("缺少本地推理依赖：" + ", ".join(missing))
        else:
            configured = bool(self._api_base and self._api_key and self._chat_model)
            if self._api_base:
                notes.append("Chat端点：" + _api_endpoint(self._api_base, "chat"))
            if self.embedding_ready:
                notes.append(
                    "Embedding端点："
                    + _api_endpoint(self._api_base, "embeddings")
                )
        return ModelPublicStatus(
            provider=self._provider,
            configured=configured,
            generation_ready=self.generation_ready,
            embedding_ready=self.embedding_ready,
            api_base=self._api_base,
            api_key_present=bool(self._api_key),
            api_key_masked=_mask_secret(self._api_key),
            chat_model=self._chat_model,
            embedding_model=self._embedding_model,
            local_model_path=self._local_model_path,
            local_adapter_path=self._local_adapter_path,
            local_artifact_type=artifact_type,
            resolved_model_path=resolved_model_path,
            resolved_adapter_path=resolved_adapter_path,
            model_loaded=self._local_model is not None,
            local_device=self._local_device,
            local_dtype=self._local_dtype,
            max_new_tokens=self._max_new_tokens,
            enable_planner=self._enable_planner,
            enable_answer=self._enable_answer,
            source=(
                "backend/.env"
                if self._provider == "environment"
                else "runtime-session"
            ),
            notes=notes,
        )

    def _release_local_model(self) -> None:
        had_model = self._local_model is not None
        self._local_model = None
        self._local_tokenizer = None
        self._local_signature = ""
        self._local_loaded_device = ""
        self._local_loaded_dtype = ""
        if not had_model:
            return
        gc.collect()
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def configure(self, request: ModelConfigRequest) -> ModelPublicStatus:
        with self._lock:
            previous_key = self._api_key
            previous_provider = self._provider
            previous_signature = self._current_local_signature()
            self._provider = request.provider
            self._api_base = request.api_base.strip().rstrip("/")
            if request.api_key is not None and request.api_key.strip():
                self._api_key = request.api_key.strip()
            elif request.provider == "remote_api" and not previous_key:
                self._api_key = ""
            self._chat_model = request.chat_model.strip()
            self._embedding_model = request.embedding_model.strip()
            self._local_model_path = _clean_local_path(request.local_model_path)
            self._local_adapter_path = _clean_local_path(request.local_adapter_path)
            self._local_device = request.local_device
            self._local_dtype = request.local_dtype
            self._max_new_tokens = request.max_new_tokens
            self._enable_planner = request.enable_planner
            self._enable_answer = request.enable_answer

            if (
                self._provider != previous_provider
                or self._current_local_signature() != previous_signature
            ):
                self._release_local_model()
        return self.public_status()

    def _current_local_signature(self) -> str:
        return "|".join(
            [
                self._local_model_path,
                self._local_adapter_path,
                self._local_device,
                self._local_dtype,
            ]
        )

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not self.embedding_ready:
            raise RuntimeError("尚未配置可用的Embedding API")
        response = httpx.post(
            _api_endpoint(self._api_base, "embeddings"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._embedding_model, "input": texts},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        rows = sorted(payload["data"], key=lambda item: item["index"])
        return [list(map(float, item["embedding"])) for item in rows]

    def _load_local(
        self,
    ) -> tuple[Any, Any, Any, str, LocalArtifactResolution]:
        resolution, problem = self._resolve_local_artifact()
        if resolution is None:
            raise RuntimeError(problem)
        missing = self._missing_local_dependencies(resolution)
        if missing:
            raise RuntimeError("缺少本地推理依赖：" + ", ".join(missing))

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        signature = self._current_local_signature()
        with self._local_load_lock:
            if (
                self._local_model is not None
                and self._local_tokenizer is not None
                and self._local_signature == signature
            ):
                return (
                    torch,
                    self._local_tokenizer,
                    self._local_model,
                    self._local_loaded_device,
                    resolution,
                )

            device = self._local_device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "选择了 CUDA，但当前 RAG 后端虚拟环境中的 PyTorch 检测不到 GPU"
                )
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError(
                    "选择了 Apple MPS，但当前 PyTorch 或 Mac 不支持 MPS"
                )

            dtype_name = self._local_dtype
            if dtype_name == "auto":
                dtype_name = "float16" if device in {"cuda", "mps"} else "float32"
            if device == "cpu" and dtype_name == "float16":
                raise RuntimeError("CPU 本地推理请使用 FP32，不要选择 FP16")
            if device == "mps" and dtype_name == "bfloat16":
                raise RuntimeError("Apple MPS 本地推理请使用 FP16 或 FP32，不要选择 BF16")
            if (
                device == "cuda"
                and dtype_name == "bfloat16"
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError("当前 GPU 不支持 BF16，请改用 FP16 或 FP32")
            dtype = getattr(torch, dtype_name)

            model_path = str(resolution.model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if resolution.adapter_path is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    str(resolution.adapter_path),
                    local_files_only=True,
                    is_trainable=False,
                )
            model.to(device)
            model.eval()
            self._local_model = model
            self._local_tokenizer = tokenizer
            self._local_signature = signature
            self._local_loaded_device = device
            self._local_loaded_dtype = dtype_name
            return torch, tokenizer, model, device, resolution

    def _generate_local(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> tuple[str, LocalArtifactResolution]:
        torch, tokenizer, model, device, resolution = self._load_local()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if getattr(tokenizer, "chat_template", None):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = f"System: {system}\nUser: {prompt}\nAssistant:"
        generation_tokens = max_tokens or self._max_new_tokens
        context_limit = int(
            getattr(model.config, "max_position_embeddings", 4096) or 4096
        )
        max_input_tokens = max(
            256,
            min(8192, context_limit - generation_tokens - 8),
        )
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        generated = outputs[0, inputs["input_ids"].shape[1] :]
        return (
            tokenizer.decode(generated, skip_special_tokens=True).strip(),
            resolution,
        )

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> tuple[str, str, int]:
        if not self.generation_ready:
            raise RuntimeError("当前生成模型尚未就绪")
        started = time.perf_counter()
        if self._provider == "local_huggingface":
            output, resolution = self._generate_local(
                system,
                prompt,
                max_tokens=max_tokens,
            )
            if resolution.adapter_path is not None:
                provider = f"{resolution.adapter_path.name} · LoRA"
            else:
                provider = resolution.model_path.name
        else:
            response = httpx.post(
                _api_endpoint(self._api_base, "chat"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens or self._max_new_tokens,
                },
                timeout=120,
            )
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"].strip()
            provider = self._chat_model
        if not output:
            raise RuntimeError("模型返回了空响应")
        elapsed = int((time.perf_counter() - started) * 1000)
        return output, provider, elapsed

    def test_connection(self) -> ModelTestResponse:
        started = time.perf_counter()
        if self._provider == "local_huggingface":
            resolution, problem = self._resolve_local_artifact()
            if resolution is None:
                return ModelTestResponse(
                    ok=False,
                    provider="local_huggingface",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    message=problem,
                    details={"loaded": False},
                )
            dependencies = {
                name: importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "peft")
            }
            missing = self._missing_local_dependencies(resolution)
            if missing:
                return ModelTestResponse(
                    ok=False,
                    provider="local_huggingface",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    message=(
                        "模型目录有效，但缺少本地推理依赖："
                        + ", ".join(missing)
                    ),
                    details={
                        "dependencies": dependencies,
                        "artifact_type": resolution.artifact_type,
                        "model_path": str(resolution.model_path),
                        "adapter_path": (
                            str(resolution.adapter_path)
                            if resolution.adapter_path is not None
                            else ""
                        ),
                        "loaded": False,
                    },
                )
            try:
                import torch

                sample, loaded_resolution = self._generate_local(
                    "你是模型运行自检器，只需给出简短响应。",
                    "请回答：模型连接正常。",
                    max_tokens=12,
                )
                if not sample:
                    raise RuntimeError("模型权重已加载，但试运行返回了空文本")
                return ModelTestResponse(
                    ok=True,
                    provider="local_huggingface",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    message="模型权重已加载，并完成一次真实本地生成",
                    details={
                        "dependencies": dependencies,
                        "artifact_type": loaded_resolution.artifact_type,
                        "model_path": str(loaded_resolution.model_path),
                        "adapter_path": (
                            str(loaded_resolution.adapter_path)
                            if loaded_resolution.adapter_path is not None
                            else ""
                        ),
                        "base_model_source": loaded_resolution.base_model_source,
                        "device": self._local_loaded_device,
                        "dtype": self._local_loaded_dtype,
                        "torch": torch.__version__,
                        "cuda_runtime": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(),
                        "sample": sample[:120],
                        "loaded": True,
                        "inspection_only": False,
                    },
                )
            except Exception as exc:
                return ModelTestResponse(
                    ok=False,
                    provider="local_huggingface",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    message=f"模型加载或试运行失败：{exc}",
                    details={
                        "dependencies": dependencies,
                        "artifact_type": resolution.artifact_type,
                        "model_path": str(resolution.model_path),
                        "adapter_path": (
                            str(resolution.adapter_path)
                            if resolution.adapter_path is not None
                            else ""
                        ),
                        "loaded": False,
                    },
                )
        try:
            answer, provider, latency = self.generate(
                system="你是连接测试器，只输出 OK。",
                prompt="输出 OK",
                temperature=0,
                max_tokens=8,
            )
            return ModelTestResponse(
                ok=True,
                provider=provider,
                latency_ms=latency,
                message="远程模型连接正常",
                details={
                    "sample": answer[:60],
                    "chat_endpoint": _api_endpoint(self._api_base, "chat"),
                    "embedding_ready": self.embedding_ready,
                },
            )
        except Exception as exc:
            return ModelTestResponse(
                ok=False,
                provider="remote_api",
                latency_ms=int((time.perf_counter() - started) * 1000),
                message=f"连接失败：{exc}",
            )


model_runtime = RuntimeModelManager()
