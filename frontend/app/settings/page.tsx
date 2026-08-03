"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { PlatformHeader } from "@/components/PlatformHeader";
import {
  loadModelStatus,
  saveModelConfig,
  testModelConnection,
} from "@/lib/api";
import type {
  ModelConfigInput,
  ModelStatus,
  ModelTestResult,
} from "@/lib/types";

const EMPTY: ModelConfigInput = {
  provider: "remote_api",
  api_base: "",
  api_key: null,
  chat_model: "",
  embedding_model: "",
  local_model_path: "",
  local_adapter_path: "",
  local_device: "auto",
  local_dtype: "auto",
  max_new_tokens: 512,
  enable_planner: false,
  enable_answer: true,
};

const ARTIFACT_LABEL: Record<ModelStatus["local_artifact_type"], string> = {
  unknown: "等待识别",
  full_model: "完整 / 已合并模型",
  lora_adapter: "LoRA 微调输出",
  full_model_with_adapter: "基础模型 + LoRA Adapter",
};

function endpoint(base: string, kind: "chat" | "embedding") {
  const clean = base.trim().replace(/\/$/, "");
  if (!clean) return "等待填写 API Base";
  const root = clean.replace(/\/(chat\/completions|embeddings)$/, "");
  return `${root}/${kind === "chat" ? "chat/completions" : "embeddings"}`;
}

export default function SettingsPage() {
  const [form, setForm] = useState<ModelConfigInput>(EMPTY);
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [testResult, setTestResult] = useState<ModelTestResult | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    loadModelStatus()
      .then((value) => {
        setStatus(value);
        setForm({
          provider:
            value.provider === "local_huggingface"
              ? "local_huggingface"
              : "remote_api",
          api_base: value.api_base,
          api_key: null,
          chat_model: value.chat_model,
          embedding_model: value.embedding_model,
          local_model_path: value.local_model_path,
          local_adapter_path: value.local_adapter_path,
          local_device: value.local_device as ModelConfigInput["local_device"],
          local_dtype: value.local_dtype as ModelConfigInput["local_dtype"],
          max_new_tokens: value.max_new_tokens,
          enable_planner: value.enable_planner,
          enable_answer: value.enable_answer,
        });
      })
      .catch((error) => setMessage(error.message));
  }, []);

  const providerLabel = useMemo(
    () =>
      form.provider === "remote_api"
        ? form.chat_model || "Remote API"
        : form.local_model_path.split(/[\\/]/).filter(Boolean).at(-1) ||
          "Local model",
    [form],
  );

  function update<K extends keyof ModelConfigInput>(
    key: K,
    value: ModelConfigInput[K],
  ) {
    setForm((current) => ({ ...current, [key]: value }));
    setMessage("");
    setTestResult(null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    try {
      const next = await saveModelConfig(form);
      setStatus(next);
      setForm((current) => ({ ...current, api_key: null }));
      setMessage("配置已保存；现在可以回到实验台执行 RAG 推理");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  async function test() {
    setTesting(true);
    setTestResult(null);
    setMessage("");
    try {
      const next = await saveModelConfig(form);
      setStatus(next);
      setForm((current) => ({ ...current, api_key: null }));
      const result = await testModelConnection();
      setTestResult(result);
      setStatus(await loadModelStatus());
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "测试失败");
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="platform-frame">
      <PlatformHeader serviceOnline={Boolean(status)} modelReady={status?.generation_ready} />
      <main className="platform-page model-page">
        <header className="platform-page-head">
          <div>
            <span>MODEL CONNECTION</span>
            <h1>模型连接</h1>
            <p>远程兼容 API，或直接连接微调课程使用的本地模型目录。</p>
          </div>
          <div className={status?.generation_ready ? "runtime-card ready" : "runtime-card"}>
            <i />
            <span>{status?.generation_ready ? "READY" : "NOT READY"}</span>
            <strong>{providerLabel}</strong>
            <small>{status?.source || "尚未读取后端状态"}</small>
          </div>
        </header>

        <form className="model-console" onSubmit={submit}>
          <div className="provider-switch">
            <button
              type="button"
              className={form.provider === "remote_api" ? "active" : ""}
              onClick={() => update("provider", "remote_api")}
            >
              <span>REMOTE</span>
              <strong>OpenAI 兼容 API</strong>
            </button>
            <button
              type="button"
              className={form.provider === "local_huggingface" ? "active" : ""}
              onClick={() => update("provider", "local_huggingface")}
            >
              <span>LOCAL</span>
              <strong>Hugging Face 模型目录</strong>
            </button>
          </div>

          {form.provider === "remote_api" ? (
            <section className="settings-grid remote-settings">
              <label className="field wide">
                <span>API Base</span>
                <input
                  value={form.api_base}
                  onChange={(event) => update("api_base", event.target.value)}
                  placeholder="https://example.com/v1"
                  spellCheck={false}
                />
                <small>可填写 /v1，也兼容完整的 /chat/completions 地址</small>
              </label>
              <label className="field wide">
                <span>API Key</span>
                <input
                  type="password"
                  value={form.api_key ?? ""}
                  onChange={(event) => update("api_key", event.target.value)}
                  placeholder={status?.api_key_present ? status.api_key_masked : "sk-..."}
                  autoComplete="new-password"
                />
                <small>留空保留当前 Key；读取接口不会返回明文</small>
              </label>
              <label className="field">
                <span>生成模型</span>
                <input
                  value={form.chat_model}
                  onChange={(event) => update("chat_model", event.target.value)}
                  placeholder="gpt-5.4"
                />
              </label>
              <label className="field">
                <span>Embedding 模型</span>
                <input
                  value={form.embedding_model}
                  onChange={(event) => update("embedding_model", event.target.value)}
                  placeholder="text-embedding-3-small"
                />
              </label>
              <div className="endpoint-map wide">
                <div><span>CHAT</span><code>{endpoint(form.api_base, "chat")}</code></div>
                <div><span>EMBED</span><code>{endpoint(form.api_base, "embedding")}</code></div>
              </div>
            </section>
          ) : (
            <section className="settings-grid local-settings">
              <label className="field wide">
                <span>模型或微调输出目录</span>
                <input
                  value={form.local_model_path}
                  onChange={(event) => update("local_model_path", event.target.value)}
                  placeholder={String.raw`C:\Users\卢航青\Downloads\LoRA-Visual-Lab-v0.7.0-Vocab-Field\lora-visual-lab\backend\models\Qwen2.5-0.5B-Instruct`}
                  spellCheck={false}
                />
                <small>
                  可直接填写完整模型、已合并模型，或 outputs\run_xxx。选择 LoRA 输出时，平台会读取 adapter_config.json 并自动定位基础模型。
                </small>
              </label>
              <label className="field wide">
                <span>高级：独立 LoRA Adapter 目录（通常留空）</span>
                <input
                  value={form.local_adapter_path}
                  onChange={(event) => update("local_adapter_path", event.target.value)}
                  placeholder="仅当上方填写基础模型、并希望手动指定 Adapter 时使用"
                  spellCheck={false}
                />
                <small>直接在上方填写 run_xxx 时，不要重复填写此项。</small>
              </label>
              <label className="field">
                <span>计算设备</span>
                <select
                  value={form.local_device}
                  onChange={(event) => update("local_device", event.target.value as ModelConfigInput["local_device"])}
                >
                  <option value="auto">自动检测</option>
                  <option value="cuda">CUDA</option>
                  <option value="mps">Apple MPS</option>
                  <option value="cpu">CPU</option>
                </select>
              </label>
              <label className="field">
                <span>权重精度</span>
                <select
                  value={form.local_dtype}
                  onChange={(event) => update("local_dtype", event.target.value as ModelConfigInput["local_dtype"])}
                >
                  <option value="auto">自动</option>
                  <option value="float16">FP16</option>
                  <option value="bfloat16">BF16</option>
                  <option value="float32">FP32</option>
                </select>
              </label>
              <div className="local-runtime-map wide">
                <div><span>目录识别</span><i /><b>完整模型 / LoRA</b></div>
                <div><span>权重装配</span><i /><b>Transformers + PEFT</b></div>
                <div><span>RAG 证据</span><i /><b>本地生成</b></div>
              </div>
              <div className="artifact-resolution wide">
                <header>
                  <span>AUTO RESOLUTION</span>
                  <strong>{status ? ARTIFACT_LABEL[status.local_artifact_type] : "等待保存检测"}</strong>
                  <small>{status?.model_loaded ? "权重已进入运行时" : "权重尚未加载"}</small>
                </header>
                <div>
                  <span>基础权重</span>
                  <code>{status?.resolved_model_path || "保存后显示实际加载目录"}</code>
                </div>
                <div>
                  <span>Adapter</span>
                  <code>{status?.resolved_adapter_path || "无 / 尚未识别"}</code>
                </div>
              </div>
            </section>
          )}

          <section className="settings-grid common-settings">
            <label className="field">
              <span>最大生成 Token</span>
              <input
                type="number"
                min={32}
                max={4096}
                value={form.max_new_tokens}
                onChange={(event) => update("max_new_tokens", Number(event.target.value))}
              />
            </label>
            <label className="toggle-field">
              <input
                type="checkbox"
                checked={form.enable_answer}
                onChange={(event) => update("enable_answer", event.target.checked)}
              />
              <span><b>启用证据约束生成</b><small>检索完成后调用当前模型</small></span>
            </label>
            <label className="toggle-field">
              <input
                type="checkbox"
                checked={form.enable_planner}
                onChange={(event) => update("enable_planner", event.target.checked)}
              />
              <span><b>启用 LLM 查询规划</b><small>通用表数据会生成受限 SQL 计划</small></span>
            </label>
          </section>

          <div className="model-privacy-note">
            <i />
            <p>
              <strong>数据边界</strong>
              {form.provider === "remote_api"
                ? "启用回答生成时会向该 API 发送问题与 Top-K 证据；启用查询规划时会发送表字段、类型和少量样例值。文档建向量索引时，文本块会发送到 Embedding 端点。"
                : "回答证据只进入本机模型；若仍配置远程 Embedding，新文档的文本块会发送到该 Embedding 端点。"}
            </p>
          </div>

          {status?.notes.length ? (
            <div className="status-notes">
              {status.notes.map((note) => <span key={note}>{note}</span>)}
            </div>
          ) : null}

          <footer className="settings-actions">
            <span>{message}</span>
            <button type="button" className="secondary-action" onClick={test} disabled={testing || saving}>
              {testing ? "正在加载并试运行" : "保存并加载测试"}
            </button>
            <button type="submit" className="primary-action" disabled={saving || testing}>
              {saving ? "保存中" : "保存到当前会话"}
            </button>
          </footer>
        </form>

        {testResult && (
          <section className={testResult.ok ? "test-result success" : "test-result failed"}>
            <i />
            <div><span>{testResult.ok ? "CONNECTION OK" : "CHECK FAILED"}</span><strong>{testResult.message}</strong></div>
            <b>{testResult.latency_ms} ms</b>
            {Object.keys(testResult.details).length > 0 && (
              <dl>
                {Boolean(testResult.details.artifact_type) && <><dt>类型</dt><dd>{String(testResult.details.artifact_type)}</dd></>}
                {Boolean(testResult.details.device) && <><dt>设备</dt><dd>{String(testResult.details.device)}</dd></>}
                {Boolean(testResult.details.dtype) && <><dt>精度</dt><dd>{String(testResult.details.dtype)}</dd></>}
                {Boolean(testResult.details.sample) && <><dt>试运行</dt><dd>{String(testResult.details.sample)}</dd></>}
              </dl>
            )}
          </section>
        )}
      </main>
    </div>
  );
}
