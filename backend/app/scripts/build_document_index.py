from __future__ import annotations

import argparse
import json

from app.data import teaching_data
from app.services.document_rag import DocumentRagService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成并持久化文档FAISS向量索引。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略现有索引并重新生成全部文档向量。",
    )
    args = parser.parse_args()

    service = DocumentRagService(teaching_data)
    metadata = service.vector_index.ensure(force=args.force)
    print(
        json.dumps(
            {
                "status": "ok",
                "index_type": metadata["index_type"],
                "provider": metadata["provider"],
                "dimensions": metadata["dimensions"],
                "vector_count": metadata["vector_count"],
                "load_source": metadata["load_source"],
                "index_file": str(service.vector_index.index_path),
                "metadata_file": str(
                    service.vector_index.metadata_path
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
