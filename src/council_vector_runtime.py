#!/usr/bin/env python3
"""공개 포트폴리오용 벡터 검색 공통 런타임.

대회 원문/DB dump/embedding 파일은 저장소에 포함하지 않는다.
실행 시 로컬에서 준비된 임베딩 아티팩트와 PostgreSQL+pgvector를 사용한다.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from bm25_runtime import BM25Index, reciprocal_rank_fusion
from search_aliases import committee_alias_values, council_alias_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
DEFAULT_REVISION = "55ec6e9358a56d56af759bc8372e970caf8c305f"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def json_safe(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    return json.dumps(value, ensure_ascii=False, default=str)


def date_string(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


class EmbeddingArtifact:
    def __init__(self, embedding_dir: Path):
        root = embedding_dir.expanduser().resolve()
        self.root = root
        self.paths = type("Paths", (), {
            "root": root,
            "vectors": root / "embeddings.f16.npy",
            "metadata": root / "chunk_metadata.parquet",
            "summary": root / "embedding_summary.json",
        })()
        if not self.paths.vectors.is_file() or not self.paths.metadata.is_file():
            raise FileNotFoundError("임베딩 NPY와 metadata parquet가 필요합니다")
        self.vectors = np.load(self.paths.vectors, mmap_mode="r")
        self.parquet = pq.ParquetFile(self.paths.metadata)
        self.columns = self.parquet.schema_arrow.names
        self.summary = json.loads(self.paths.summary.read_text(encoding="utf-8")) if self.paths.summary.is_file() else {}
        if len(self.vectors) != self.parquet.metadata.num_rows:
            raise ValueError("벡터와 metadata 행 수가 다릅니다")

    @property
    def row_count(self) -> int:
        return len(self.vectors)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def model_name(self) -> str:
        return str(self.summary.get("model_name") or DEFAULT_MODEL)

    @property
    def revision(self) -> str:
        return str(self.summary.get("revision") or DEFAULT_REVISION)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "dimension": self.dimension,
            "model_name": self.model_name,
            "revision": self.revision,
        }


@dataclass
class SearchFilters:
    council: str | None = None
    year: int | None = None
    committee: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    priorities: list[str] | None = None
    policy_category: str | None = None


class QueryEncoder:
    def __init__(
        self,
        artifact: EmbeddingArtifact,
        device: str = "auto",
        model_cache: Path | None = None,
        local_files_only: bool = True,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        kwargs: dict[str, Any] = {"device": resolved, "local_files_only": local_files_only}
        if model_cache:
            kwargs["cache_folder"] = str(model_cache.expanduser().resolve())
        if artifact.revision:
            kwargs["revision"] = artifact.revision
        self.model = SentenceTransformer(artifact.model_name, **kwargs)
        self.model.max_seq_length = int(artifact.summary.get("max_seq_length") or 2048)
        self.model_name = artifact.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        kwargs = {
            "batch_size": min(16, max(1, len(texts))),
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if "query" in getattr(self.model, "prompts", {}):
            kwargs["prompt_name"] = "query"
        return self.model.encode(texts, **kwargs).astype(np.float32)


class CouncilSearchService:
    """PostgreSQL+pgvector dense 검색과 선택적 BM25 RRF를 제공한다."""

    def __init__(
        self,
        embedding_dir: Path,
        pg_dsn: str | None = None,
        pg_table: str = "council_arctic_raw_v1",
        retrieval_mode: str = "hybrid",
        bm25_index: Path | None = None,
        device: str = "auto",
    ):
        self.artifact = EmbeddingArtifact(embedding_dir)
        self.encoder = QueryEncoder(self.artifact, device=device, local_files_only=False)
        self.pg_dsn = pg_dsn or os.environ.get("PGVECTOR_DSN")
        if not self.pg_dsn:
            raise ValueError("PGVECTOR_DSN이 필요합니다")
        self.pg_table = pg_table
        self.retrieval_mode = retrieval_mode
        self.bm25 = BM25Index(bm25_index) if retrieval_mode == "hybrid" and bm25_index else None

    def _dense(self, vector: np.ndarray, limit: int, filters: SearchFilters) -> list[dict[str, Any]]:
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        conditions = []
        params: list[Any] = []
        councils = council_alias_values(filters.council)
        if councils:
            conditions.append(sql.SQL("council = ANY(%s)")); params.append(list(councils))
        if filters.year is not None:
            conditions.append(sql.SQL("year = %s")); params.append(filters.year)
        committees = committee_alias_values(filters.committee)
        if committees:
            conditions.append(sql.SQL("(") + sql.SQL(" OR ").join(sql.SQL("committee ILIKE %s") for _ in committees) + sql.SQL(")"))
            params.extend(f"%{x}%" for x in committees)
        if filters.date_from:
            conditions.append(sql.SQL("meeting_date >= %s")); params.append(filters.date_from)
        if filters.date_to:
            conditions.append(sql.SQL("meeting_date <= %s")); params.append(filters.date_to)
        where = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
        statement = sql.SQL(
            "SELECT chunk_id, document, metadata, 1-(embedding <=> %s) similarity "
            "FROM {}{} ORDER BY embedding <=> %s LIMIT %s"
        ).format(sql.Identifier(self.pg_table), where)
        with psycopg.connect(self.pg_dsn) as conn:
            register_vector(conn)
            rows = conn.execute(statement, [vector, *params, vector, limit]).fetchall()
        return [
            {"chunk_id": str(r[0]), "text": str(r[1] or ""), "metadata": dict(r[2] or {}), "similarity": float(r[3])}
            for r in rows
        ]

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 100,
        filters: SearchFilters | None = None,
        include_parent: bool = True,
    ) -> dict[str, Any]:
        filters = filters or SearchFilters()
        started = time.perf_counter()
        vector = self.encoder.encode([query])[0]
        dense = self._dense(vector, candidate_k, filters)
        candidates = dense

        if self.retrieval_mode == "hybrid" and self.bm25 is not None:
            hits, tokens = self.bm25.search(
                query, candidate_k,
                council=filters.council, year=filters.year,
                committee=filters.committee, date_from=filters.date_from,
                date_to=filters.date_to, priorities=filters.priorities,
                policy_category=filters.policy_category,
            )
            by_id = {row["chunk_id"]: row for row in dense}
            bm25_rows = [
                {
                    "chunk_id": hit.chunk_id,
                    "text": by_id.get(hit.chunk_id, {}).get("text", ""),
                    "metadata": by_id.get(hit.chunk_id, {}).get("metadata", {}),
                    "similarity": 0.0,
                    "bm25_score": hit.score,
                }
                for hit in hits
            ]
            candidates = reciprocal_rank_fusion(dense, bm25_rows, candidate_k)
        else:
            tokens = []

        results = []
        for rank, row in enumerate(candidates[:top_k], start=1):
            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "similarity": round(float(row.get("similarity") or 0.0), 6),
                "retrieval_score": row.get("rrf_score"),
                "text": row.get("text", ""),
                "metadata": {k: json_safe(v) for k, v in (row.get("metadata") or {}).items()},
            })
        return {
            "query": query,
            "backend": "postgresql+pgvector",
            "retrieval_mode": self.retrieval_mode,
            "filters": asdict(filters),
            "result_count": len(results),
            "bm25_tokens": tokens,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "results": results,
        }
