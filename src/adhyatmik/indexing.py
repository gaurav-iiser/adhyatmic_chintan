from __future__ import annotations

import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from .config import Settings
from .models import (
    IndexedChunk,
    RagChunk,
    RagChunkingResult,
    RagIndex,
)


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write_json(
    path: Path,
    payload: dict,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _find_rag_chunk_files(
    data_dir: Path,
) -> list[Path]:
    return sorted(
        path
        for path in data_dir.glob(
            "*/rag_chunks.json"
        )
        if path.is_file()
    )


def _load_all_chunks(
    *,
    data_dir: Path,
) -> tuple[list[str], list[RagChunk]]:
    rag_files = _find_rag_chunk_files(
        data_dir
    )

    if not rag_files:
        raise FileNotFoundError(
            "No rag_chunks.json files found under "
            f"{data_dir}."
        )

    lecture_ids: list[str] = []
    chunks: list[RagChunk] = []

    seen_lecture_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()

    for path in rag_files:
        payload = _read_json(path)

        result = (
            RagChunkingResult.model_validate(
                payload
            )
        )

        if result.lecture_id in seen_lecture_ids:
            raise ValueError(
                "Duplicate lecture_id encountered "
                f"while building index: "
                f"{result.lecture_id}"
            )

        seen_lecture_ids.add(
            result.lecture_id
        )

        lecture_ids.append(
            result.lecture_id
        )

        for chunk in result.chunks:
            if chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    "Duplicate chunk_id encountered: "
                    f"{chunk.chunk_id}"
                )

            if not chunk.text.strip():
                raise ValueError(
                    "Cannot embed empty RAG chunk: "
                    f"{chunk.chunk_id}"
                )

            seen_chunk_ids.add(
                chunk.chunk_id
            )

            chunks.append(chunk)

    lecture_ids.sort()

    chunks.sort(
        key=lambda chunk: (
            chunk.lecture_id,
            chunk.chunk_id,
        )
    )

    return lecture_ids, chunks


def _embed_chunks(
    *,
    client: OpenAI,
    model: str,
    chunks: list[RagChunk],
    batch_size: int,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError(
            "embedding_batch_size must be "
            "greater than zero."
        )

    embeddings: list[list[float]] = []

    for start in range(
        0,
        len(chunks),
        batch_size,
    ):
        batch = chunks[
            start:start + batch_size
        ]

        # IMPORTANT:
        # Embed only the exact RAG chunk text.
        texts = [
            chunk.text
            for chunk in batch
        ]

        response = client.embeddings.create(
            model=model,
            input=texts,
        )

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        if len(ordered_data) != len(batch):
            raise RuntimeError(
                "Embedding API returned an "
                "unexpected number of embeddings. "
                f"Expected {len(batch)}, "
                f"received {len(ordered_data)}."
            )

        for item in ordered_data:
            embeddings.append(
                item.embedding
            )

    return embeddings


def _validate_embeddings(
    *,
    chunks: list[RagChunk],
    embeddings: list[list[float]],
) -> int:
    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk count and embedding count "
            "do not match."
        )

    if not embeddings:
        raise ValueError(
            "No embeddings were generated."
        )

    dimensions = len(embeddings[0])

    if dimensions == 0:
        raise ValueError(
            "Embedding vector is empty."
        )

    for index, embedding in enumerate(
        embeddings
    ):
        if len(embedding) != dimensions:
            raise ValueError(
                "Embedding dimension mismatch at "
                f"position {index}. "
                f"Expected {dimensions}, "
                f"got {len(embedding)}."
            )

    return dimensions


def build_index(
    settings: Settings,
    *,
    force: bool = False,
) -> RagIndex:
    index_dir = (
        settings.data_dir / "_index"
    )

    index_json_path = (
        index_dir / "rag_index.json"
    )

    if index_json_path.exists() and not force:
        raise FileExistsError(
            "RAG index already exists: "
            f"{index_json_path}. "
            "Use --force only if you intentionally "
            "want to rebuild all embeddings."
        )

    lecture_ids, chunks = (
        _load_all_chunks(
            data_dir=settings.data_dir
        )
    )

    client = OpenAI(
        api_key=settings.openai_api_key
    )

    embeddings = _embed_chunks(
        client=client,
        model=settings.embedding_model,
        chunks=chunks,
        batch_size=(
            settings.embedding_batch_size
        ),
    )

    dimensions = _validate_embeddings(
        chunks=chunks,
        embeddings=embeddings,
    )

    indexed_chunks: list[
        IndexedChunk
    ] = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
        strict=True,
    ):
        indexed_chunks.append(
            IndexedChunk(
                chunk_id=chunk.chunk_id,
                lecture_id=chunk.lecture_id,

                text_sha256=_sha256_text(
                    chunk.text
                ),

                embedding=embedding,

                chunk=chunk,
            )
        )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = RagIndex(
        embedding_model=(
            settings.embedding_model
        ),

        embedding_dimensions=dimensions,

        source_lecture_ids=lecture_ids,

        source_chunk_count=len(
            indexed_chunks
        ),

        created_at_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        index_json_path=str(
            index_json_path
        ),

        chunks=indexed_chunks,
    )

    _write_json(
        index_json_path,
        result.model_dump(),
    )

    return result