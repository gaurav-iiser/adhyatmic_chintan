from __future__ import annotations

import json
import math
import re
from pathlib import Path

from openai import OpenAI

from .config import Settings
from .models import (
    RagChunk,
    RagIndex,
    RetrievalHit,
    RetrievalResult,
)


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    if len(a) != len(b):
        raise ValueError(
            "Embedding dimension mismatch: "
            f"{len(a)} != {len(b)}"
        )

    dot_product = sum(
        x * y
        for x, y in zip(a, b, strict=True)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Cannot calculate cosine similarity "
            "for a zero-length embedding vector."
        )

    return dot_product / (
        norm_a * norm_b
    )


def _load_index(
    settings: Settings,
) -> RagIndex:
    index_path = (
        settings.data_dir
        / "_index"
        / "rag_index.json"
    )

    if not index_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: "
            f"{index_path}. "
            "Run 'adhyatmik index' first."
        )

    payload = _read_json(
        index_path
    )

    return RagIndex.model_validate(
        payload
    )


def _embed_query(
    *,
    client: OpenAI,
    model: str,
    query: str,
) -> list[float]:
    response = client.embeddings.create(
        model=model,
        input=query,
    )

    if not response.data:
        raise RuntimeError(
            "Embedding API returned no "
            "query embedding."
        )

    return response.data[0].embedding

_LECTURE_ID_PATTERN = re.compile(
    r"^lecture(?P<chapter>\d+)\.(?P<lecture>\d+)$",
    re.IGNORECASE,
)


def _lecture_parts(
    lecture_id: str,
) -> tuple[int, int] | None:
    match = _LECTURE_ID_PATTERN.match(
        lecture_id.strip()
    )

    if match is None:
        return None

    return (
        int(match.group("chapter")),
        int(match.group("lecture")),
    )


def _lecture_is_in_chapter(
    lecture_id: str,
    chapter: int,
) -> bool:
    parts = _lecture_parts(
        lecture_id
    )

    if parts is None:
        return False

    return parts[0] == chapter


def _chunk_sort_key(
    chunk: RagChunk,
) -> tuple[int, int, str]:
    parts = _lecture_parts(
        chunk.lecture_id
    )

    if parts is None:
        return (
            10**9,
            10**9,
            chunk.chunk_id,
        )

    return (
        parts[0],
        parts[1],
        chunk.chunk_id,
    )


def get_chapter_chunks(
    chapter: int,
    settings: Settings,
) -> tuple[RagIndex, list[RagChunk]]:
    if chapter <= 0:
        raise ValueError(
            "Chapter number must be greater "
            "than zero."
        )

    index = _load_index(
        settings
    )

    chunks = [
        indexed_chunk.chunk
        for indexed_chunk in index.chunks
        if _lecture_is_in_chapter(
            indexed_chunk.lecture_id,
            chapter,
        )
    ]

    if not chunks:
        raise ValueError(
            f"No indexed chunks found for "
            f"chapter {chapter}."
        )

    chunks.sort(
        key=_chunk_sort_key
    )

    return index, chunks

def retrieve(
    query: str,
    settings: Settings,
    *,
    top_k: int = 7,
    chapter: int | None = None,
    max_per_lecture: int | None = None,
) -> RetrievalResult:
    query = query.strip()

    if not query:
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if (
        max_per_lecture is not None
        and max_per_lecture <= 0
    ):
        raise ValueError(
            "max_per_lecture must be "
            "greater than zero."
        )

    index = _load_index(
        settings
    )

    if not index.chunks:
        raise ValueError(
            "RAG index contains no chunks."
        )

    candidates = list(
        index.chunks
    )

    if chapter is not None:
        candidates = [
            indexed_chunk
            for indexed_chunk in candidates
            if _lecture_is_in_chapter(
                indexed_chunk.lecture_id,
                chapter,
            )
        ]

        if not candidates:
            raise ValueError(
                f"No indexed chunks found "
                f"for chapter {chapter}."
            )

    client = OpenAI(
        api_key=settings.openai_api_key
    )
    # IMPORTANT:
    # Use the SAME embedding model that was
    # used when the index was built.

    query_embedding = _embed_query(
        client=client,
        model=index.embedding_model,
        query=query,
    )

    if (
        len(query_embedding)
        != index.embedding_dimensions
    ):
        raise ValueError(
            "Query embedding dimensions do not "
            "match the index. "
            f"Query: {len(query_embedding)}, "
            f"index: "
            f"{index.embedding_dimensions}."
        )

    scored_chunks: list[
        tuple[float, RagChunk]
    ] = []

    for indexed_chunk in candidates:
        score = _cosine_similarity(
            query_embedding,
            indexed_chunk.embedding,
        )

        scored_chunks.append(
            (
                score,
                indexed_chunk.chunk,
            )
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    target_count = min(
        top_k,
        len(scored_chunks),
    )

    if max_per_lecture is None:
        selected = scored_chunks[
            :target_count
        ]

    else:
        selected: list[
            tuple[float, RagChunk]
        ] = []

        lecture_counts: dict[
            str,
            int,
        ] = {}

        for score, chunk in scored_chunks:
            count = lecture_counts.get(
                chunk.lecture_id,
                0,
            )

            if count >= max_per_lecture:
                continue

            selected.append(
                (
                    score,
                    chunk,
                )
            )

            lecture_counts[
                chunk.lecture_id
            ] = count + 1

            if len(selected) >= target_count:
                break

        # If the diversity limit prevented us
        # from reaching top_k, fill the remainder
        # with the next best chunks.
        if len(selected) < target_count:
            selected_ids = {
                chunk.chunk_id
                for _, chunk in selected
            }

            for score, chunk in scored_chunks:
                if (
                    chunk.chunk_id
                    in selected_ids
                ):
                    continue

                selected.append(
                    (
                        score,
                        chunk,
                    )
                )

                selected_ids.add(
                    chunk.chunk_id
                )

                if (
                    len(selected)
                    >= target_count
                ):
                    break

    hits = [
        RetrievalHit(
            rank=rank,
            score=score,
            chunk=chunk,
        )
        for rank, (score, chunk)
        in enumerate(
            selected,
            start=1,
        )
    ]

    return RetrievalResult(
        query=query,
        embedding_model=(
            index.embedding_model
        ),
        top_k=len(hits),
        hits=hits,
    )

