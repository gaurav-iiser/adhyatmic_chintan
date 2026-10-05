from __future__ import annotations

import json
import math
from pathlib import Path

from openai import OpenAI

from .config import Settings
from .models import (
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


def retrieve(
    query: str,
    settings: Settings,
    *,
    top_k: int = 5,
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

    index = _load_index(
        settings
    )

    if not index.chunks:
        raise ValueError(
            "RAG index contains no chunks."
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
        tuple[float, object]
    ] = []

    for indexed_chunk in index.chunks:
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

    selected = scored_chunks[
        :min(top_k, len(scored_chunks))
    ]

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