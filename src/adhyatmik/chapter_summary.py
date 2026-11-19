from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .models import (
    AnswerReference,
    AnswerResult,
    RagChunk,
)
from .retrieval import (
    get_chapter_chunks,
)


class _SummaryDraft(BaseModel):
    text: str

    source_chunk_ids: list[str] = Field(
        default_factory=list
    )


class _DigestUnit(BaseModel):
    unit_index: int
    summary: str

    source_chunk_ids: list[str] = Field(
        default_factory=list
    )


class _ChapterDigest(BaseModel):
    cache_version: str = "v1"

    chapter: int

    index_created_at_utc: str
    source_chunk_count: int

    model: str
    batch_chars: int

    units: list[_DigestUnit] = Field(
        default_factory=list
    )


DIGEST_INSTRUCTIONS = """
You create a grounded digest of lecture transcript excerpts.

The supplied transcript chunks are the ONLY source.

You must NOT use:
- outside knowledge
- general knowledge of Bhagavad Gita
- outside Vedanta knowledge
- outside Sanskrit knowledge

Summarize the substantive teachings, distinctions, explanations,
arguments, examples, and conclusions present in the supplied chunks.

Do not invent connections that are not supported.

Do not turn questions raised by the speaker into conclusions unless
the supplied material actually answers them.

The digest should be concise enough to combine with other digests later,
but complete enough that major ideas in this batch are not lost.

source_chunk_ids must contain representative supplied chunk IDs that
directly support the digest.
"""


FINAL_INSTRUCTIONS = """
You answer using ONLY the supplied chapter digest units.

The digest units were themselves produced from the chapter's indexed
lecture transcript chunks.

Do NOT add outside knowledge.

Answer in the same primary language as the user's request.

Give one natural, clear, synthesized answer.

Do not mention chunk IDs, retrieval mechanics, or timestamps inside
the answer text.

If the request asks for a chapter summary, give a coherent overview
of the major teachings rather than describing lecture-by-lecture.

If optional user context is supplied, use it only to understand or
tailor the response. It is not a source of spiritual teaching.

source_chunk_ids must contain representative raw source IDs from the
supplied digest units that support the final answer.

Use no more than about 12 representative source chunk IDs.
"""


def _write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _dedupe(
    values: list[str],
) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        output.append(value)

    return output


def _batch_chunks(
    chunks: list[RagChunk],
    *,
    max_chars: int,
) -> list[list[RagChunk]]:
    if max_chars <= 0:
        raise ValueError(
            "chapter_summary_batch_chars "
            "must be greater than zero."
        )

    batches: list[
        list[RagChunk]
    ] = []

    current: list[RagChunk] = []
    current_chars = 0

    for chunk in chunks:
        approximate_chars = (
            len(chunk.text)
            + len(chunk.chunk_id)
            + len(chunk.lecture_id)
            + 100
        )

        if (
            current
            and (
                current_chars
                + approximate_chars
                > max_chars
            )
        ):
            batches.append(
                current
            )

            current = []
            current_chars = 0

        current.append(
            chunk
        )

        current_chars += (
            approximate_chars
        )

    if current:
        batches.append(
            current
        )

    return batches


def _build_chunk_packet(
    chunks: list[RagChunk],
) -> str:
    sections: list[str] = []

    for chunk in chunks:
        sections.append(
            f"""
SOURCE CHUNK ID: {chunk.chunk_id}
LECTURE: {chunk.lecture_id}

TRANSCRIPT:
{chunk.text}
""".strip()
        )

    return "\n\n---\n\n".join(
        sections
    )


def _call_summary_model(
    *,
    client: OpenAI,
    model: str,
    instructions: str,
    prompt: str,
) -> _SummaryDraft:
    completion = (
        client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=_SummaryDraft,
        )
    )

    message = completion.choices[
        0
    ].message

    if message.refusal:
        raise RuntimeError(
            "Summary model refused: "
            f"{message.refusal}"
        )

    if message.parsed is None:
        raise RuntimeError(
            "Summary model returned no "
            "parsed structured output."
        )

    if not message.parsed.text.strip():
        raise RuntimeError(
            "Summary model returned "
            "an empty summary."
        )

    return message.parsed


def _validate_source_ids(
    source_ids: list[str],
    *,
    allowed_ids: set[str],
) -> list[str]:
    source_ids = _dedupe(
        source_ids
    )

    valid_ids = [
        chunk_id
        for chunk_id in source_ids
        if chunk_id in allowed_ids
    ]

    if not valid_ids:
        raise ValueError(
            "Summary did not cite any valid "
            "supplied source chunks."
        )

    return valid_ids


def _build_digest(
    *,
    chapter: int,
    chunks: list[RagChunk],
    index_created_at_utc: str,
    settings: Settings,
) -> _ChapterDigest:
    batches = _batch_chunks(
        chunks,
        max_chars=(
            settings
            .chapter_summary_batch_chars
        ),
    )

    client = OpenAI(
        api_key=settings.openai_api_key
    )

    units: list[
        _DigestUnit
    ] = []

    for unit_index, batch in enumerate(
        batches,
        start=1,
    ):
        packet = _build_chunk_packet(
            batch
        )

        prompt = f"""
CHAPTER

{chapter}


TRANSCRIPT CHUNKS

{packet}


Create a grounded digest of ALL substantive material
in these supplied chunks.

Do not use outside knowledge.
"""

        draft = _call_summary_model(
            client=client,
            model=settings.answer_model,
            instructions=(
                DIGEST_INSTRUCTIONS
            ),
            prompt=prompt,
        )

        allowed_ids = {
            chunk.chunk_id
            for chunk in batch
        }

        source_ids = (
            _validate_source_ids(
                draft.source_chunk_ids,
                allowed_ids=allowed_ids,
            )
        )

        units.append(
            _DigestUnit(
                unit_index=unit_index,
                summary=(
                    draft.text.strip()
                ),
                source_chunk_ids=(
                    source_ids
                ),
            )
        )

    return _ChapterDigest(
        chapter=chapter,
        index_created_at_utc=(
            index_created_at_utc
        ),
        source_chunk_count=len(
            chunks
        ),
        model=settings.answer_model,
        batch_chars=(
            settings
            .chapter_summary_batch_chars
        ),
        units=units,
    )


def _load_or_build_digest(
    *,
    chapter: int,
    settings: Settings,
) -> tuple[
    _ChapterDigest,
    list[RagChunk],
]:
    index, chunks = (
        get_chapter_chunks(
            chapter,
            settings,
        )
    )

    cache_path = (
        settings.data_dir
        / "_summaries"
        / f"chapter_{chapter}_digest.json"
    )

    if cache_path.exists():
        try:
            payload = json.loads(
                cache_path.read_text(
                    encoding="utf-8"
                )
            )

            cached = (
                _ChapterDigest
                .model_validate(
                    payload
                )
            )

            cache_valid = (
                cached.cache_version
                == "v1"
                and cached.chapter
                == chapter
                and (
                    cached
                    .index_created_at_utc
                    == index.created_at_utc
                )
                and (
                    cached
                    .source_chunk_count
                    == len(chunks)
                )
                and cached.model
                == settings.answer_model
                and cached.batch_chars
                == (
                    settings
                    .chapter_summary_batch_chars
                )
                and bool(cached.units)
            )

            if cache_valid:
                return (
                    cached,
                    chunks,
                )

        except Exception:
            # Derived cache only.
            # If invalid/corrupt, rebuild it.
            pass

    digest = _build_digest(
        chapter=chapter,
        chunks=chunks,
        index_created_at_utc=(
            index.created_at_utc
        ),
        settings=settings,
    )

    _write_json(
        cache_path,
        digest.model_dump(),
    )

    return digest, chunks


def _build_digest_packet(
    digest: _ChapterDigest,
) -> str:
    sections: list[str] = []

    for unit in digest.units:
        source_ids = ", ".join(
            unit.source_chunk_ids
        )

        sections.append(
            f"""
DIGEST UNIT: {unit.unit_index}
SOURCE CHUNK IDS: {source_ids}

SUMMARY:
{unit.summary}
""".strip()
        )

    return "\n\n---\n\n".join(
        sections
    )


def summarize_chapter(
    *,
    query: str,
    context: str | None,
    chapter: int,
    settings: Settings,
) -> AnswerResult:
    digest, chunks = (
        _load_or_build_digest(
            chapter=chapter,
            settings=settings,
        )
    )

    digest_packet = (
        _build_digest_packet(
            digest
        )
    )

    context_block = (
        context
        if context
        else (
            "(No additional user "
            "context provided.)"
        )
    )

    prompt = f"""
CHAPTER

{chapter}


USER CONTEXT

{context_block}


USER REQUEST

{query}


CHAPTER DIGEST

{digest_packet}


Answer the USER REQUEST using ONLY the supplied
chapter digest.

Give one coherent synthesized answer.

If this is a request for a chapter summary, cover the
major themes across the chapter rather than focusing on
only one lecture.

Do not include references or timestamps inside the answer.
"""

    client = OpenAI(
        api_key=settings.openai_api_key
    )

    draft = _call_summary_model(
        client=client,
        model=settings.answer_model,
        instructions=(
            FINAL_INSTRUCTIONS
        ),
        prompt=prompt,
    )

    allowed_ids = {
        chunk_id
        for unit in digest.units
        for chunk_id
        in unit.source_chunk_ids
    }

    cited_ids = (
        _validate_source_ids(
            draft.source_chunk_ids,
            allowed_ids=allowed_ids,
        )
    )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    references = [
        AnswerReference(
            chunk_id=chunk_id,
            lecture_id=(
                chunks_by_id[
                    chunk_id
                ].lecture_id
            ),
            start_seconds=(
                chunks_by_id[
                    chunk_id
                ].start_seconds
            ),
            end_seconds=(
                chunks_by_id[
                    chunk_id
                ].end_seconds
            ),
        )
        for chunk_id in cited_ids
    ]

    return AnswerResult(
        query=query,
        context=context,
        answer_model=(
            settings.answer_model
        ),
        sufficient_context=True,
        rendered_answer=(
            draft.text.strip()
        ),
        references=references,
        retrieved_chunk_ids=[
            chunk.chunk_id
            for chunk in chunks
        ],
        cited_chunk_ids=(
            cited_ids
        ),
    )