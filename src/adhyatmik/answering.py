from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from .config import Settings
from .models import (
    AnswerResult,
    GroundedAnswerDraft,
    RetrievalResult,
)
from .retrieval import retrieve


ANSWER_INSTRUCTIONS = """
You answer questions using ONLY the retrieved lecture excerpts provided
to you.

These excerpts come from human-reviewed transcripts of spiritual
lectures.

The retrieved excerpts are the ONLY authoritative source for your answer.

You must NOT:
- answer from your own knowledge
- supplement from general knowledge
- introduce teachings not present in the excerpts
- use outside knowledge of Bhagavad Gita
- use outside knowledge of Mahabharata
- use outside knowledge of Vedanta
- use outside Sanskrit knowledge
- invent quotations
- invent explanations
- invent scriptural references
- invent motives, causes, or philosophical conclusions

If the retrieved material does not adequately answer the question,
set sufficient_context to false.

Do not attempt to be helpful by filling gaps from memory.


ANSWER STYLE

Answer in the same primary language as the user's question.

If the question is in Hindi, answer naturally in Hindi.

Use Sanskrit terms when they occur naturally in the lecture material.

Prefer the terminology and framing used by the speaker.

Explain clearly, but do not make the speaker's teaching more systematic,
certain, or explicit than the retrieved source supports.


GROUNDING

Each answer point must contain:
- text
- source_chunk_ids

Every source_chunk_id must correspond to one of the supplied sources.

Use only sources that actually support that answer point.

Do not cite a source merely because it is generally related.

If different retrieved excerpts express different aspects of the answer,
you may use multiple answer points.

Prefer a concise, focused answer over a broad lecture summary.


QUESTIONS VS ANSWERS

If the lecture only raises a question but does not answer it in the
retrieved material, clearly say that the retrieved lecture excerpt raises
the question but does not provide the answer.

Never convert a question raised by the speaker into a teaching or
conclusion.


INSUFFICIENT CONTEXT

Set sufficient_context=false when:
- the retrieved passages do not substantially address the question
- only related terminology appears
- the answer would require outside knowledge
- the relevant passage appears incomplete

When sufficient_context=false:
- answer_points should normally be empty
- insufficiency_message should briefly explain that the available
  lecture material does not provide enough information
"""


def _format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)

    hours, remainder = divmod(
        total_seconds,
        3600,
    )

    minutes, secs = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}:"
            f"{minutes:02d}:"
            f"{secs:02d}"
        )

    return f"{minutes}:{secs:02d}"


def _build_source_packet(
    retrieval: RetrievalResult,
) -> str:
    sections: list[str] = []

    for hit in retrieval.hits:
        chunk = hit.chunk

        sections.append(
            f"""
SOURCE CHUNK ID: {chunk.chunk_id}
LECTURE: {chunk.lecture_id}
TIME RANGE: {_format_seconds(chunk.start_seconds)}–{_format_seconds(chunk.end_seconds)}
RETRIEVAL SCORE: {hit.score:.4f}

TRANSCRIPT:
{chunk.text}
""".strip()
        )

    return "\n\n---\n\n".join(
        sections
    )


def _validate_answer_sources(
    *,
    draft: GroundedAnswerDraft,
    retrieval: RetrievalResult,
) -> None:
    allowed_ids = {
        hit.chunk.chunk_id
        for hit in retrieval.hits
    }

    for point in draft.answer_points:
        if not point.source_chunk_ids:
            raise ValueError(
                "Answer point contains no supporting "
                f"source chunk: {point.text}"
            )

        for chunk_id in point.source_chunk_ids:
            if chunk_id not in allowed_ids:
                raise ValueError(
                    "Answer model cited a chunk that "
                    "was not retrieved: "
                    f"{chunk_id}"
                )


def _render_answer(
    *,
    draft: GroundedAnswerDraft,
    retrieval: RetrievalResult,
) -> tuple[str, list[str]]:
    chunks_by_id = {
        hit.chunk.chunk_id: hit.chunk
        for hit in retrieval.hits
    }

    if not draft.sufficient_context:
        message = (
            draft.insufficiency_message
            or (
                "उपलब्ध lecture excerpts में इस प्रश्न का "
                "पर्याप्त उत्तर नहीं मिला।"
            )
        )

        return message, []

    rendered_points: list[str] = []
    cited_ids: list[str] = []

    for point in draft.answer_points:
        citations: list[str] = []

        for chunk_id in point.source_chunk_ids:
            chunk = chunks_by_id[
                chunk_id
            ]

            if chunk_id not in cited_ids:
                cited_ids.append(
                    chunk_id
                )

            citations.append(
                (
                    f"{chunk.lecture_id} "
                    f"[{_format_seconds(chunk.start_seconds)}"
                    f"–"
                    f"{_format_seconds(chunk.end_seconds)}]"
                )
            )

        citation_text = "; ".join(
            citations
        )

        rendered_points.append(
            f"{point.text} ({citation_text})"
        )

    return (
        "\n\n".join(rendered_points),
        cited_ids,
    )


def answer_question(
    query: str,
    settings: Settings,
    *,
    top_k: int | None = None,
) -> AnswerResult:
    query = query.strip()

    if not query:
        raise ValueError(
            "Question cannot be empty."
        )

    if top_k is None:
        top_k = settings.answer_top_k

    retrieval = retrieve(
        query,
        settings,
        top_k=top_k,
    )

    source_packet = (
        _build_source_packet(
            retrieval
        )
    )

    prompt = f"""
USER QUESTION

{query}


RETRIEVED LECTURE SOURCES

{source_packet}


Answer the user's question using ONLY the retrieved lecture sources.

If those sources are insufficient, say so rather than using outside
knowledge.
"""

    client = OpenAI(
        api_key=settings.openai_api_key
    )

    completion = (
        client.beta.chat.completions.parse(
            model=settings.answer_model,
            messages=[
                {
                    "role": "system",
                    "content": ANSWER_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=GroundedAnswerDraft,
        )
    )

    message = completion.choices[0].message

    if message.refusal:
        raise RuntimeError(
            f"Answer model refused: "
            f"{message.refusal}"
        )

    if message.parsed is None:
        raise RuntimeError(
            "Answer model returned no parsed "
            "structured output."
        )

    draft = message.parsed

    _validate_answer_sources(
        draft=draft,
        retrieval=retrieval,
    )

    rendered_answer, cited_ids = (
        _render_answer(
            draft=draft,
            retrieval=retrieval,
        )
    )

    return AnswerResult(
        query=query,
        answer_model=(
            settings.answer_model
        ),
        sufficient_context=(
            draft.sufficient_context
        ),
        rendered_answer=(
            rendered_answer
        ),
        retrieved_chunk_ids=[
            hit.chunk.chunk_id
            for hit in retrieval.hits
        ],
        cited_chunk_ids=cited_ids,
    )