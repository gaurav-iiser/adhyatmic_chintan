from __future__ import annotations

import re

from openai import OpenAI
from pydantic import BaseModel

from .config import Settings
from .models import (
    AnswerReference,
    AnswerResult,
    GroundedAnswerDraft,
    RetrievalResult,
)
from .retrieval import retrieve
from .chapter_summary import (
    summarize_chapter,
)


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

Give one natural, clear, synthesized answer to the user's actual
question.

Do NOT present the answer as a list of what different lecture excerpts
say.

Do NOT write phrases such as:
- "the lecturer says"
- "in this lecture"
- "at this timestamp"
- "according to chunk..."
unless the user specifically asks about the source itself.

Do NOT include citations, lecture IDs, chunk IDs, or timestamps inside
the answer text. References will be shown separately by the application.

Prefer:
1. a simple direct answer first
2. a short explanation when useful

Usually keep the answer concise and easy to understand rather than
turning it into a long lecture summary.

If the question is in Hindi, answer naturally in Hindi.

Use Sanskrit terms when they occur naturally in the lecture material.

Prefer the terminology and framing used by the speaker.

USER CONTEXT

The user may provide optional personal/background context.

When user context is supplied:
- use it to understand what the user is actually asking
- tailor the explanation to that situation
- connect the retrieved teaching to the supplied situation when supported
- you may treat details in USER CONTEXT as facts supplied by the user

However:
- USER CONTEXT is NOT a source of spiritual or scriptural teaching
- teaching claims must still come from the retrieved lecture sources
- do not invent additional facts about the user's situation
- do not make assumptions beyond what the user provided

When no user context is supplied:
- interpret the question according to its natural plain meaning
- do not invent a personal background
- give a simple, direct explanation of what the question is asking about
  and answer it from the retrieved sources

GROUNDING

Each answer point must contain:
- text
- source_chunk_ids

Every source_chunk_id must correspond to one of the supplied sources.

Use only sources that actually support that answer point.

Do not cite a source merely because it is generally related.

If different retrieved excerpts express different aspects of the answer,
you may use multiple answer points.

Match the breadth of the answer to the user's request.

For a focused question, give a concise focused answer.

For a broad question, synthesize the major distinct aspects supported
by the supplied sources rather than discussing only one narrow passage.


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
) -> tuple[
    str,
    list[str],
    list[AnswerReference],
]:
    chunks_by_id = {
        hit.chunk.chunk_id: hit.chunk
        for hit in retrieval.hits
    }

    if not draft.sufficient_context:
        message = (
            draft.insufficiency_message
            or (
                "उपलब्ध व्याख्यान सामग्री में इस प्रश्न का "
                "पर्याप्त उत्तर नहीं मिला।"
            )
        )

        return message, [], []

    rendered_points: list[str] = []
    cited_ids: list[str] = []

    for point in draft.answer_points:
        text = point.text.strip()

        if text:
            rendered_points.append(
                text
            )

        for chunk_id in point.source_chunk_ids:
            if chunk_id not in cited_ids:
                cited_ids.append(
                    chunk_id
                )

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

    return (
        "\n\n".join(
            rendered_points
        ),
        cited_ids,
        references,
    )

_DEVANAGARI_DIGITS = str.maketrans(
    "०१२३४५६७८९",
    "0123456789",
)


def _extract_chapter(
    query: str,
) -> int | None:
    normalized = query.translate(
        _DEVANAGARI_DIGITS
    )

    patterns = (
        r"\bchapter\s*[:\-]?\s*(\d{1,2})\b",
        r"अध्याय\s*[:\-]?\s*(\d{1,2})",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        chapter = int(
            match.group(1)
        )

        if 1 <= chapter <= 18:
            return chapter

    return None


def _looks_like_chapter_summary(
    query: str,
) -> bool:
    text = query.casefold()

    markers = (
        "summary",
        "summarize",
        "summarise",
        "overview",
        "सारांश",
        "का सार",
        "सार बत",
        "संक्षेप",
        "संक्षिप्त सार",
    )

    return any(
        marker in text
        for marker in markers
    )


def _looks_like_broad_question(
    query: str,
) -> bool:
    text = query.casefold()

    markers = (
        "main teachings",
        "key teachings",
        "key ideas",
        "major themes",
        "overall",
        "in general",
        "all teachings",
        "मुख्य शिक्षा",
        "मुख्य शिक्षाएँ",
        "मुख्य बातें",
        "प्रमुख शिक्षा",
        "प्रमुख बातें",
        "समग्र",
        "क्या-क्या",
        "सभी पहलू",
        "विस्तार से",
    )

    return any(
        marker in text
        for marker in markers
    )


def _resolve_answer_mode(
    *,
    query: str,
    mode: str,
    chapter: int | None,
) -> tuple[str, int | None]:
    normalized_mode = (
        mode.strip()
        .casefold()
        .replace("-", "_")
    )

    aliases = {
        "chapter": "chapter_summary",
        "summary": "chapter_summary",
    }

    normalized_mode = aliases.get(
        normalized_mode,
        normalized_mode,
    )

    allowed_modes = {
        "auto",
        "focused",
        "broad",
        "chapter_summary",
    }

    if normalized_mode not in allowed_modes:
        raise ValueError(
            "mode must be one of: "
            "auto, focused, broad, "
            "chapter-summary"
        )

    resolved_chapter = (
        chapter
        if chapter is not None
        else _extract_chapter(
            query
        )
    )

    if normalized_mode == "auto":
        if (
            resolved_chapter
            is not None
            and _looks_like_chapter_summary(
                query
            )
        ):
            return (
                "chapter_summary",
                resolved_chapter,
            )

        if _looks_like_broad_question(
            query
        ):
            return (
                "broad",
                resolved_chapter,
            )

        return (
            "focused",
            resolved_chapter,
        )

    if (
        normalized_mode
        == "chapter_summary"
        and resolved_chapter is None
    ):
        raise ValueError(
            "Chapter-summary mode requires "
            "a chapter number."
        )

    return (
        normalized_mode,
        resolved_chapter,
    )

def answer_question(
    query: str,
    settings: Settings,
    *,
    context: str | None = None,
    top_k: int | None = None,
    mode: str = "auto",
    chapter: int | None = None,
) -> AnswerResult:
    query = query.strip()

    context = (
        context.strip()
        if context
        and context.strip()
        else None
    )

    if not query:
        raise ValueError(
            "Question cannot be empty."
        )

    answer_mode, resolved_chapter = (
        _resolve_answer_mode(
            query=query,
            mode=mode,
            chapter=chapter,
        )
    )

    if answer_mode == "chapter_summary":
        return summarize_chapter(
            query=query,
            context=context,
            chapter=resolved_chapter,
            settings=settings,
        )

    if top_k is None:
        if answer_mode == "broad":
            top_k = (
                settings
                .answer_broad_top_k
            )
        else:
            top_k = (
                settings.answer_top_k
            )

    if context:
        retrieval_query = (
            f"Question:\n{query}\n\n"
            f"User context:\n{context}"
        )
    else:
        retrieval_query = query


    retrieval = retrieve(
        retrieval_query,
        settings,
        top_k=top_k,
        chapter=resolved_chapter,
        max_per_lecture=(
            settings
            .answer_broad_max_per_lecture
            if answer_mode == "broad"
            else None
        ),
    )

    source_packet = (
        _build_source_packet(
            retrieval
        )
    )

    context_block = (
        context
        if context
        else "(No additional user context provided.)"
    )

    if answer_mode == "broad":
        mode_guidance = """
    This is a BROAD question.

    Synthesize across the distinct retrieved sources.

    Cover the main supported themes and avoid letting one
    lecture dominate the answer merely because several
    similar chunks were retrieved from it.
    """
    else:
        mode_guidance = """
    This is a FOCUSED question.

    Answer the specific question directly and concisely.
    """

    prompt = f"""
    USER CONTEXT

    {context_block}

    USER QUESTION

    {query}


    RETRIEVED LECTURE SOURCES

    {source_packet}

    ANSWER MODE GUIDANCE

    {mode_guidance}

    Answer the USER QUESTION.

    Use USER CONTEXT to understand and tailor the answer when it is
    provided.

    Use ONLY the retrieved lecture sources for spiritual, philosophical,
    scriptural, or doctrinal claims.

    Give one clear, natural answer.

    Do not mention chunk IDs, timestamps, or retrieval mechanics in the
    answer itself.

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

    (
        rendered_answer,
        cited_ids,
        references,
    ) = _render_answer(
        draft=draft,
        retrieval=retrieval,
    )

    return AnswerResult(
        query=query,
        context=context,
        answer_model=(
            settings.answer_model
        ),
        sufficient_context=(
            draft.sufficient_context
        ),
        rendered_answer=(
            rendered_answer
        ),
        references=references,
        retrieved_chunk_ids=[
            hit.chunk.chunk_id
            for hit in retrieval.hits
        ],
        cited_chunk_ids=cited_ids,
    )