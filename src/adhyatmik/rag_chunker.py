from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .models import (
    EnrichmentResult,
    RagChunk,
    RagChunkingResult,
    RagSourceFragment,
)


class _ChunkBoundary(BaseModel):
    start_sentence_index: int
    end_sentence_index: int


class _ChunkPlan(BaseModel):
    chunks: list[_ChunkBoundary] = Field(default_factory=list)


@dataclass(frozen=True)
class _SentenceSpan:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _LectureSentence:
    global_index: int

    source_segment_index: int
    local_sentence_index: int

    char_start: int
    char_end: int

    text: str


CHUNKING_INSTRUCTIONS = """
You are the semantic RAG-chunking stage for the Adhyatmik project.

You receive the ordered transcript of an entire lecture, divided into
numbered sentence units.

The transcript originates from human-reviewed source segments.

IMPORTANT:

SOURCE SEGMENT BOUNDARIES ARE NOT SEMANTIC BOUNDARIES.

The original source segments were created largely for transcription and
processing convenience.

You may create a semantic chunk that crosses from one source segment
into the next when the speaker's thought continues across that boundary.

Your ONLY task is to decide which consecutive sentences belong together
as coherent retrieval chunks.

You must NOT:
- rewrite the transcript
- summarize the transcript
- correct the transcript
- translate the transcript
- remove transcript content
- reorder sentences
- duplicate sentences
- invent content

Return only GLOBAL sentence ranges.

Every input sentence must appear in exactly one chunk.

Chunks must:
- contain consecutive sentences
- preserve source order
- have no gaps
- have no overlaps


SEMANTIC COHERENCE

Prefer keeping together:
- a question and its immediate answer
- a definition and its explanation
- a Sanskrit term and its explanation
- a quotation or verse and its immediate interpretation
- an example and the teaching it directly illustrates
- a claim and the speaker's immediate reasoning
- a thought that begins near the end of one source segment and
  continues into the next

Prefer splitting when:
- one teaching has substantially concluded and another begins
- the speaker clearly changes topic
- a new independent question begins
- a new example begins that serves a different teaching
- narrative material shifts into a separate conceptual explanation
- a conceptual explanation shifts into a substantially separate example
- incidental room/logistics material interrupts the discourse

BOUNDARY QUALITY

A good retrieval chunk should be understandable when it begins.

Avoid starting a new chunk with a sentence that depends strongly on
the previous sentence for its grammatical or conceptual meaning.

Examples of weak chunk beginnings include sentences starting with:
- इसलिए
- तो
- लेकिन
- और
- क्योंकि
- इसीलिए
- उसका / उसकी / उनके / इसका
- तब / फिर
- ऐसा / ऐसा क्यों
- "उस समय..." when the referenced time or situation was established
  only in the previous chunk

These words do NOT automatically forbid a boundary.

However, before starting a chunk there, ask:

"Would a reader who retrieves only this new chunk understand what this
opening sentence refers to?"

If not, keep enough preceding sentences in the same chunk to make the
opening understandable.


PREFER COMPLETED THOUGHT TRANSITIONS

Prefer placing a chunk boundary AFTER a thought, argument, explanation,
example, or question-answer sequence has substantially completed.

Prefer boundaries where the speaker:
- returns from an example to the main teaching
- finishes an explanation and begins a new teaching
- explicitly shifts back to an earlier topic
- begins a new independent narrative or argument
- uses a transition that clearly opens a new subject

Do not split merely because the chunk has reached the target size.

If moving the boundary by a few sentences produces a substantially more
self-contained chunk, prefer the more coherent boundary even if chunk
sizes become less equal.


SIZE GUIDANCE

The caller supplies:
- minimum preferred size
- target size
- maximum preferred size

These are GUIDELINES, not semantic boundaries.

The TARGET is a useful approximate retrieval size.

The MAXIMUM is a ceiling, not a reason to combine everything below it.

For example:

A 2200-character passage containing three distinct teachings should
normally become multiple chunks even though it is below a 2600-character
maximum.

Semantic coherence is more important than reaching the exact target.

Avoid tiny fragments when neighboring sentences clearly form one thought.

Do not exceed the maximum unless a single indivisible sentence itself
already exceeds the maximum.

The ideal chunk should:
1. make sense when retrieved by itself
2. preserve enough context to understand the teaching
3. remain focused enough to match a specific user question

Return only start_sentence_index and end_sentence_index for each chunk.
"""


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _trimmed_span(
    text: str,
    raw_start: int,
    raw_end: int,
) -> tuple[int, int] | None:
    raw = text[raw_start:raw_end]

    if not raw.strip():
        return None

    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw.rstrip())

    start = raw_start + left_trim
    end = raw_start + right_trim

    return start, end


def _sentence_spans(
    text: str,
) -> list[_SentenceSpan]:
    spans: list[_SentenceSpan] = []

    boundary_pattern = re.compile(
        r"[।!?]+(?=\s|$)"
    )

    cursor = 0

    for match in boundary_pattern.finditer(text):
        trimmed = _trimmed_span(
            text,
            cursor,
            match.end(),
        )

        if trimmed is not None:
            start, end = trimmed

            spans.append(
                _SentenceSpan(
                    index=len(spans),
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )

        cursor = match.end()

    if cursor < len(text):
        trimmed = _trimmed_span(
            text,
            cursor,
            len(text),
        )

        if trimmed is not None:
            start, end = trimmed

            spans.append(
                _SentenceSpan(
                    index=len(spans),
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )

    if not spans and text.strip():
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())

        spans.append(
            _SentenceSpan(
                index=0,
                start=start,
                end=end,
                text=text[start:end],
            )
        )

    return spans


def _build_lecture_sentences(
    enriched: EnrichmentResult,
) -> list[_LectureSentence]:
    lecture_sentences: list[_LectureSentence] = []

    for segment in enriched.segments:
        local_spans = _sentence_spans(
            segment.cleaned_text
        )

        for span in local_spans:
            lecture_sentences.append(
                _LectureSentence(
                    global_index=len(
                        lecture_sentences
                    ),
                    source_segment_index=(
                        segment.index
                    ),
                    local_sentence_index=(
                        span.index
                    ),
                    char_start=span.start,
                    char_end=span.end,
                    text=span.text,
                )
            )

    return lecture_sentences


def _chunk_char_length(
    sentences: list[_LectureSentence],
    start_index: int,
    end_index: int,
) -> int:
    selected = sentences[
        start_index:end_index + 1
    ]

    if not selected:
        return 0

    # Approximate final assembled length.
    # One separator may be introduced between sentence/source units.
    return (
        sum(len(sentence.text) for sentence in selected)
        + max(0, len(selected) - 1)
    )

def _validate_chunk_structure(
    *,
    plan: _ChunkPlan,
    sentence_count: int,
) -> None:
    if not plan.chunks:
        raise ValueError(
            "Chunking model returned no chunks."
        )

    expected_start = 0

    for chunk_number, boundary in enumerate(plan.chunks):
        start = boundary.start_sentence_index
        end = boundary.end_sentence_index

        if start != expected_start:
            raise ValueError(
                "Chunk plan contains a gap, overlap, "
                "or reordered sentence at chunk "
                f"{chunk_number}. "
                f"Expected start {expected_start}, "
                f"got {start}."
            )

        if start < 0 or end < start:
            raise ValueError(
                f"Invalid sentence range: {start}-{end}."
            )

        if end >= sentence_count:
            raise ValueError(
                f"Chunk range {start}-{end} exceeds "
                "available sentence indexes "
                f"0-{sentence_count - 1}."
            )

        expected_start = end + 1

    if expected_start != sentence_count:
        raise ValueError(
            "Chunk plan does not cover all sentences. "
            f"Covered through {expected_start - 1}, "
            f"but final sentence is "
            f"{sentence_count - 1}."
        )

def _validate_chunk_plan(
    *,
    plan: _ChunkPlan,
    sentences: list[_LectureSentence],
    max_chars: int,
) -> None:
    _validate_chunk_structure(
        plan=plan,
        sentence_count=len(sentences),
    )

    for boundary in plan.chunks:
        start = boundary.start_sentence_index
        end = boundary.end_sentence_index

        char_length = _chunk_char_length(
            sentences,
            start,
            end,
        )

        single_sentence_overflow = (
            start == end
            and len(sentences[start].text) > max_chars
        )

        if (
            char_length > max_chars
            and not single_sentence_overflow
        ):
            raise ValueError(
                f"Chunk {start}-{end} is approximately "
                f"{char_length} characters, above "
                f"max_chars={max_chars}."
            )

def _deterministic_size_split(
    *,
    sentences: list[_LectureSentence],
    target_chars: int,
    max_chars: int,
) -> _ChunkPlan:
    """
    Safe fallback splitter.

    Used only when the semantic refinement model fails to produce
    chunks within max_chars.

    It preserves sentence boundaries and source order.
    """
    chunks: list[_ChunkBoundary] = []

    start = 0

    while start < len(sentences):
        # If one individual sentence itself exceeds max_chars,
        # we cannot safely split it without rewriting source text.
        if len(sentences[start].text) > max_chars:
            chunks.append(
                _ChunkBoundary(
                    start_sentence_index=start,
                    end_sentence_index=start,
                )
            )

            start += 1
            continue

        end = start

        while end + 1 < len(sentences):
            next_length = _chunk_char_length(
                sentences,
                start,
                end + 1,
            )

            if next_length > max_chars:
                break

            current_length = _chunk_char_length(
                sentences,
                start,
                end,
            )

            # Once we are around the target size,
            # prefer closing this fallback chunk rather
            # than continuing to grow it.
            if current_length >= target_chars:
                break

            end += 1

        chunks.append(
            _ChunkBoundary(
                start_sentence_index=start,
                end_sentence_index=end,
            )
        )

        start = end + 1

    return _ChunkPlan(
        chunks=chunks
    )

def _refine_oversized_chunks(
    *,
    client: OpenAI,
    model: str,
    plan: _ChunkPlan,
    sentences: list[_LectureSentence],
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> _ChunkPlan:
    """
    Re-examine only first-pass semantic chunks that exceed max_chars.

    The first model call identifies broad semantic regions.
    This second pass finds internal semantic boundaries inside any
    region that is too large.

    If the second model call still produces an invalid size plan,
    fall back to deterministic sentence-boundary splitting.
    """
    refined_chunks: list[_ChunkBoundary] = []

    for boundary in plan.chunks:
        start = boundary.start_sentence_index
        end = boundary.end_sentence_index

        char_length = _chunk_char_length(
            sentences,
            start,
            end,
        )

        # Already acceptable: preserve the original semantic chunk.
        if char_length <= max_chars:
            refined_chunks.append(
                boundary
            )
            continue

        selected_sentences = sentences[
            start:end + 1
        ]

        listing = "\n".join(
            (
                f"[{local_index}] "
                f"(source segment "
                f"{sentence.source_segment_index}, "
                f"sentence "
                f"{sentence.local_sentence_index}, "
                f"{len(sentence.text)} chars) "
                f"{sentence.text}"
            )
            for local_index, sentence
            in enumerate(selected_sentences)
        )

        prompt = f"""
        The earlier semantic chunk below is coherent, but it is too large
        for retrieval.

        Current approximate size:
        {char_length} characters

        Minimum preferred size:
        {min_chars} characters

        Target size:
        {target_chars} characters

        Maximum allowed size:
        {max_chars} characters


        TASK

        Split this material into TWO OR MORE coherent retrieval chunks.

        Use LOCAL sentence indexes shown below.

        Every sentence must appear exactly once.

        Preserve:
        - sentence order
        - semantic continuity
        - question with answer
        - term with explanation
        - quotation with immediate interpretation
        - example with the teaching it directly illustrates

        Prefer a natural semantic boundary.

        A new chunk should preferably begin with a sentence that can be
        understood without needing the previous chunk.

        Avoid boundaries that make the next chunk begin with a strongly
        dependent continuation such as:
        - इसलिए
        - तो
        - लेकिन
        - क्योंकि
        - इसका / उसका / उनकी
        - फिर / तब
        - "उस समय..." when its reference is established only earlier

        These are not absolute rules. The important test is:

        "Would this chunk make sense if retrieved by itself?"

        Prefer splitting AFTER a completed explanation, example, argument,
        or question-answer sequence.

        If the speaker finishes an example and then returns to the main
        teaching, that return is often a strong semantic boundary.

        Do not split merely to make chunks equal in size.

        Do not:
        - rewrite text
        - remove sentences
        - duplicate sentences
        - reorder sentences
        - invent content

        Every resulting chunk should normally be at most
        {max_chars} characters.

        If a single sentence itself exceeds that size, it may remain alone.


        SENTENCES

        {listing}


        Partition LOCAL sentence indexes 0 through
        {len(selected_sentences) - 1}.
        """

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": CHUNKING_INSTRUCTIONS,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format=_ChunkPlan,
        )

        message = completion.choices[0].message

        use_fallback = False

        if message.refusal:
            use_fallback = True

        elif message.parsed is None:
            use_fallback = True

        if use_fallback:
            local_plan = _deterministic_size_split(
                sentences=selected_sentences,
                target_chars=target_chars,
                max_chars=max_chars,
            )

        else:
            local_plan = message.parsed

            try:
                _validate_chunk_plan(
                    plan=local_plan,
                    sentences=selected_sentences,
                    max_chars=max_chars,
                )

            except ValueError:
                # The refinement model still produced
                # an invalid or oversized plan.
                local_plan = _deterministic_size_split(
                    sentences=selected_sentences,
                    target_chars=target_chars,
                    max_chars=max_chars,
                )

        # The refinement model worked with LOCAL indexes.
        # Convert those back into GLOBAL lecture indexes.
        for local_boundary in local_plan.chunks:
            refined_chunks.append(
                _ChunkBoundary(
                    start_sentence_index=(
                        start
                        + local_boundary.start_sentence_index
                    ),
                    end_sentence_index=(
                        start
                        + local_boundary.end_sentence_index
                    ),
                )
            )

    return _ChunkPlan(
        chunks=refined_chunks
    )


def _plan_lecture_chunks(
    *,
    client: OpenAI,
    model: str,
    sentences: list[_LectureSentence],
    min_chars: int,
    target_chars: int,
    max_chars: int,
) -> _ChunkPlan:
    if not sentences:
        raise ValueError(
            "Lecture contains no sentence units."
        )

    # Only skip the model when there is literally one sentence.
    if len(sentences) == 1:
        return _ChunkPlan(
            chunks=[
                _ChunkBoundary(
                    start_sentence_index=0,
                    end_sentence_index=0,
                )
            ]
        )

    listing_parts: list[str] = []

    previous_segment_index: int | None = None

    for sentence in sentences:
        if (
            sentence.source_segment_index
            != previous_segment_index
        ):
            listing_parts.append(
                "\n"
                f"--- SOURCE SEGMENT "
                f"{sentence.source_segment_index} ---"
            )

            previous_segment_index = (
                sentence.source_segment_index
            )

        listing_parts.append(
            f"[{sentence.global_index}] "
            f"(source segment "
            f"{sentence.source_segment_index}, "
            f"sentence "
            f"{sentence.local_sentence_index}, "
            f"{len(sentence.text)} chars) "
            f"{sentence.text}"
        )

    sentence_listing = "\n".join(
        listing_parts
    )

    prompt = f"""
    CHUNK SIZE GUIDANCE

    Minimum preferred size:
    {min_chars} characters

    Target size:
    {target_chars} characters

    Maximum preferred size:
    {max_chars} characters


    IMPORTANT

    The SOURCE SEGMENT markers below are provenance markers only.

    They are NOT required chunk boundaries.

    If a thought begins in one source segment and continues in the next,
    keep it together when semantically appropriate.


    FULL LECTURE SENTENCE STREAM

    {sentence_listing}


    Partition GLOBAL sentence indexes 0 through
    {len(sentences) - 1} into consecutive semantic retrieval chunks.

    Every global sentence must appear exactly once.
    """

    messages = [
        {
            "role": "system",
            "content": CHUNKING_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    plan: _ChunkPlan | None = None
    last_structure_error: ValueError | None = None

    for attempt in range(2):
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=_ChunkPlan,
        )

        message = completion.choices[0].message

        if message.refusal:
            raise RuntimeError(
                f"Chunking model refused: "
                f"{message.refusal}"
            )

        if message.parsed is None:
            raise RuntimeError(
                "Chunking model returned no parsed "
                "chunk plan."
            )

        candidate_plan = message.parsed

        try:
            _validate_chunk_structure(
                plan=candidate_plan,
                sentence_count=len(sentences),
            )

            plan = candidate_plan
            break

        except ValueError as exc:
            last_structure_error = exc

            # Give the model one opportunity to correct
            # its own structural mistake.
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        candidate_plan.model_dump(),
                        ensure_ascii=False,
                    ),
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": f"""
                    The chunk plan above is structurally invalid.

                    Validator error:

                    {exc}

                    Return a COMPLETE corrected chunk plan for GLOBAL sentence indexes
                    0 through {len(sentences) - 1}.

                    IMPORTANT:
                    - every sentence must appear exactly once
                    - no skipped indexes
                    - no duplicated indexes
                    - no gaps
                    - no overlaps
                    - preserve order

                    Do not return only the corrected region.
                    Return the complete plan.
                    """,
                }
            )


    if plan is None:
        # If the model fails twice, preserve every sentence
        # using the deterministic safety fallback rather than
        # losing transcript content or terminating the pipeline.
        plan = _deterministic_size_split(
            sentences=sentences,
            target_chars=target_chars,
            max_chars=max_chars,
        )

        _validate_chunk_structure(
            plan=plan,
            sentence_count=len(sentences),
        )

    # A first-pass semantic region is allowed to be
    # oversized. Oversized regions are examined again
    # with a narrower semantic splitting task.
    plan = _refine_oversized_chunks(
        client=client,
        model=model,
        plan=plan,
        sentences=sentences,
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
    )

    # Now the final plan must satisfy everything,
    # including the maximum chunk size.
    _validate_chunk_plan(
        plan=plan,
        sentences=sentences,
        max_chars=max_chars,
    )

    return plan


def _build_source_fragments(
    *,
    selected_sentences: list[_LectureSentence],
    segments_by_index: dict,
) -> list[RagSourceFragment]:
    fragments: list[RagSourceFragment] = []

    position = 0

    while position < len(selected_sentences):
        first = selected_sentences[position]
        segment_index = (
            first.source_segment_index
        )

        end_position = position

        while (
            end_position + 1
            < len(selected_sentences)
            and selected_sentences[
                end_position + 1
            ].source_segment_index
            == segment_index
        ):
            end_position += 1

        last = selected_sentences[end_position]

        segment = segments_by_index[
            segment_index
        ]

        fragment_text = segment.cleaned_text[
            first.char_start:last.char_end
        ]

        fragments.append(
            RagSourceFragment(
                source_segment_index=segment_index,
                source_review_decision=(
                    segment.source_review_decision
                ),

                source_sentence_start=(
                    first.local_sentence_index
                ),
                source_sentence_end=(
                    last.local_sentence_index
                ),

                source_char_start=(
                    first.char_start
                ),
                source_char_end=(
                    last.char_end
                ),

                start_seconds=(
                    segment.start_seconds
                ),
                end_seconds=(
                    segment.end_seconds
                ),

                text=fragment_text,
            )
        )

        position = end_position + 1

    return fragments


def _dedupe_strings(
    items: list[str],
) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output


def chunk_lecture(
    lecture_id: str,
    settings: Settings,
    *,
    force: bool = False,
) -> RagChunkingResult:
    lecture_dir = settings.data_dir / lecture_id

    if not lecture_dir.exists():
        raise FileNotFoundError(
            f"Lecture folder not found: "
            f"{lecture_dir}"
        )

    enriched_json_path = (
        lecture_dir
        / "enriched_transcript.json"
    )

    if not enriched_json_path.exists():
        raise FileNotFoundError(
            f"Enriched transcript not found: "
            f"{enriched_json_path}. "
            f"Run 'adhyatmik enrich --lecture-id "
            f"{lecture_id}' first."
        )

    rag_chunks_path = (
        lecture_dir / "rag_chunks.json"
    )

    if rag_chunks_path.exists() and not force:
        raise FileExistsError(
            f"RAG chunks already exist for "
            f"'{lecture_id}': "
            f"{rag_chunks_path}. "
            "Use --force only if you intentionally "
            "want to regenerate them."
        )

    payload = _read_json(
        enriched_json_path
    )

    enriched = (
        EnrichmentResult.model_validate(
            payload
        )
    )

    if enriched.lecture_id != lecture_id:
        raise ValueError(
            "enriched_transcript.json belongs "
            "to a different lecture. "
            f"Expected '{lecture_id}', found "
            f"'{enriched.lecture_id}'."
        )

    segments_by_index = {
        segment.index: segment
        for segment in enriched.segments
    }

    lecture_sentences = (
        _build_lecture_sentences(
            enriched
        )
    )

    client = OpenAI(
        api_key=settings.openai_api_key
    )

    plan = _plan_lecture_chunks(
        client=client,
        model=settings.rag_chunking_model,
        sentences=lecture_sentences,
        min_chars=settings.rag_chunk_min_chars,
        target_chars=(
            settings.rag_chunk_target_chars
        ),
        max_chars=settings.rag_chunk_max_chars,
    )

    rag_chunks: list[RagChunk] = []

    for chunk_number, boundary in enumerate(
        plan.chunks
    ):
        selected_sentences = (
            lecture_sentences[
                boundary.start_sentence_index:
                boundary.end_sentence_index + 1
            ]
        )

        fragments = _build_source_fragments(
            selected_sentences=(
                selected_sentences
            ),
            segments_by_index=(
                segments_by_index
            ),
        )

        # Each fragment is copied exactly from
        # human-reviewed cleaned_text.
        #
        # Newlines are only separators between
        # separate source segments.
        chunk_text = "\n".join(
            fragment.text
            for fragment in fragments
        )

        contributing_segment_indices = (
            _dedupe_strings(
                [
                    str(
                        fragment
                        .source_segment_index
                    )
                    for fragment in fragments
                ]
            )
        )

        contributing_segments = [
            segments_by_index[int(index)]
            for index
            in contributing_segment_indices
        ]

        parent_titles = _dedupe_strings(
            [
                segment.title
                for segment
                in contributing_segments
            ]
        )

        parent_topics = _dedupe_strings(
            [
                topic
                for segment
                in contributing_segments
                for topic in segment.topics
            ]
        )

        parent_concepts = _dedupe_strings(
            [
                concept.canonical_term
                for segment
                in contributing_segments
                for concept in segment.concepts
            ]
        )

        parent_scriptural_references = (
            _dedupe_strings(
                [
                    reference
                    for segment
                    in contributing_segments
                    for reference
                    in segment.scriptural_references
                ]
            )
        )

        parent_likely_questions = (
            _dedupe_strings(
                [
                    question
                    for segment
                    in contributing_segments
                    for question
                    in segment.likely_questions
                ]
            )
        )

        seen_entities: set[
            tuple[str, str]
        ] = set()

        parent_named_entities = []

        for segment in contributing_segments:
            for entity in (
                segment.named_entities
            ):
                key = (
                    entity.name,
                    entity.entity_type,
                )

                if key in seen_entities:
                    continue

                seen_entities.add(key)
                parent_named_entities.append(
                    entity
                )

        timestamp_precision = (
            "source_segment"
            if len(fragments) == 1
            else "source_segments"
        )

        rag_chunks.append(
            RagChunk(
                chunk_id=(
                    f"{lecture_id}:"
                    f"c{chunk_number:04d}"
                ),
                lecture_id=lecture_id,

                source_fragments=fragments,

                start_seconds=(
                    fragments[0].start_seconds
                ),
                end_seconds=(
                    fragments[-1].end_seconds
                ),

                timestamp_precision=(
                    timestamp_precision
                ),

                text=chunk_text,

                parent_titles=parent_titles,
                parent_topics=parent_topics,
                parent_concepts=parent_concepts,
                parent_scriptural_references=(
                    parent_scriptural_references
                ),
                parent_named_entities=(
                    parent_named_entities
                ),
                parent_likely_questions=(
                    parent_likely_questions
                ),
            )
        )

    result = RagChunkingResult(
        lecture_id=lecture_id,

        source_enriched_json=str(
            enriched_json_path
        ),

        chunking_model=(
            settings.rag_chunking_model
        ),
        target_chars=(
            settings.rag_chunk_target_chars
        ),
        max_chars=(
            settings.rag_chunk_max_chars
        ),

        source_segment_count=len(
            enriched.segments
        ),
        source_sentence_count=len(
            lecture_sentences
        ),

        rag_chunks_path=str(
            rag_chunks_path
        ),

        chunks=rag_chunks,
    )

    _write_json(
        rag_chunks_path,
        result.model_dump(),
    )

    return result