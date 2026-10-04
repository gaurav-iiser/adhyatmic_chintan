from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings
# from .models import EnrichedConcept, EnrichedSegment, EnrichmentResult
from .models import (
    EnrichedConcept,
    EnrichedNamedEntity,
    EnrichedSegment,
    EnrichmentResult,
)


class _ModelSegmentEnrichment(BaseModel):
    title: str
    summary: str

    topics: list[str] = Field(
        default_factory=list,
        max_length=5,
    )

    concepts: list[EnrichedConcept] = Field(
        default_factory=list,
        max_length=5,
    )

    scriptural_references: list[str] = Field(
        default_factory=list,
        max_length=5,
    )

    named_entities: list[EnrichedNamedEntity] = Field(
        default_factory=list,
        max_length=10,
    )

    likely_questions: list[str] = Field(
        default_factory=list,
        max_length=5,
    )


ENRICHMENT_INSTRUCTIONS = """
You are the semantic-enrichment stage for the Adhyatmik project.

The input comes from a HUMAN-REVIEWED transcript of a spiritual discourse,
primarily involving Bhagavad Gita, Vedanta, Sanskrit terminology, Hindi,
and occasional English.

The human-reviewed transcript text is authoritative.

Your task is ONLY to create metadata that will later help a retrieval system
find the correct teaching.

You must NOT:
- rewrite the transcript
- correct the transcript
- translate the transcript
- add missing teaching
- improve the speaker's argument
- infer doctrine that is not supported by the supplied text
- invent quotations, verse numbers, chapter numbers, or scriptural sources

The previous and next segment excerpts are supplied only to resolve
continuity at segment boundaries.

They may help identify what an opening pronoun or incomplete sentence
refers to, but they must not contribute independent metadata.

Create metadata for the CURRENT SEGMENT only.

Return:

1. title
   A short, neutral description of the main teaching or discussion.

2. summary
   A concise description of what the speaker says in this segment.
   Describe the speaker's teaching rather than presenting your own
   philosophical conclusion.
   The summary should compress what the speaker says, but must not make 
   the discourse more explicit, systematic, causal, or polished than the
   source itself.

When two statements occur next to one another but their relationship
is uncertain, keep them separate rather than joining them with
"because", "therefore", "इसलिए", or equivalent causal wording.

3. topics
   Important themes discussed in this segment.
   Prefer specific useful topics over generic labels such as "spirituality".

4. concepts
   Important Sanskrit, Vedanta, philosophical, or technical concepts.

   For each concept:
   - canonical_term: the clearest standard name for the concept
   - aliases: useful genuine spelling/transliteration/language variants
   - contextual_meaning: what the term means specifically in this segment

   Do not claim two terms are equivalent unless that relationship is sound.

5. scriptural_references
   Include only references actually stated or clearly identifiable from
   the supplied transcript.

   A general reference such as "Bhagavad Gita" or
   "Bhagavad Gita, Chapter 1" is acceptable when supported.

   Never invent a verse number.

6. named_entities

    Identify explicitly mentioned named people, traditional figures,
    groups, and places that are useful for retrieval.

    For every entity provide:

    - name
    - entity_type: person, group, or place

    Examples:

    Krishna -> person
    Arjuna -> person
    Pandavas -> group
    Kauravas -> group
    Hastinapur -> place

    Do not include generic descriptions such as:
    - वक्ता
    - भाष्यकार
    - राजा
    - गुरु
    - भगवान

    unless the current segment clearly identifies which specific named
    figure that title refers to.

    Do not infer an identity that the transcript itself does not establish.

7. likely_questions
   Generate realistic questions for which this segment would be useful
   evidence or explanation.

   Questions are retrieval aids and must remain grounded in the segment.
   They may naturally be in Hindi, Hinglish, or English.

GROUNDING RULES

Every metadata item must be supported by the CURRENT SEGMENT.

The previous and next excerpts may be used only to resolve an incomplete
sentence, pronoun, or reference that crosses a segment boundary.

Do not use neighboring context to introduce a separate topic, teaching,
person, quotation, or conclusion that is not actually part of the current
segment.

Do not infer a causal relationship merely because two statements,
examples, or events occur next to each other.

Words such as:
- because
- therefore
- caused by
- resulting from
- इसलिए
- इस कारण
- फलस्वरूप

should appear in metadata only when that relationship is stated or
clearly developed by the speaker.

When uncertain, describe the statements separately rather than joining
them into a causal claim.

EXTERNAL KNOWLEDGE

Do not supplement the transcript using your own knowledge of
Vedanta, Sanskrit, Bhagavad Gita, Mahabharata, Indian philosophy,
history, or traditional definitions.

Even if you know the conventional meaning of a term, include that
meaning only when the CURRENT SEGMENT states or develops it.

If a technical term is mentioned but not defined, contextual_meaning
should describe only how the term is being used or positioned in this
segment.

For example:

If the speaker merely says that Gita presents
"प्रवृत्ति-लक्षणात्मक धर्म" and "निवृत्ति-लक्षणात्मक धर्म",

do not independently define प्रवृत्ति as action and निवृत्ति as
renunciation unless the speaker provides that explanation here.

It is acceptable to say:

"गीता में बताए गए धर्म के दो प्रकारों में से एक"

rather than supplying an external philosophical definition.

CONTEXTUAL MEANING

contextual_meaning must explain how the CURRENT SEGMENT uses the term.

Prefer vocabulary actually used by the speaker.

Do not introduce a new philosophical label, category, synonym, or
technical term merely to make the explanation sound more sophisticated.

If a simple literal description is sufficient, use it.

VERY IMPORTANT:

Distinguish between:
- a question the speaker raises
- an explanation the speaker gives
- a conclusion the speaker reaches

If the speaker merely raises a question, metadata must say that the
question is being raised. Do not describe the segment as though it
contains the answer.

For example:

If the speaker says:
"धर्मक्षेत्र क्यों कहा गया? इसमें धर्म क्या है?"

acceptable metadata would be:
"धर्मक्षेत्र कहे जाने का प्रश्न"

Do NOT write:
"धर्मक्षेत्र का सांकेतिक अर्थ"

unless the current segment actually explains that meaning.

Do not infer:
- motives
- causal relationships
- symbolic meanings
- doctrinal conclusions
- philosophical significance

unless the speaker states or clearly develops them.

UNSTATED REASONS

Never invent or supply the reason, justification, mechanism,
motivation, or rationale behind a statement unless the speaker
actually provides it.

This applies even when the reason seems obvious, conventional,
historically likely, or philosophically correct.

For example:

If the speaker states a traditional rule about who may become king
but does not explain WHY that rule exists, describe the rule only.

Do not generate a rationale for it.

Prefer:

"वक्ता यह नियम बताते हैं कि..."

over:

"यह नियम इसलिए है क्योंकि..."

CONCEPT SELECTION

The concepts field is intentionally selective.

Include primarily:
- Sanskrit philosophical terms
- Vedantic concepts
- scriptural technical terms
- terms whose meaning is being explained or used importantly

Do not treat ordinary nouns, events, people, quotations, or general
themes as concepts merely because they are important.

For example, "युद्ध" should normally be a topic rather than a concept
unless the speaker is specifically explaining युद्ध as a technical or
philosophical concept.

A mantra, verse, quotation, prayer, or scriptural passage is NOT
automatically a concept.

If it is merely quoted or recited, place it only in
scriptural_references.

Include it in concepts only when the speaker actually discusses,
defines, interprets, or develops its meaning.

A concept should normally be a TERM or compact technical expression,
not an entire sentence, verse, quotation, mantra, prayer, or message.

Do NOT put a full quotation in concepts merely because it is important.

Examples:

GOOD concepts:
- धर्मक्षेत्र
- समवेताः युयुत्सवः
- निःश्रेयस
- प्रवृत्ति-लक्षणात्मक धर्म

NOT concepts unless their meaning is explicitly analysed:
- an entire Sanskrit verse
- Kunti's full quoted message
- a mantra recited at the end of the lecture

Such material belongs in scriptural_references.

SOURCE ATTRIBUTION

Do not identify the external source of a quotation from memory.

If the current segment says that a quotation is from the Bhagavad Gita,
Mahabharata, Upanishad, Gita-bhashya, etc., record that attribution.

If the segment quotes words without explicitly identifying their source,
record the quotation without supplying the source yourself.

Do not add descriptive labels such as:
- sacred
- holy
- traditional
- famous
- well-known

unless the speaker uses or establishes that characterization.

QUESTION GENERATION

A likely question must be substantially answerable from the CURRENT
SEGMENT.

Do not generate a question merely because something is mentioned.

If a mantra is only recited but its significance is not explained,
do not generate a question asking for the significance of that mantra.

Keep metadata focused:
- preferably 3 to 5 topics
- preferably 0 to 5 concepts
- preferably 2 to 5 likely questions

Prefer fewer high-confidence metadata items over many speculative ones.

Be conservative.

If a category has no well-supported items, return an empty list rather
than inventing something.
"""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_glossary(path: Path) -> str:
    if not path.exists():
        return ""

    terms = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    terms = [
        term
        for term in terms
        if term and not term.startswith("#")
    ]

    return "\n".join(terms[:400])


def _load_reviewed_transcript(
    clean_json_path: Path,
    lecture_id: str,
) -> dict[str, Any]:
    payload = _read_json(clean_json_path)

    if payload.get("lecture_id") != lecture_id:
        raise ValueError(
            "clean_transcript.json belongs to a different lecture. "
            f"Expected '{lecture_id}', found "
            f"'{payload.get('lecture_id')}'."
        )

    if payload.get("human_review_status") != "approved":
        raise ValueError(
            "Transcript has not completed human review. "
            "Semantic enrichment is allowed only for an approved "
            "clean_transcript.json."
        )

    segments = payload.get("segments")

    if not isinstance(segments, list) or not segments:
        raise ValueError(
            "Approved clean transcript contains no segments."
        )

    previous_index: int | None = None
    previous_end: float | None = None

    for segment in segments:
        if "index" not in segment:
            raise ValueError("A reviewed segment is missing its index.")

        index = int(segment["index"])

        if previous_index is not None and index <= previous_index:
            raise ValueError(
                "Reviewed segment indexes are not strictly increasing."
            )

        text = str(segment.get("cleaned_text", "")).strip()

        if not text:
            raise ValueError(
                f"Reviewed segment {index} has empty cleaned_text."
            )

        start_seconds = float(segment.get("start_seconds", 0))
        end_seconds = float(segment.get("end_seconds", 0))

        if end_seconds < start_seconds:
            raise ValueError(
                f"Reviewed segment {index} ends before it starts."
            )

        if (
            previous_end is not None
            and start_seconds < previous_end
        ):
            raise ValueError(
                f"Reviewed segment {index} overlaps the previous "
                "segment in time."
            )

        previous_index = index
        previous_end = end_seconds

    return payload


def _enrich_one_segment(
    *,
    client: OpenAI,
    model: str,
    current_text: str,
    previous_context: str,
    next_context: str,
    glossary: str,
) -> _ModelSegmentEnrichment:
    prompt = f"""
VEDANTA / SANSKRIT GLOSSARY HINTS
These are recognition hints only. Do not force them into the metadata.

{glossary or "(none)"}


PREVIOUS REVIEWED SEGMENT TAIL
Context only. Do not create metadata for this text.

{previous_context or "(none)"}


CURRENT HUMAN-REVIEWED SEGMENT
Create metadata ONLY for this text.

{current_text}


NEXT REVIEWED SEGMENT HEAD
Context only. Do not create metadata for this text.

{next_context or "(none)"}
"""

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": ENRICHMENT_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format=_ModelSegmentEnrichment,
    )

    message = completion.choices[0].message

    if message.refusal:
        raise RuntimeError(
            f"Enrichment model refused segment: {message.refusal}"
        )

    if message.parsed is None:
        raise RuntimeError(
            "Enrichment model returned no parsed structured output."
        )

    return message.parsed


def enrich_lecture(
    lecture_id: str,
    settings: Settings,
    *,
    force: bool = False,
) -> EnrichmentResult:
    lecture_dir = settings.data_dir / lecture_id

    if not lecture_dir.exists():
        raise FileNotFoundError(
            f"Lecture folder not found: {lecture_dir}"
        )

    clean_json_path = lecture_dir / "clean_transcript.json"

    if not clean_json_path.exists():
        raise FileNotFoundError(
            f"Human-reviewed transcript not found: {clean_json_path}. "
            f"Run 'adhyatmik review --lecture-id {lecture_id}' first."
        )

    enriched_json_path = lecture_dir / "enriched_transcript.json"

    if enriched_json_path.exists() and not force:
        raise FileExistsError(
            f"Enriched transcript already exists for '{lecture_id}': "
            f"{enriched_json_path}. "
            "Use --force only if you intentionally want to regenerate it."
        )

    reviewed = _load_reviewed_transcript(
        clean_json_path,
        lecture_id,
    )

    source_segments = reviewed["segments"]

    glossary = _load_glossary(settings.glossary_path)

    client = OpenAI(api_key=settings.openai_api_key)

    enriched_segments: list[EnrichedSegment] = []

    for position, segment in enumerate(source_segments):
        current_text = str(segment["cleaned_text"]).strip()

        if position > 0:
            previous_text = str(
                source_segments[position - 1]["cleaned_text"]
            )
            previous_context = previous_text[
                -settings.enrichment_context_chars:
            ]
        else:
            previous_context = ""

        if position + 1 < len(source_segments):
            next_text = str(
                source_segments[position + 1]["cleaned_text"]
            )
            next_context = next_text[
                :settings.enrichment_context_chars
            ]
        else:
            next_context = ""

        metadata = _enrich_one_segment(
            client=client,
            model=settings.enrichment_model,
            current_text=current_text,
            previous_context=previous_context,
            next_context=next_context,
            glossary=glossary,
        )

        enriched_segments.append(
            EnrichedSegment(
                index=int(segment["index"]),
                start_seconds=float(segment["start_seconds"]),
                end_seconds=float(segment["end_seconds"]),

                # IMPORTANT:
                # copied directly from the approved transcript;
                # never generated by the enrichment model.
                cleaned_text=current_text,

                source_review_decision=str(
                    segment.get(
                        "review_decision",
                        "unknown",
                    )
                ),

                title=metadata.title.strip(),
                summary=metadata.summary.strip(),
                topics=[
                    item.strip()
                    for item in metadata.topics
                    if item.strip()
                ],
                concepts=metadata.concepts,
                scriptural_references=[
                    item.strip()
                    for item in metadata.scriptural_references
                    if item.strip()
                ],
                named_entities=metadata.named_entities,
                likely_questions=[
                    item.strip()
                    for item in metadata.likely_questions
                    if item.strip()
                ],
            )
        )

    result = EnrichmentResult(
        lecture_id=lecture_id,
        source_clean_json=str(clean_json_path),
        source_human_review_completed_at=reviewed.get(
            "human_review_completed_at"
        ),
        enrichment_model=settings.enrichment_model,
        enriched_json_path=str(enriched_json_path),
        segments=enriched_segments,
    )

    _write_json(
        enriched_json_path,
        result.model_dump(),
    )

    return result