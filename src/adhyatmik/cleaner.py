from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .models import CleanedSegment, CleaningChange, CleaningResult, TranscriptResult
from .pipeline import format_timestamp


class _ModelChange(BaseModel):
    original: str
    corrected: str
    category: str
    reason: str
    confidence: str


class _ModelCleanedSegment(BaseModel):
    cleaned_text: str
    changes: list[_ModelChange] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)


CLEANER_INSTRUCTIONS = """You are the conservative transcript-cleaning stage for the Adhyatmik project.
The input is an automatic Hindi lecture transcript that may contain Sanskrit, Vedanta vocabulary, English code-switching, proper names, quotations, and shlokas.

Your job is transcription correction, NOT rewriting, summarizing, explaining, translating, or improving the speaker's teaching.

Rules:
1. Preserve the speaker's meaning, order, examples, repetitions, questions, and style.
2. Correct only high-confidence ASR errors, obvious spelling errors, punctuation, sentence boundaries, and clearly misrecognized proper/Sanskrit/Vedanta terms.
3. Never add philosophical content that is not present in the raw transcript.
4. Never delete a meaningful sentence merely because it sounds repetitive or informal.
5. Preserve Hindi/English code-switching as spoken.
6. Treat the supplied glossary as recognition hints, not as words that must be inserted.
7. For Sanskrit quotations or shlokas, correct only when the intended wording is sufficiently clear from the transcript/context. If uncertain, preserve the raw wording and add a review note.
8. If a possible correction could materially change doctrinal/philosophical meaning, prefer preserving the raw wording and flagging it for human review.
9. Do not include timestamps in cleaned_text; timestamps are preserved by the application.
10. Each change entry must describe an actual textual change. Do not log punctuation-only changes unless they materially improve sentence boundaries.

Change categories should be one of: spelling, asr_term, proper_name, sanskrit, punctuation, sentence_boundary, other.
Confidence should be one of: high, medium, low.
"""


def _load_glossary(path: Path) -> str:
    if not path.exists():
        return ""
    terms = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    terms = [term for term in terms if term and not term.startswith("#")]
    return "\n".join(terms[:400])


def _load_raw_result(raw_json_path: Path) -> TranscriptResult:
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    return TranscriptResult.model_validate(payload)


def _clean_one_segment(
    *,
    client: OpenAI,
    model: str,
    raw_text: str,
    previous_context: str,
    next_context: str,
    glossary: str,
) -> _ModelCleanedSegment:
    context = f"""VEDANTA/SANSKRIT GLOSSARY HINTS:
{glossary or '(none)'}

PREVIOUS SEGMENT TAIL (context only; do not output it):
{previous_context or '(none)'}

CURRENT RAW SEGMENT (clean ONLY this text):
{raw_text}

NEXT SEGMENT HEAD (context only; do not output it):
{next_context or '(none)'}
"""

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CLEANER_INSTRUCTIONS},
            {"role": "user", "content": context},
        ],
        response_format=_ModelCleanedSegment,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Cleaner model refused segment: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Cleaner model returned no parsed structured output")
    return message.parsed


def clean_lecture(lecture_id: str, settings: Settings, *, force: bool = False) -> CleaningResult:
    lecture_dir = settings.data_dir / lecture_id
    raw_json_path = lecture_dir / "raw_transcript.json"
    if not raw_json_path.exists():
        raise FileNotFoundError(f"Raw transcript JSON not found: {raw_json_path}")

    clean_txt_path = lecture_dir / "proposed_clean_transcript.txt"
    clean_json_path = lecture_dir / "proposed_clean_transcript.json"
    report_path = lecture_dir / "proposed_cleaning_report.json"

    if not force and (clean_txt_path.exists() or clean_json_path.exists() or report_path.exists()):
        raise FileExistsError(
            f"Proposed cleaner outputs already exist for '{lecture_id}'. Use --force only if you intentionally want to regenerate the AI proposal."
        )

    raw = _load_raw_result(raw_json_path)
    glossary = _load_glossary(settings.glossary_path)
    client = OpenAI(api_key=settings.openai_api_key)

    cleaned_segments: list[CleanedSegment] = []
    all_changes: list[CleaningChange] = []
    review_notes: list[str] = []

    for index, segment in enumerate(raw.segments):
        prev_text = raw.segments[index - 1].text[-settings.cleaner_context_chars :] if index > 0 else ""
        next_text = raw.segments[index + 1].text[: settings.cleaner_context_chars] if index + 1 < len(raw.segments) else ""

        parsed = _clean_one_segment(
            client=client,
            model=settings.cleaning_model,
            raw_text=segment.text,
            previous_context=prev_text,
            next_context=next_text,
            glossary=glossary,
        )

        changes = [
            CleaningChange(
                segment_index=segment.index,
                original=change.original,
                corrected=change.corrected,
                category=change.category,
                reason=change.reason,
                confidence=change.confidence,
            )
            for change in parsed.changes
        ]
        all_changes.extend(changes)
        review_notes.extend(
            f"Segment {segment.index}: {note}" for note in parsed.review_notes if note.strip()
        )

        cleaned_segments.append(
            CleanedSegment(
                index=segment.index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                raw_text=segment.text,
                cleaned_text=parsed.cleaned_text.strip(),
                changes=changes,
                review_notes=parsed.review_notes,
            )
        )

    text_blocks = [
        f"[{format_timestamp(seg.start_seconds)} - {format_timestamp(seg.end_seconds)}]\n{seg.cleaned_text}"
        for seg in cleaned_segments
    ]
    clean_txt_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")

    result = CleaningResult(
        lecture_id=raw.lecture_id,
        source_raw_json=str(raw_json_path),
        cleaning_model=settings.cleaning_model,
        clean_transcript_path=str(clean_txt_path),
        clean_json_path=str(clean_json_path),
        report_path=str(report_path),
        segments=cleaned_segments,
        total_changes=len(all_changes),
        review_notes=review_notes,
    )
    clean_json_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "lecture_id": raw.lecture_id,
        "cleaning_model": settings.cleaning_model,
        "total_segments": len(cleaned_segments),
        "total_changes": len(all_changes),
        "review_notes": review_notes,
        "changes": [change.model_dump() for change in all_changes],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def clean_all_lectures(settings: Settings, *, force: bool = False) -> list[CleaningResult]:
    settings.ensure_dirs()
    results: list[CleaningResult] = []
    for lecture_dir in sorted(settings.data_dir.iterdir()):
        if not lecture_dir.is_dir() or not (lecture_dir / "raw_transcript.json").exists():
            continue
        if not force and (
            (lecture_dir / "proposed_clean_transcript.json").exists()
            or (lecture_dir / "clean_transcript.json").exists()
        ):
            continue
        results.append(clean_lecture(lecture_dir.name, settings, force=force))
    return results
