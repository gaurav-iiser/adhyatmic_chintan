from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from .config import Settings
from .media import download_youtube, extract_audio, probe_duration, split_audio
from .models import TranscriptResult
from .transcribe import transcribe_chunks


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return value or "lecture"


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_result(
    *,
    output_dir: Path,
    lecture_id: str,
    title: str | None,
    source: str,
    duration: float,
    segments,
) -> TranscriptResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "raw_transcript.txt"
    json_path = output_dir / "raw_transcript.json"

    text_blocks = []
    for seg in segments:
        text_blocks.append(
            f"[{format_timestamp(seg.start_seconds)} - {format_timestamp(seg.end_seconds)}]\n{seg.text}"
        )
    txt_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")

    result = TranscriptResult(
        lecture_id=lecture_id,
        title=title,
        source=source,
        duration_seconds=duration,
        transcript_path=str(txt_path),
        json_path=str(json_path),
        segments=segments,
    )
    json_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def process_youtube(url: str, lecture_id: str, settings: Settings) -> TranscriptResult:
    settings.ensure_dirs()
    safe_id = slugify(lecture_id)
    output_dir = settings.data_dir / safe_id

    with tempfile.TemporaryDirectory(prefix="adhyatmik-") as tmp:
        workdir = Path(tmp)
        source_path, info = download_youtube(url, workdir / "download")
        audio_path = extract_audio(source_path, workdir / "audio.mp3")
        duration = probe_duration(audio_path)
        chunks = split_audio(audio_path, workdir / "chunks", settings.chunk_seconds)
        segments = transcribe_chunks(
            chunks,
            model=settings.transcription_model,
            api_key=settings.openai_api_key,
            chunk_seconds=settings.chunk_seconds,
            total_duration=duration,
            glossary_path=settings.glossary_path,
        )
        return _write_result(
            output_dir=output_dir,
            lecture_id=safe_id,
            title=info.get("title"),
            source=url,
            duration=duration,
            segments=segments,
        )


def process_file(source_path: Path, lecture_id: str, settings: Settings) -> TranscriptResult:
    settings.ensure_dirs()
    safe_id = slugify(lecture_id)
    output_dir = settings.data_dir / safe_id

    with tempfile.TemporaryDirectory(prefix="adhyatmik-") as tmp:
        workdir = Path(tmp)
        local_source = workdir / source_path.name
        shutil.copy2(source_path, local_source)
        audio_path = extract_audio(local_source, workdir / "audio.mp3")
        duration = probe_duration(audio_path)
        chunks = split_audio(audio_path, workdir / "chunks", settings.chunk_seconds)
        segments = transcribe_chunks(
            chunks,
            model=settings.transcription_model,
            api_key=settings.openai_api_key,
            chunk_seconds=settings.chunk_seconds,
            total_duration=duration,
            glossary_path=settings.glossary_path,
        )
        return _write_result(
            output_dir=output_dir,
            lecture_id=safe_id,
            title=source_path.stem,
            source=str(source_path),
            duration=duration,
            segments=segments,
        )
