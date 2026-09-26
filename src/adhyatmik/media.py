from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary '{name}' was not found. Install ffmpeg/ffprobe first.")


def probe_duration(path: Path) -> float:
    _require_binary("ffprobe")
    proc = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    return float(payload["format"]["duration"])


def download_youtube(url: str, workdir: Path) -> tuple[Path, dict]:
    from yt_dlp import YoutubeDL
    workdir.mkdir(parents=True, exist_ok=True)
    output_template = str(workdir / "source.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        source_path = Path(ydl.prepare_filename(info))
    return source_path, info


def extract_audio(source_path: Path, audio_path: Path) -> Path:
    _require_binary("ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(source_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path


def split_audio(audio_path: Path, chunks_dir: Path, chunk_seconds: int) -> list[Path]:
    _require_binary("ffmpeg")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunks_dir / "chunk_%04d.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    return sorted(chunks_dir.glob("chunk_*.mp3"))
