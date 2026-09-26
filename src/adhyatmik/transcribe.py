from __future__ import annotations

from pathlib import Path

from .models import TranscriptSegment


def load_keywords(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(previous_text: str = "") -> str:
    base = (
        "यह भगवद्गीता और वेदान्त पर हिन्दी आध्यात्मिक प्रवचन है। "
        "जहाँ सम्भव हो देवनागरी लिपि रखें। संस्कृत श्लोक, नाम और वेदान्तिक शब्दों को सावधानी से लिखें। "
        "जो बोला गया है उसी को लिखें; सारांश या व्याख्या न जोड़ें।"
    )
    if previous_text:
        tail = previous_text[-1200:]
        base += f"\nपिछले अंश का संदर्भ (केवल निरंतरता के लिए):\n{tail}"
    return base


def transcribe_chunks(
    chunks: list[Path],
    *,
    model: str,
    api_key: str,
    chunk_seconds: int,
    total_duration: float,
    glossary_path: Path,
) -> list[TranscriptSegment]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    keywords = load_keywords(glossary_path)
    segments: list[TranscriptSegment] = []
    previous_text = ""

    for index, chunk in enumerate(chunks):
        with chunk.open("rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                prompt=build_prompt(previous_text),
                extra_body={
                    "keywords": keywords,
                    "languages": ["hi", "en"],
                },
            )

        text = transcript.text.strip()
        start = index * chunk_seconds
        end = min((index + 1) * chunk_seconds, total_duration)
        segments.append(
            TranscriptSegment(
                index=index,
                start_seconds=start,
                end_seconds=end,
                text=text,
            )
        )
        previous_text = text

    return segments
