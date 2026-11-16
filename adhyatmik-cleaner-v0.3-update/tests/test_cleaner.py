import json
from pathlib import Path

from adhyatmik.cleaner import _load_raw_result


def test_load_raw_result(tmp_path: Path):
    path = tmp_path / "raw_transcript.json"
    path.write_text(
        json.dumps(
            {
                "lecture_id": "lecture-1",
                "title": "Lecture 1",
                "source": "example.mp4",
                "duration_seconds": 300,
                "transcript_path": "raw_transcript.txt",
                "json_path": "raw_transcript.json",
                "segments": [
                    {
                        "index": 0,
                        "start_seconds": 0,
                        "end_seconds": 300,
                        "text": "विशाद योग कैसे हो सकता है",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = _load_raw_result(path)
    assert result.lecture_id == "lecture-1"
    assert result.segments[0].text == "विशाद योग कैसे हो सकता है"
