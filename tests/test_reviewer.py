import json
from pathlib import Path

from adhyatmik.reviewer import (
    finalize_review,
    load_or_create_review_state,
    migrate_v03_cleaner_outputs,
    pending_segment_indexes,
)


def _proposal():
    return {
        "lecture_id": "lecture1.1",
        "source_raw_json": "data/lecture1.1/raw_transcript.json",
        "cleaning_model": "test-model",
        "segments": [
            {
                "index": 0,
                "start_seconds": 0,
                "end_seconds": 300,
                "raw_text": "विशाद योग कैसे हो सकता है",
                "cleaned_text": "विषाद योग कैसे हो सकता है?",
                "changes": [
                    {
                        "segment_index": 0,
                        "original": "विशाद",
                        "corrected": "विषाद",
                        "category": "asr_term",
                        "reason": "term correction",
                        "confidence": "high",
                    }
                ],
                "review_notes": [],
            },
            {
                "index": 1,
                "start_seconds": 300,
                "end_seconds": 600,
                "raw_text": "अगला वाक्य",
                "cleaned_text": "अगला वाक्य",
                "changes": [],
                "review_notes": [],
            },
        ],
    }


def test_review_state_auto_accepts_unchanged_and_finalizes(tmp_path: Path):
    proposal = _proposal()
    state = load_or_create_review_state(tmp_path, "lecture1.1", proposal)
    assert pending_segment_indexes(proposal, state) == [0]
    assert state["decisions"]["1"]["decision"] == "auto_accept_unchanged"

    state["decisions"]["0"] = {
        "decision": "accepted",
        "final_text": "विषाद योग कैसे हो सकता है?",
        "reviewer_note": "",
        "reviewed_at": "2026-09-28T00:00:00+00:00",
    }
    report = finalize_review(tmp_path, proposal, state)

    assert report["status"] == "approved"
    assert (tmp_path / "clean_transcript.txt").exists()
    payload = json.loads((tmp_path / "clean_transcript.json").read_text(encoding="utf-8"))
    assert payload["human_review_status"] == "approved"
    assert payload["segments"][0]["cleaned_text"] == "विषाद योग कैसे हो सकता है?"
    assert payload["segments"][1]["review_decision"] == "auto_accept_unchanged"


def test_migrate_v03_outputs(tmp_path: Path):
    (tmp_path / "clean_transcript.txt").write_text("proposal", encoding="utf-8")
    (tmp_path / "clean_transcript.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cleaning_report.json").write_text("{}", encoding="utf-8")

    assert migrate_v03_cleaner_outputs(tmp_path) is True
    assert not (tmp_path / "clean_transcript.json").exists()
    assert (tmp_path / "proposed_clean_transcript.json").exists()
    assert (tmp_path / "proposed_cleaning_report.json").exists()
