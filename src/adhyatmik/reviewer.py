from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .config import Settings
from .pipeline import format_timestamp

console = Console()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _proposal_paths(lecture_dir: Path) -> tuple[Path, Path, Path]:
    return (
        lecture_dir / "proposed_clean_transcript.txt",
        lecture_dir / "proposed_clean_transcript.json",
        lecture_dir / "proposed_cleaning_report.json",
    )


def migrate_v03_cleaner_outputs(lecture_dir: Path) -> bool:
    """Move v0.3 cleaner outputs into the v0.4 proposed-output names.

    Returns True if any migration occurred. Raw transcript files are never touched.
    """
    proposed_txt, proposed_json, proposed_report = _proposal_paths(lecture_dir)
    legacy_txt = lecture_dir / "clean_transcript.txt"
    legacy_json = lecture_dir / "clean_transcript.json"
    legacy_report = lecture_dir / "cleaning_report.json"

    if proposed_json.exists():
        return False
    if not legacy_json.exists():
        return False

    if proposed_txt.exists() or proposed_report.exists():
        raise FileExistsError(
            "Found a partial mix of v0.3 and v0.4 cleaner files. Back up the lecture folder, "
            "then resolve the duplicate proposed/clean files before reviewing."
        )

    if legacy_txt.exists():
        legacy_txt.replace(proposed_txt)
    legacy_json.replace(proposed_json)
    if legacy_report.exists():
        legacy_report.replace(proposed_report)
    return True

def _segment_needs_human_review(
    segment: dict[str, Any],
) -> bool:
    """
    Human Review v2:

    Prefer the cleaner's explicit needs_human_review flag.

    For older cleaner outputs that do not contain that field,
    fall back to the old behavior.
    """
    explicit_flag = segment.get(
        "needs_human_review"
    )

    if explicit_flag is not None:
        return bool(explicit_flag)

    # Backward compatibility with old cleaner outputs.
    return bool(
        segment.get("changes")
        or segment.get("review_notes")
    )

def _new_state(
    lecture_id: str,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}

    for segment in proposal.get("segments", []):
        idx = str(segment["index"])

        # New Human Review v2 behavior:
        # trust the cleaner's explicit review gate.
        #
        # Backward compatibility:
        # older proposed transcripts may not contain
        # needs_human_review, so fall back to the old rule.
        needs_human_review = segment.get(
            "needs_human_review"
        )

        if needs_human_review is None:
            has_review_work = bool(
                segment.get("changes")
                or segment.get("review_notes")
            )
            should_auto_accept = not has_review_work
        else:
            should_auto_accept = not bool(
                needs_human_review
            )

        if should_auto_accept:
            decisions[idx] = {
                "decision": "auto_accept_unchanged",
                "final_text": segment.get(
                    "cleaned_text",
                    "",
                ),
                "reviewer_note": "",
                "reviewed_at": _utc_now(),
            }

    return {
        "lecture_id": lecture_id,
        "status": "in_progress",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "decisions": decisions,
    }

def _sync_auto_accepted_segments(
    proposal: dict[str, Any],
    state: dict[str, Any],
) -> bool:
    """
    Add automatic decisions for safe segments that
    do not already have a decision.

    This is especially important for review states
    created before Human Review v2.
    """
    decisions = state.setdefault(
        "decisions",
        {},
    )

    changed = False

    for segment in proposal.get(
        "segments",
        [],
    ):
        idx = str(segment["index"])

        # Never overwrite an existing human decision.
        if idx in decisions:
            continue

        if _segment_needs_human_review(
            segment
        ):
            continue

        decisions[idx] = {
            "decision": "auto_accept_unchanged",
            "final_text": segment.get(
                "cleaned_text",
                "",
            ),
            "reviewer_note": "",
            "reviewed_at": _utc_now(),
        }

        changed = True

    return changed


def load_or_create_review_state(
    lecture_dir: Path,
    lecture_id: str,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    state_path = (
        lecture_dir
        / "human_review_state.json"
    )

    if state_path.exists():
        state = _read_json(
            state_path
        )

        if (
            state.get("lecture_id")
            != lecture_id
        ):
            raise ValueError(
                "Existing review state belongs "
                "to a different lecture ID."
            )

        changed = (
            _sync_auto_accepted_segments(
                proposal,
                state,
            )
        )

        if changed:
            _save_state(
                lecture_dir,
                state,
            )

        return state

    state = _new_state(
        lecture_id,
        proposal,
    )

    _write_json(
        state_path,
        state,
    )

    return state


def _save_state(lecture_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _write_json(lecture_dir / "human_review_state.json", state)


def _launch_editor(initial_text: str, lecture_dir: Path) -> str:
    temp_path = lecture_dir / ".human_review_edit.txt"
    temp_path.write_text(initial_text.rstrip() + "\n", encoding="utf-8")

    try:
        if os.name == "nt":
            subprocess.run(["notepad.exe", str(temp_path)], check=False)
        else:
            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
            if editor:
                subprocess.run([editor, str(temp_path)], check=False)
            else:
                raise RuntimeError(
                    "No text editor configured. Set the EDITOR environment variable, for example EDITOR=nano."
                )
        edited = temp_path.read_text(encoding="utf-8").strip()
        if not edited:
            raise ValueError("Edited text is empty; no decision was saved.")
        return edited
    finally:
        temp_path.unlink(missing_ok=True)


def _display_segment(segment: dict[str, Any], position: int, total: int) -> None:
    start = format_timestamp(float(segment.get("start_seconds", 0)))
    end = format_timestamp(float(segment.get("end_seconds", 0)))
    console.rule(f"Segment {segment['index']}  [{start} - {end}]  ({position}/{total})")
    console.print(Panel(segment.get("raw_text", ""), title="RAW", border_style="yellow"))
    console.print(Panel(segment.get("cleaned_text", ""), title="AI PROPOSAL", border_style="cyan"))

    changes = segment.get("changes") or []
    if changes:
        table = Table(title="Logged AI changes", show_lines=True)
        table.add_column("Category")
        table.add_column("Confidence")
        table.add_column("Original")
        table.add_column("Proposed")
        table.add_column("Reason")
        for change in changes:
            table.add_row(
                str(change.get("category", "")),
                str(change.get("confidence", "")),
                str(change.get("original", "")),
                str(change.get("corrected", "")),
                str(change.get("reason", "")),
            )
        console.print(table)

    notes = [str(n) for n in (segment.get("review_notes") or []) if str(n).strip()]
    if notes:
        console.print("[bold yellow]Human-review notes:[/bold yellow]")
        for note in notes:
            console.print(f"  • {note}")


def pending_segment_indexes( proposal: dict[str, Any],
    state: dict[str, Any],
    ) -> list[int]:
    decisions = state.get(
        "decisions",
        {},
    )

    pending: list[int] = []

    for segment in proposal.get("segments", []):
        idx = int(segment["index"])
        if (
            _segment_needs_human_review(
                segment
            )
            and str(idx) not in decisions
            ):
            pending.append(
                idx
            )

    return pending


def finalize_review(lecture_dir: Path, proposal: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    _sync_auto_accepted_segments( proposal, state,)
    pending = pending_segment_indexes(proposal, state)
    if pending:
        raise RuntimeError(f"Cannot finalize: {len(pending)} segment(s) still need review.")

    final_segments: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    counts = {"accepted": 0, "rejected": 0, "edited": 0, "auto_accept_unchanged": 0}

    for segment in proposal.get("segments", []):
        idx = str(segment["index"])
        decision = state["decisions"][idx]
        decision_name = decision["decision"]
        counts[decision_name] = counts.get(decision_name, 0) + 1
        final_text = decision["final_text"].strip()

        final_segment = {
            "index": segment["index"],
            "start_seconds": segment["start_seconds"],
            "end_seconds": segment["end_seconds"],
            "raw_text": segment.get("raw_text", ""),
            "proposed_text": segment.get("cleaned_text", ""),
            "cleaned_text": final_text,
            "model_changes": segment.get("changes", []),
            "model_review_notes": segment.get("review_notes", []),
            "review_decision": decision_name,
            "reviewer_note": decision.get("reviewer_note", ""),
            "reviewed_at": decision.get("reviewed_at"),
        }
        final_segments.append(final_segment)
        text_blocks.append(
            f"[{format_timestamp(float(segment['start_seconds']))} - "
            f"{format_timestamp(float(segment['end_seconds']))}]\n{final_text}"
        )

    clean_txt_path = lecture_dir / "clean_transcript.txt"
    clean_json_path = lecture_dir / "clean_transcript.json"
    report_path = lecture_dir / "human_review_report.json"

    clean_txt_path.write_text("\n\n".join(text_blocks) + "\n", encoding="utf-8")

    final_payload = {
        "lecture_id": proposal.get("lecture_id"),
        "source_raw_json": proposal.get("source_raw_json"),
        "source_proposed_json": str(lecture_dir / "proposed_clean_transcript.json"),
        "cleaning_model": proposal.get("cleaning_model"),
        "human_review_status": "approved",
        "human_review_completed_at": _utc_now(),
        "segments": final_segments,
    }
    _write_json(clean_json_path, final_payload)

    report = {
        "lecture_id": proposal.get("lecture_id"),
        "status": "approved",
        "completed_at": _utc_now(),
        "counts": counts,
        "decisions": state.get("decisions", {}),
        "final_clean_transcript": str(clean_txt_path),
        "final_clean_json": str(clean_json_path),
    }
    _write_json(report_path, report)

    state["status"] = "completed"
    state["completed_at"] = _utc_now()
    _save_state(lecture_dir, state)
    return report


def review_lecture(lecture_id: str, settings: Settings) -> dict[str, Any]:
    lecture_dir = settings.data_dir / lecture_id
    if not lecture_dir.exists():
        raise FileNotFoundError(f"Lecture folder not found: {lecture_dir}")

    migrated = migrate_v03_cleaner_outputs(lecture_dir)
    if migrated:
        console.print(
            "[cyan]Migrated v0.3 cleaner outputs to proposed_clean_transcript.* so the AI proposal is preserved.[/cyan]"
        )

    proposed_txt, proposed_json, _ = _proposal_paths(lecture_dir)
    if not proposed_json.exists():
        raise FileNotFoundError(
            f"No proposed cleaned transcript found: {proposed_json}. Run 'adhyatmik clean --lecture-id {lecture_id}' first."
        )
    if not proposed_txt.exists():
        console.print("[yellow]Warning: proposed_clean_transcript.txt is missing; review will use the JSON proposal.[/yellow]")

    proposal = _read_json(proposed_json)
    state = load_or_create_review_state(lecture_dir, lecture_id, proposal)

    if state.get("status") == "completed":
        console.print("[green]This lecture has already completed human review.[/green]")
        return _read_json(lecture_dir / "human_review_report.json")

    pending = pending_segment_indexes(proposal, state)
    review_segments = [
        segment for segment in proposal.get("segments", []) if int(segment["index"]) in set(pending)
    ]

    if not review_segments:
        report = finalize_review(lecture_dir, proposal, state,)

        console.print(
            "[green]"
            "No segments require human review. "
            "All segments were automatically accepted "
            "and the final clean transcript was written."
            "[/green]"
        )

        return report

    console.print(
        f"[bold]Human review: {lecture_id}[/bold]\n"
        f"{len(review_segments)} segment(s) need human review. All other segments are accepted automatically.\n"
        "[dim]A=accept AI proposal, R=reject and use raw, E=edit in Notepad, S=skip for now, Q=save and quit[/dim]"
    )

    total = len(review_segments)
    for position, segment in enumerate(review_segments, start=1):
        _display_segment(segment, position, total)
        choice = Prompt.ask(
            "Decision",
            choices=["a", "r", "e", "s", "q"],
            default="a",
            show_choices=True,
        ).lower()

        if choice == "q":
            _save_state(lecture_dir, state)
            remaining = len(pending_segment_indexes(proposal, state))
            console.print(f"[yellow]Review saved. {remaining} segment(s) remain. Run the same review command to resume.[/yellow]")
            return {"status": "in_progress", "remaining": remaining}

        if choice == "s":
            continue

        if choice == "a":
            final_text = segment.get("cleaned_text", "")
            decision_name = "accepted"
        elif choice == "r":
            final_text = segment.get("raw_text", "")
            decision_name = "rejected"
        else:
            final_text = _launch_editor(segment.get("cleaned_text", ""), lecture_dir)
            decision_name = "edited"

        reviewer_note = ""
        if choice in {"r", "e"}:
            reviewer_note = Prompt.ask("Optional reviewer note (press Enter to leave blank)", default="")

        state.setdefault("decisions", {})[str(segment["index"])] = {
            "decision": decision_name,
            "final_text": final_text.strip(),
            "reviewer_note": reviewer_note.strip(),
            "reviewed_at": _utc_now(),
        }
        _save_state(lecture_dir, state)
        console.print(f"[green]Saved: {decision_name}[/green]\n")

    remaining = pending_segment_indexes(proposal, state)
    if remaining:
        console.print(
            f"[yellow]{len(remaining)} segment(s) were skipped. Review state is saved; rerun the command to finish.[/yellow]"
        )
        return {"status": "in_progress", "remaining": len(remaining)}

    report = finalize_review(lecture_dir, proposal, state)
    console.print("[bold green]Human review complete.[/bold green]")
    console.print(f"[green]Final transcript:[/green] {lecture_dir / 'clean_transcript.txt'}")
    console.print(f"[green]Final JSON:[/green] {lecture_dir / 'clean_transcript.json'}")
    console.print(f"[green]Review audit:[/green] {lecture_dir / 'human_review_report.json'}")
    return report
