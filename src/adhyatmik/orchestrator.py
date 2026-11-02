from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .cleaner import clean_lecture
from .config import Settings
from .enrichment import enrich_lecture
from .rag_chunker import chunk_lecture
from .reviewer import review_lecture


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _review_is_approved(
    lecture_dir: Path,
) -> bool:
    clean_json = (
        lecture_dir
        / "clean_transcript.json"
    )

    if not clean_json.exists():
        return False

    try:
        payload = _read_json(
            clean_json
        )
    except Exception:
        return False

    return (
        payload.get(
            "human_review_status"
        )
        == "approved"
    )


def lecture_status(
    lecture_id: str,
    settings: Settings,
) -> dict[str, Any]:
    lecture_dir = (
        settings.data_dir
        / lecture_id
    )

    raw_exists = (
        lecture_dir
        / "raw_transcript.json"
    ).exists()

    proposed_exists = (
        lecture_dir
        / "proposed_clean_transcript.json"
    ).exists()

    review_approved = (
        _review_is_approved(
            lecture_dir
        )
    )

    enriched_exists = (
        lecture_dir
        / "enriched_transcript.json"
    ).exists()

    chunks_exist = (
        lecture_dir
        / "rag_chunks.json"
    ).exists()

    return {
        "lecture_id": lecture_id,
        "raw": raw_exists,
        "clean": (
            proposed_exists
            or review_approved
        ),
        "review": review_approved,
        "enrich": enriched_exists,
        "chunk": chunks_exist,
    }


def process_lecture(
    lecture_id: str,
    settings: Settings, *, interactive_review: bool = True
) -> dict[str, Any]:
    """
    Resume one lecture from its first incomplete stage.

    Existing completed stages are never regenerated.
    """

    lecture_dir = (
        settings.data_dir
        / lecture_id
    )

    if not lecture_dir.exists():
        raise FileNotFoundError(
            f"Lecture folder not found: "
            f"{lecture_dir}"
        )

    raw_json = (
        lecture_dir
        / "raw_transcript.json"
    )

    if not raw_json.exists():
        raise FileNotFoundError(
            f"Raw transcript not found: "
            f"{raw_json}"
        )

    # -------------------------------------------------
    # CLEAN
    # -------------------------------------------------

    proposed_json = (
        lecture_dir
        / "proposed_clean_transcript.json"
    )

    final_clean_json = (
        lecture_dir
        / "clean_transcript.json"
    )

    if (
        not proposed_json.exists()
        and not final_clean_json.exists()
    ):
        clean_lecture(
            lecture_id,
            settings,
            force=False,
        )

    # -------------------------------------------------
    # REVIEW
    # -------------------------------------------------

    if not _review_is_approved(
        lecture_dir
    ):
        review_result = review_lecture(
            lecture_id,
            settings,
            interactive=interactive_review,
        )

        # The reviewer allows S/Q, so review may still
        # be incomplete after returning.
        if not _review_is_approved(
            lecture_dir
        ):
            return {
                "lecture_id": lecture_id,
                "status": "needs_review",
                "remaining": review_result.get(
                    "remaining",
                    0,
                ),
                "review_segment_indexes": (
                    review_result.get(
                        "review_segment_indexes",
                        [],
                    )
                ),
            }
            # return {
            #     "lecture_id": lecture_id,
            #     "status": "needs_review",
            #     "review_result": (
            #         review_result
            #     ),
            # }

    # -------------------------------------------------
    # ENRICH
    # -------------------------------------------------

    enriched_json = (
        lecture_dir
        / "enriched_transcript.json"
    )

    if not enriched_json.exists():
        enrich_lecture(
            lecture_id,
            settings,
            force=False,
        )

    # -------------------------------------------------
    # CHUNK
    # -------------------------------------------------

    rag_chunks = (
        lecture_dir
        / "rag_chunks.json"
    )

    if not rag_chunks.exists():
        chunk_lecture(
            lecture_id,
            settings,
            force=False,
        )

    return {
        "lecture_id": lecture_id,
        "status": "complete",
    }


def process_all_lectures(
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Process every lecture folder containing a raw transcript.

    A failure in one lecture does not prevent the remaining
    lectures from being attempted.
    """

    settings.ensure_dirs()

    results: list[dict[str, Any]] = []

    lecture_dirs = [
        path
        for path
        in settings.data_dir.iterdir()
        if (
            path.is_dir()
            and (
                path
                / "raw_transcript.json"
            ).exists()
        )
    ]

    for lecture_dir in sorted(
        lecture_dirs,
        key=lambda path: path.name,
    ):
        lecture_id = (
            lecture_dir.name
        )

        try:
            result = process_lecture(
                lecture_id,
                settings,
                interactive_review=False,
            )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            result = {
                "lecture_id": (
                    lecture_id
                ),
                "status": "error",
                "error": str(exc),
            }

        results.append(
            result
        )

    return results


def all_lecture_statuses(
    settings: Settings,
) -> list[dict[str, Any]]:
    settings.ensure_dirs()

    statuses: list[
        dict[str, Any]
    ] = []

    for lecture_dir in sorted(
        settings.data_dir.iterdir(),
        key=lambda path: path.name,
    ):
        if not lecture_dir.is_dir():
            continue

        if lecture_dir.name.startswith(
            "_"
        ):
            continue

        if not (
            lecture_dir
            / "raw_transcript.json"
        ).exists():
            continue

        statuses.append(
            lecture_status(
                lecture_dir.name,
                settings,
            )
        )

    return statuses