from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .cleaner import clean_all_lectures, clean_lecture
from .config import Settings
from .pipeline import format_timestamp, process_file, process_youtube
from .reviewer import review_lecture
from .enrichment import enrich_lecture
from .rag_chunker import chunk_lecture
from .indexing import build_index
from .retrieval import retrieve
from .answering import answer_question
from .orchestrator import (all_lecture_statuses,
    process_all_lectures,
    process_lecture,
    )

app = typer.Typer(help="Adhyatmik lecture ingestion, transcription, cleaning,"
                        " human review tools, and semantic enrichment tools.")
console = Console()


@app.command("youtube")
def transcribe_youtube(
    url: str = typer.Argument(..., help="YouTube lecture URL"),
    lecture_id: str = typer.Option(..., "--lecture-id", "-i"),
):
    """Download a YouTube lecture, extract audio, and create a raw Hindi transcript."""
    result = process_youtube(url, lecture_id, Settings())
    console.print(f"[green]Transcript:[/green] {result.transcript_path}")
    console.print(f"[green]JSON:[/green] {result.json_path}")


@app.command("file")
def transcribe_file(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    lecture_id: str = typer.Option(..., "--lecture-id", "-i"),
):
    """Transcribe a local audio/video file."""
    result = process_file(path, lecture_id, Settings())
    console.print(f"[green]Transcript:[/green] {result.transcript_path}")
    console.print(f"[green]JSON:[/green] {result.json_path}")


@app.command("clean")
def clean_transcript(
    lecture_id: str = typer.Option(..., "--lecture-id", "-i", help="Existing data/<lecture-id> folder"),
    force: bool = typer.Option(False, "--force", help="Regenerate existing AI-proposed cleaned outputs"),
):
    """Create a conservative AI cleaning proposal while preserving the raw source."""
    result = clean_lecture(lecture_id, Settings(), force=force)
    console.print(f"[green]Proposed clean transcript:[/green] {result.clean_transcript_path}")
    console.print(f"[green]Proposed clean JSON:[/green] {result.clean_json_path}")
    console.print(f"[green]Proposed audit report:[/green] {result.report_path}")
    console.print(f"[cyan]Changes logged:[/cyan] {result.total_changes}")
    if result.review_notes:
        console.print(f"[yellow]Human-review notes:[/yellow] {len(result.review_notes)}")
    console.print(f"[bold]Next:[/bold] adhyatmik review --lecture-id {lecture_id}")


@app.command("review")
def review_transcript(
    lecture_id: str = typer.Option(..., "--lecture-id", "-i", help="Existing data/<lecture-id> folder"),
):
    """Interactively accept, reject, or edit AI cleaner proposals and write the final clean transcript."""
    review_lecture(lecture_id, Settings())

@app.command("enrich")
def enrich_transcript(
    lecture_id: str = typer.Option(
        ...,
        "--lecture-id",
        "-i",
        help="Existing data/<lecture-id> folder",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate an existing enriched transcript",
    ),
):
    """Create semantic metadata from a human-reviewed transcript."""
    result = enrich_lecture(
        lecture_id,
        Settings(),
        force=force,
    )

    console.print(
        f"[green]Enriched JSON:[/green] "
        f"{result.enriched_json_path}"
    )
    console.print(
        f"[cyan]Segments enriched:[/cyan] "
        f"{len(result.segments)}"
    )


@app.command("clean-all")
def clean_all(
    force: bool = typer.Option(False, "--force", help="Regenerate lectures that already have an AI proposal"),
):
    """Create AI cleaning proposals for every lecture under data/."""
    results = clean_all_lectures(Settings(), force=force)
    if not results:
        console.print("[yellow]No lectures needed cleaning.[/yellow]")
        return
    for result in results:
        console.print(
            f"[green]{result.lecture_id}[/green]: {result.total_changes} changes, "
            f"{len(result.review_notes)} review notes"
        )

@app.command("chunk")
def chunk_transcript(
    lecture_id: str = typer.Option(
        ...,
        "--lecture-id",
        "-i",
        help="Existing data/<lecture-id> folder",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate existing RAG chunks",
    ),
):
    """Create semantic retrieval chunks from an enriched transcript."""
    result = chunk_lecture(
        lecture_id,
        Settings(),
        force=force,
    )

    console.print(
        f"[green]RAG chunks:[/green] "
        f"{result.rag_chunks_path}"
    )

    console.print(
        f"[cyan]Chunks created:[/cyan] "
        f"{len(result.chunks)}"
    )

@app.command("process")
def process_pipeline(
    lecture_id: str = typer.Option(
        ...,
        "--lecture-id",
        "-i",
        help="Lecture to process",
    ),
):
    """
    Resume one lecture through clean, review,
    enrichment, and RAG chunking.
    """

    result = process_lecture(
        lecture_id,
        Settings(),
    )

    status = result["status"]

    console.print()

    if status == "complete":
        console.print(
            f"[green]✓ {lecture_id} "
            "is fully processed.[/green]"
        )

    elif status == "needs_review":
        console.print(
            f"[yellow]{lecture_id} still has "
            "segments requiring human review."
            "[/yellow]"
        )

@app.command("index")
def index_rag_chunks(
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Rebuild the global RAG index "
            "and regenerate all embeddings"
            ),
        ),
    ):
    """Embed all RAG chunks and build the global retrieval index."""

    result = build_index(
        Settings(),
        force=force,
    )

    console.print(
        f"[green]RAG index:[/green] "
        f"{result.index_json_path}"
    )

    console.print(
        f"[cyan]Lectures indexed:[/cyan] "
        f"{len(result.source_lecture_ids)}"
    )

    console.print(
        f"[cyan]Chunks indexed:[/cyan] "
        f"{result.source_chunk_count}"
    )

    console.print(
        f"[cyan]Embedding model:[/cyan] "
        f"{result.embedding_model}"
    )

    console.print(
        f"[cyan]Embedding dimensions:[/cyan] "
        f"{result.embedding_dimensions}"
    )

@app.command("process-all")
def process_all_pipeline(
    rebuild_index: bool = typer.Option(
        False,
        "--index",
        help=(
            "Rebuild the global RAG index after "
            "all lectures finish processing"
        ),
    ),
):
    """
    Resume every lecture through the complete
    processing pipeline.
    """

    settings = Settings()

    results = process_all_lectures(
        settings
    )

    review_needed_results = [
        result
        for result in results
        if result["status"] == "needs_review"
    ]

    console.print()

    complete = 0
    needs_review = 0
    errors = 0

    for result in results:
        lecture_id = result[
            "lecture_id"
        ]

        status = result[
            "status"
        ]

        if status == "complete":
            complete += 1

            console.print(
                f"[green]✓[/green] "
                f"{lecture_id}"
            )

        elif status == "needs_review":
            needs_review += 1

            console.print(
                f"[yellow]![/yellow] "
                f"{lecture_id} "
                "needs human review"
            )

        else:
            errors += 1

            console.print(
                f"[red]✗[/red] "
                f"{lecture_id}: "
                f"{result.get('error')}"
            )

    console.print()

    console.print(
        f"[green]Complete:[/green] "
        f"{complete}"
    )

    console.print(
        f"[yellow]Needs review:[/yellow] "
        f"{needs_review}"
    )

    console.print(
        f"[red]Errors:[/red] "
        f"{errors}"
    )

    if review_needed_results:
        console.print()

        console.print(
            "[bold yellow]"
            "Human review required:"
            "[/bold yellow]"
        )

        for result in review_needed_results:
            indexes = result.get(
                "review_segment_indexes",
                [],
            )

            indexes_text = ", ".join(
                str(index)
                for index in indexes
            )

            console.print(
                f"  • {result['lecture_id']}: "
                f"{result.get('remaining', len(indexes))} "
                f"segment(s)"
                + (
                    f" [{indexes_text}]"
                    if indexes_text
                    else ""
                )
            )

    # -----------------------------------------
    # Optional global index rebuild
    # -----------------------------------------

    if rebuild_index:
        if needs_review or errors:
            console.print()
            console.print(
                "[yellow]"
                "Index was NOT rebuilt because "
                "some lectures are incomplete."
                "[/yellow]"
            )

        else:
            console.print()
            console.print(
                "[cyan]"
                "Rebuilding global RAG index..."
                "[/cyan]"
            )

            index = build_index(
                settings,
                force=True,
            )

            console.print(
                f"[green]✓ Index rebuilt:[/green] "
                f"{index.source_chunk_count} chunks "
                f"from "
                f"{len(index.source_lecture_ids)} "
                "lectures."
            )

@app.command("status")
def pipeline_status():
    """
    Show pipeline progress for every lecture.
    """

    settings = Settings()

    statuses = all_lecture_statuses(
        settings
    )

    table = Table(
        title="Adhyatmik Pipeline Status"
    )

    table.add_column(
        "Lecture",
        style="bold",
    )

    table.add_column(
        "Raw",
        justify="center",
    )

    table.add_column(
        "Clean",
        justify="center",
    )

    table.add_column(
        "Review",
        justify="center",
    )

    table.add_column(
        "Enrich",
        justify="center",
    )

    table.add_column(
        "Chunk",
        justify="center",
    )

    def mark(
        value: bool,
    ) -> str:
        return (
            "[green]✓[/green]"
            if value
            else "[dim]-[/dim]"
        )

    for status in statuses:
        if (
            status["clean"]
            and not status["review"]
        ):
            review_mark = (
                "[yellow]![/yellow]"
            )
        else:
            review_mark = mark(
                status["review"]
            )

        table.add_row(
            status["lecture_id"],
            mark(status["raw"]),
            mark(status["clean"]),
            review_mark,
            mark(status["enrich"]),
            mark(status["chunk"]),
        )

    console.print(table)

@app.command("retrieve")
def retrieve_chunks(
    query: str = typer.Argument(
        ...,
        help="Question or search query",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of chunks to return",
    ),
):
    """Retrieve the most relevant RAG chunks."""

    result = retrieve(
        query,
        Settings(),
        top_k=top_k,
    )

    console.print(
        f"\n[bold]Query:[/bold] "
        f"{result.query}"
    )

    console.print(
        f"[dim]Embedding model: "
        f"{result.embedding_model}[/dim]\n"
    )

    for hit in result.hits:
        chunk = hit.chunk

        console.rule(
            f"#{hit.rank} "
            f"{chunk.chunk_id}"
        )

        console.print(
            f"[cyan]Score:[/cyan] "
            f"{hit.score:.4f}"
        )

        console.print(
            f"[cyan]Lecture:[/cyan] "
            f"{chunk.lecture_id}"
        )

        console.print(
            f"[cyan]Time:[/cyan] "
            f"{chunk.start_seconds:.1f}"
            f"–"
            f"{chunk.end_seconds:.1f} sec"
        )

        source_segments = [
            str(
                fragment.source_segment_index
            )
            for fragment
            in chunk.source_fragments
        ]

        console.print(
            f"[cyan]Source segments:[/cyan] "
            f"{', '.join(source_segments)}"
        )

        console.print()

        console.print(
            chunk.text
        )

        console.print()

@app.command("ask")
def ask_question(
    query: str = typer.Argument(
        ...,
        help="Question to answer from the indexed lectures",
    ),
    context: str | None = typer.Option(
        None,
        "--context",
        "-c",
        help=(
            "Optional background/context "
            "for the question"
        ),
    ),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        "-k",
        help=(
            "Optional override for automatic "
            "retrieval size"
        ),
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help=(
            "Answer mode: auto, focused, "
            "broad, or chapter-summary"
        ),
    ),
    chapter: int | None = typer.Option(
        None,
        "--chapter",
        help=(
            "Optional explicit Bhagavad Gita "
            "chapter number"
        ),
    ),
):
    """Answer a question using only retrieved lecture material."""

    result = answer_question(
        query,
        Settings(),
        context=context,
        top_k=top_k,
        mode=mode,
        chapter=chapter,
    )

    console.print()

    console.print(
        f"[bold]Question:[/bold] "
        f"{result.query}"
    )

    console.print()

    if result.sufficient_context:
        console.print(
            "[green]Answer:[/green]"
        )
    else:
        console.print(
            "[yellow]Insufficient context:[/yellow]"
        )

    console.print(
        result.rendered_answer
    )

    console.print()

    if result.references:
        console.print()

        console.print(
            "[bold]References[/bold]"
        )

        for reference in result.references:
            console.print(
                "• "
                f"{reference.lecture_id} — "
                f"{format_timestamp(reference.start_seconds)}"
                "–"
                f"{format_timestamp(reference.end_seconds)}"
            )

    # console.print(
    #     f"[dim]Retrieved chunks: "
    #     f"{', '.join(result.retrieved_chunk_ids)}"
    #     f"[/dim]"
    # )

    # if result.cited_chunk_ids:
    #     console.print(
    #         f"[dim]Used in answer: "
    #         f"{', '.join(result.cited_chunk_ids)}"
    #         f"[/dim]"
    #     )


if __name__ == "__main__":
    app()
