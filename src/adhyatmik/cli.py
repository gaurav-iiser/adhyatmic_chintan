from pathlib import Path

import typer
from rich.console import Console

from .cleaner import clean_all_lectures, clean_lecture
from .config import Settings
from .pipeline import process_file, process_youtube
from .reviewer import review_lecture
from .enrichment import enrich_lecture
from .rag_chunker import chunk_lecture
from .indexing import build_index
from .retrieval import retrieve

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


if __name__ == "__main__":
    app()
