from pathlib import Path

import typer
from rich.console import Console

from .cleaner import clean_all_lectures, clean_lecture
from .config import Settings
from .pipeline import process_file, process_youtube
from .reviewer import review_lecture

app = typer.Typer(help="Adhyatmik lecture ingestion, transcription, cleaning, and human review tools")
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


if __name__ == "__main__":
    app()
