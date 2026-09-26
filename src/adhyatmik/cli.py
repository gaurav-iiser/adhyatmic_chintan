from pathlib import Path

import typer
from rich.console import Console

from .config import Settings
from .pipeline import process_file, process_youtube

app = typer.Typer(help="Adhyatmik lecture ingestion and transcription tools")
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


if __name__ == "__main__":
    app()
