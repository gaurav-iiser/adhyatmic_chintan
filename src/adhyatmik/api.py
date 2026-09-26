from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl

from .config import Settings
from .pipeline import process_file, process_youtube

app = FastAPI(title="Adhyatmik Lecture Transcriber", version="0.1.0")


class UrlJob(BaseModel):
    url: HttpUrl
    lecture_id: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="en">
    <head><meta charset="utf-8"><title>Adhyatmik Transcriber</title></head>
    <body style="font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px">
      <h1>Adhyatmik Lecture Transcriber</h1>
      <p>Upload a lecture audio/video file. Hindi speech is transcribed with Vedānta/Sanskrit context.</p>
      <form action="/api/transcribe/upload" method="post" enctype="multipart/form-data">
        <label>Lecture ID<br><input name="lecture_id" value="lecture-001" required></label><br><br>
        <label>Media file<br><input type="file" name="file" required></label><br><br>
        <button type="submit">Transcribe</button>
      </form>
      <p><small>For long lectures, the production version will use a background job queue.</small></p>
    </body>
    </html>
    """


@app.post("/api/transcribe/upload")
async def transcribe_upload(
    lecture_id: str = Form(...),
    file: UploadFile = File(...),
):
    suffix = Path(file.filename or "upload.bin").suffix
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            while chunk := await file.read(1024 * 1024):
                tmp.write(chunk)
            path = Path(tmp.name)

        settings = Settings()
        result = await run_in_threadpool(process_file, path, lecture_id, settings)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if 'path' in locals() and path.exists():
                path.unlink()
        except OSError:
            pass


@app.post("/api/transcribe/youtube")
async def transcribe_youtube(job: UrlJob):
    try:
        settings = Settings()
        result = await run_in_threadpool(process_youtube, str(job.url), job.lecture_id, settings)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
