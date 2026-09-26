# Adhyatmik Tools — Lecture Ingestor + Hindi Transcriber

MVP toolchain for turning a lecture video/audio source into a timestamped raw Hindi transcript that preserves Sanskrit/Vedānta terminology.

## What v0.1 does

1. Accepts a local media file or a YouTube URL.
2. Extracts mono 16 kHz audio with FFmpeg.
3. Splits long lectures into fixed time windows (default: 5 minutes).
4. Sends each window to OpenAI `gpt-transcribe` with Hindi + Vedānta context.
5. Writes:
   - `raw_transcript.txt` — human-readable transcript with time ranges.
   - `raw_transcript.json` — machine-readable segments for the later cleaning/RAG pipeline.

This is the **raw transcription** stage. Do not manually replace it with a cleaned transcript; the next pipeline stage will preserve both versions.

## Requirements

- Python 3.11+
- FFmpeg (`ffmpeg` and `ffprobe` on PATH)
- OpenAI API key

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e .
cp .env.example .env
# Put your OPENAI_API_KEY in .env
```

### Transcribe a local file

```bash
adhyatmik file lecture.mp4 --lecture-id bhagavad-gita-ch1-part19
```

### Transcribe a YouTube lecture

```bash
adhyatmik youtube "https://www.youtube.com/watch?v=..." \
  --lecture-id bhagavad-gita-ch1-part19
```

Outputs appear under:

```text
data/<lecture-id>/raw_transcript.txt
data/<lecture-id>/raw_transcript.json
```

## Run the web app

```bash
uvicorn adhyatmik.api:app --reload
```

Open `http://localhost:8000`.

API endpoints:

- `GET /health`
- `POST /api/transcribe/upload`
- `POST /api/transcribe/youtube`

## Docker

```bash
docker build -t adhyatmik-transcriber .
docker run --rm -p 10000:10000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  adhyatmik-transcriber
```

Then open `http://localhost:10000`.

## Deploy on Render

This repository includes `render.yaml` and a Dockerfile. Connect the GitHub repository to Render, create a Blueprint/Web Service, and provide `OPENAI_API_KEY` as a secret environment variable.

Important: the current v0.1 processes a transcription inside the HTTP request. That is fine for initial testing, but long lectures should be moved to a background queue/worker before opening the service to significant public traffic.

## Security / public deployment notes

- Never commit `.env` or API keys. `.env` is gitignored.
- If the site is public and uses **your** OpenAI key, add authentication, quotas, rate limiting, and file-size limits before sharing broadly, or strangers can consume your API budget.
- Uploaded files may contain private material. A production version should use deliberate retention/deletion rules rather than relying on temporary filesystem behavior.
- Only download/transcribe online media that you are permitted to access and process. YouTube URLs can also be less reliable in cloud environments than direct uploads because of platform anti-bot and access controls.

## Project layout

```text
src/adhyatmik/
  api.py          FastAPI web/API layer
  cli.py          CLI entry point
  config.py       environment settings
  media.py        YouTube download + FFmpeg processing
  transcribe.py   OpenAI transcription
  pipeline.py     orchestration + transcript outputs
  models.py       output schemas

glossary/
  vedanta_terms.txt

tests/
  test_pipeline.py
```

## Next planned stage

After validating transcription quality on Lecture 1:

`raw_transcript -> conservative cleaner -> enriched transcript -> RAG chunks`
