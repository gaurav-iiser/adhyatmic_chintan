# Adhyatmik Tools — Clone, Restore, and Run on Windows (Git Bash)

This guide is the recovery procedure for the Adhyatmik lecture transcription project.

Use it when setting up a new computer, after cloning the GitHub repository, or if your local project folder is accidentally deleted.

## What is stored in GitHub

The repository should contain:

- all Python source code under `src/`
- tests
- the Vedānta glossary
- Docker/Render configuration
- this setup guide
- each lecture's `raw_transcript.txt`
- each lecture's `raw_transcript.json`

The repository should **not** contain:

- `.env`
- your OpenAI API key
- `.venv/`
- downloaded videos
- extracted audio files
- temporary transcription chunks

Those items are either secrets, machine-specific, or can be regenerated.

---

## 1. Install prerequisites

You need:

1. Git
2. Python 3.11 or newer
3. FFmpeg
4. An OpenAI API key with API credits

### Check Git

In Git Bash:

```bash
git --version
```

### Check Python

```bash
py --version
```

Python 3.11 or newer is recommended.

### Install/check FFmpeg

From Windows Terminal, PowerShell, or a shell where `winget` is available:

```bash
winget install Gyan.FFmpeg
```

Close and reopen Git Bash after installation. Then verify:

```bash
ffmpeg -version
ffprobe -version
```

Both commands must work before lecture transcription can run.

---

## 2. Clone the repository

In Git Bash, move to the folder where you keep projects. Example:

```bash
cd /c/Users/YOUR_WINDOWS_USERNAME/Documents
```

Clone the repository:

```bash
git clone https://github.com/gaurav-iiser/adhyatmic_chintan.git
```

Enter it:

```bash
cd adhyatmic_chintan
```

Check that you are in the correct folder:

```bash
ls
```

You should see files such as `pyproject.toml`, `README.md`, `src`, `glossary`, and `data`.

---

## 3. Create the Python virtual environment

From the repository root:

```bash
py -3.11 -m venv .venv
```

Activate it in Git Bash:

```bash
source .venv/Scripts/activate
```

Your terminal prompt should now begin with something similar to:

```text
(.venv)
```

If Python 3.11 is not installed but a newer compatible Python is installed, you can use:

```bash
py -m venv .venv
source .venv/Scripts/activate
```

---

## 4. Install the Adhyatmik tools

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project and development/test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Verify that the command was installed:

```bash
adhyatmik --help
```

If Git Bash says `adhyatmik: command not found`, first make sure the virtual environment is active:

```bash
source .venv/Scripts/activate
```

Then reinstall:

```bash
python -m pip install -e ".[dev]"
```

---

## 5. Create your local `.env` file

The `.env` file is deliberately **not stored in GitHub** because it contains your API key.

Create it from the template:

```bash
cp .env.example .env
```

Open it in Notepad:

```bash
notepad .env
```

Set the values:

```text
OPENAI_API_KEY=YOUR_REAL_OPENAI_API_KEY
TRANSCRIPTION_MODEL=gpt-transcribe
CHUNK_SECONDS=300
DATA_DIR=./data
```

Save and close Notepad.

Never commit `.env` or paste your API key into GitHub.

---

## 6. Verify the installation

Run the automated tests:

```bash
pytest
```

Then verify the required programs:

```bash
python --version
ffmpeg -version
ffprobe -version
adhyatmik --help
```

If all four commands work, the installation is ready.

---

## 7. Restore existing transcripts

Existing raw transcripts are stored in Git under:

```text
data/<lecture-id>/raw_transcript.txt
data/<lecture-id>/raw_transcript.json
```

Because they are versioned in GitHub, cloning the repository automatically restores them. You do not need to retranscribe those lectures after cloning.

Example:

```text
data/
  bhagavad-gita-ch1-part19/
    raw_transcript.txt
    raw_transcript.json
  bhagavad-gita-ch1-part20/
    raw_transcript.txt
    raw_transcript.json
```

---

## 8. Transcribe a new local lecture

Suppose your lecture file is:

```text
C:\Users\YOUR_WINDOWS_USERNAME\Videos\lecture5.mp4
```

Git Bash represents that path as:

```text
/c/Users/YOUR_WINDOWS_USERNAME/Videos/lecture5.mp4
```

Run:

```bash
adhyatmik file /c/Users/YOUR_WINDOWS_USERNAME/Videos/lecture5.mp4 --lecture-id lecture5
```

When transcription finishes, the output is stored in:

```text
data/lecture5/raw_transcript.txt
data/lecture5/raw_transcript.json
```

Use a stable, meaningful lecture ID because it becomes the permanent data folder name.

Example:

```bash
adhyatmik file /c/Users/YOUR_WINDOWS_USERNAME/Videos/lecture5.mp4 \
  --lecture-id bhagavad-gita-ch1-part23
```

---

## 9. Transcribe from YouTube

If the lecture is available on YouTube:

```bash
adhyatmik youtube "https://www.youtube.com/watch?v=VIDEO_ID" \
  --lecture-id bhagavad-gita-ch1-part23
```

Direct local-file transcription is the more reliable path if YouTube blocks automated downloading.

---

## 10. Back up every new raw transcript to GitHub

After creating a transcript, inspect Git's changes:

```bash
git status
```

Only the transcript outputs you want to preserve should appear under `data/`.

Stage the new transcript files:

```bash
git add data/
```

Commit them:

```bash
git commit -m "Add raw transcript for Bhagavad Gita Ch1 Part 23"
```

Push them to GitHub:

```bash
git push origin main
```

After the push succeeds, the raw transcript exists both locally and in GitHub.

You can also use the included helper script:

```bash
bash scripts/backup_transcripts.sh "Add latest raw transcripts"
```

The script stages only `raw_transcript.txt` and `raw_transcript.json`, creates a commit, and pushes the current branch.

---

## 11. Recommended workflow for every new lecture

Use this sequence each time:

```text
1. git pull
2. activate .venv
3. run transcription
4. inspect raw_transcript.txt
5. git status
6. commit raw transcript TXT + JSON
7. git push
```

Commands:

```bash
git pull
source .venv/Scripts/activate

adhyatmik file /c/path/to/lecture.mp4 --lecture-id YOUR_LECTURE_ID

git status
git add data/
git commit -m "Add raw transcript for YOUR_LECTURE_ID"
git push origin main
```

---

## 12. Start the local web interface (optional)

Activate the virtual environment and run:

```bash
source .venv/Scripts/activate
uvicorn adhyatmik.api:app --reload
```

Open in your browser:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

Stop the server with `Ctrl+C`.

---

## 13. Updating the code later

Before starting work on an existing clone:

```bash
git pull
```

If `pyproject.toml` changed, reinstall dependencies:

```bash
source .venv/Scripts/activate
python -m pip install -e ".[dev]"
```

Then run:

```bash
pytest
```

---

## 14. Full disaster-recovery checklist

If the entire local project is lost, the recovery procedure is:

```bash
cd /c/Users/YOUR_WINDOWS_USERNAME/Documents

git clone https://github.com/gaurav-iiser/adhyatmic_chintan.git
cd adhyatmic_chintan

py -3.11 -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cp .env.example .env
notepad .env
```

Put your OpenAI API key into `.env`, then verify:

```bash
pytest
ffmpeg -version
ffprobe -version
adhyatmik --help
```

At that point:

- the source code has been restored from GitHub;
- all committed raw transcripts have been restored from GitHub;
- the Python environment has been recreated;
- only the API secret had to be supplied again.

You can immediately continue transcribing new lectures.

---

## Important data rule

Treat `raw_transcript.txt` and `raw_transcript.json` as **immutable source data** once committed.

When the cleaning stage is added, do not overwrite raw transcripts. The pipeline should instead create separate files such as:

```text
raw_transcript.txt
raw_transcript.json
clean_transcript.txt
clean_transcript.json
```

This preserves an audit trail from the original ASR result to every later processing stage.
