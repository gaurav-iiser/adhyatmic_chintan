# Human Review / Approve / Edit Workflow

The human-review stage turns an AI cleaning proposal into the canonical clean transcript.

## File roles

Each lecture folder uses this sequence:

```text
raw_transcript.txt/json                 immutable ASR output
        ↓
proposed_clean_transcript.txt/json      AI cleaner proposal
proposed_cleaning_report.json           AI change audit
        ↓
adhyatmik review
        ↓
clean_transcript.txt/json               human-reviewed canonical transcript
human_review_report.json                decision audit
human_review_state.json                 resumable review state
```

Never edit `raw_transcript.*`.

## Start review

From the repository root in Git Bash:

```bash
source .venv/Scripts/activate
adhyatmik review --lecture-id YOUR_LECTURE_ID
```

For example:

```bash
adhyatmik review --lecture-id lecture1.1
```

## Decisions

For every segment where the cleaner changed text or left a review note, the terminal displays:

- raw text
- AI-proposed cleaned text
- logged AI changes and confidence
- human-review notes

Then choose:

- `A` — accept the AI proposal for that segment
- `R` — reject the proposal and restore the raw segment
- `E` — edit the proposed segment manually
- `S` — skip this segment for now
- `Q` — save progress and quit

Segments with no AI changes and no review notes are accepted automatically.

### Edit on Windows

Choosing `E` opens the proposed segment in Notepad. Make your corrections, save the file, and close Notepad. The review tool then uses that saved text as the final segment.

## Resume later

The tool saves `human_review_state.json` after each decision. If you quit or skip items, run the same command again:

```bash
adhyatmik review --lecture-id YOUR_LECTURE_ID
```

It resumes only the segments that still need a decision.

## Completion

When every changed/flagged segment has a decision, the tool writes:

- `clean_transcript.txt`
- `clean_transcript.json`
- `human_review_report.json`

At that point `clean_transcript.*` is the canonical transcript for later enrichment/RAG work.

## Compatibility with cleaner v0.3

If a lecture was already cleaned with v0.3, the first `adhyatmik review` command automatically renames the v0.3 AI proposal files:

```text
clean_transcript.txt  → proposed_clean_transcript.txt
clean_transcript.json → proposed_clean_transcript.json
cleaning_report.json  → proposed_cleaning_report.json
```

Raw transcript files are never changed. After human review, new final `clean_transcript.*` files are created.

## Git backup after review

After checking the final transcript:

```bash
git status
git add data/ src/ docs/ .gitignore pyproject.toml
git commit -m "Add human-reviewed transcript for YOUR_LECTURE_ID"
git push origin main
```

Never commit `.env` or your OpenAI API key.
