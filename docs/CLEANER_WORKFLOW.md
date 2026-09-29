# Transcript Cleaner Workflow

The cleaner is deliberately conservative. It corrects transcription errors; it does not rewrite or summarize the lecture.

## Pipeline

```text
raw_transcript.*
      ↓
AI conservative cleaner
      ↓
proposed_clean_transcript.*
proposed_cleaning_report.json
      ↓
human review
      ↓
clean_transcript.*
human_review_report.json
```

The raw transcript files are never overwritten.

## Windows + Git Bash

Activate the project environment from the repository root:

```bash
source .venv/Scripts/activate
```

Create a cleaning proposal for one lecture:

```bash
adhyatmik clean --lecture-id YOUR_LECTURE_ID
```

Example:

```bash
adhyatmik clean --lecture-id lecture1.1
```

The AI cleaner creates:

- `proposed_clean_transcript.txt`
- `proposed_clean_transcript.json`
- `proposed_cleaning_report.json`

Then start the required human-review stage:

```bash
adhyatmik review --lecture-id YOUR_LECTURE_ID
```

See [`HUMAN_REVIEW_WORKFLOW.md`](HUMAN_REVIEW_WORKFLOW.md) for Accept / Reject / Edit / Resume instructions.

## Bulk proposal generation

After validating the cleaner and reviewer on a representative lecture, AI proposals can be generated for all remaining raw lectures:

```bash
adhyatmik clean-all
```

Human review is still done lecture-by-lecture.

## Regeneration

The cleaner refuses to overwrite an existing AI proposal by default. To intentionally regenerate the proposal:

```bash
adhyatmik clean --lecture-id YOUR_LECTURE_ID --force
```

Do not use `--force` casually. Back up/commit the current proposal and review artifacts if you need them for comparison.

## Validation priorities

Pay particular attention to Sanskrit verses and meaning-bearing Vedānta words such as आत्मा, अनात्मा, ब्रह्म, अविद्या, अध्यास, कर्म, धर्म, वैराग्य, विवेक, and मोक्ष.

A correction that changes philosophical meaning is more serious than ordinary punctuation or spelling.
