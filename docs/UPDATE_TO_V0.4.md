# Update existing Adhyatmik repo to v0.4 Human Review

These instructions assume you already have the v0.3 cleaner working and your lecture data is under `data/`.

## 1. Back up current work first

From the repository root:

```bash
git status
git add data/
git commit -m "Back up transcripts before human review v0.4"
git push origin main
```

If Git reports nothing to commit, continue.

## 2. Copy the v0.4 update into the repository root

Extract the update ZIP and copy its contents into your existing `adhyatmic_chintan` folder. Allow it to replace the matching code/docs files.

Do not delete or replace your existing `data/` folder or `.env`.

## 3. Activate the repo virtual environment

Windows + Git Bash:

```bash
source .venv/Scripts/activate
```

## 4. Reinstall the editable package

```bash
pip install -e ".[dev]"
```

## 5. Verify the new command

```bash
adhyatmik --help
```

You should see a `review` command.

## 6. Review the lecture already cleaned with v0.3

```bash
adhyatmik review --lecture-id lecture1.1
```

Use your real lecture folder name if different.

The first run automatically preserves the v0.3 cleaner output as `proposed_clean_transcript.*`, then starts interactive review.

Commands during review:

- `a` accept AI proposal
- `r` reject; use raw text
- `e` edit in Notepad
- `s` skip for now
- `q` save and quit

Run the same command again to resume.

## 7. After review completes

Confirm these exist:

```text
proposed_clean_transcript.txt
proposed_clean_transcript.json
proposed_cleaning_report.json
clean_transcript.txt
clean_transcript.json
human_review_state.json
human_review_report.json
```

The final `clean_transcript.*` files are the human-reviewed canonical transcript.

## 8. Commit the reviewed lecture

```bash
git status
git add data/ src/ docs/ scripts/ .gitignore pyproject.toml README.md
git commit -m "Add human review workflow and reviewed lecture1.1"
git push origin main
```
