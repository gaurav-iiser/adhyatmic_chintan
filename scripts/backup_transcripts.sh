#!/usr/bin/env bash
set -euo pipefail

message="${1:-Back up transcript artifacts}"

# Stage only durable transcript/review outputs. Do not stage media, .env, or temporary files.
find data -type f \( \
  -name 'raw_transcript.txt' -o \
  -name 'raw_transcript.json' -o \
  -name 'proposed_clean_transcript.txt' -o \
  -name 'proposed_clean_transcript.json' -o \
  -name 'proposed_cleaning_report.json' -o \
  -name 'human_review_state.json' -o \
  -name 'clean_transcript.txt' -o \
  -name 'clean_transcript.json' -o \
  -name 'human_review_report.json' \
\) -print0 | xargs -0 -r git add --

if git diff --cached --quiet; then
  echo "No new or changed transcript artifacts to commit."
  exit 0
fi

git status --short
git commit -m "$message"
git push
