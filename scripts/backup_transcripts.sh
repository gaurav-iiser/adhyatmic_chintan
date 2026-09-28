#!/usr/bin/env bash
set -euo pipefail

message="${1:-Back up raw transcripts}"

# Stage only durable raw transcript outputs. Do not stage media, .env, or temporary files.
find data -type f \( -name 'raw_transcript.txt' -o -name 'raw_transcript.json' \) -print0 \
  | xargs -0 -r git add --

if git diff --cached --quiet; then
  echo "No new or changed raw transcript files to commit."
  exit 0
fi

git status --short
git commit -m "$message"
git push
