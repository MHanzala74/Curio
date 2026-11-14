#!/usr/bin/env bash
# Commit and push Curio's state, digests and profile if anything changed.
# Only paths that exist are added: a single missing pathspec (profile.md before the
# first weekly reflection) would otherwise make `git add` fail and stage nothing.
set -euo pipefail

for path in state digest profile.md; do
  if [ -e "$path" ]; then
    git add -- "$path"
  fi
done

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "chore: daily digest $(date -u +%F)"
  git push
fi
