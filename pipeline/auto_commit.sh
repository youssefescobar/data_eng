#!/bin/bash

cd "$(dirname "$0")/../" || exit 1

current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
  git branch -m "$current_branch" main
fi

if [ -n "$(git status pipeline/output --porcelain)" ]; then

  git add pipeline/output
  git commit -m "Auto commit after model.py execution"
  git push -u origin main
  echo "Changes in /pipeline/output committed and pushed to remote."
else
  echo "No changes detected in /pipeline/output."
fi
