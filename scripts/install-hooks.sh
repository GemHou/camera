#!/bin/bash
# Install git hooks for the camera project
HOOK_SRC="$(dirname "$0")/pre-commit"
HOOK_DST="$(git rev-parse --show-toplevel)/.git/hooks/pre-commit"
cp "$HOOK_SRC" "$HOOK_DST"
chmod +x "$HOOK_DST"
echo "Installed pre-commit hook: auto-updates version on every commit"
