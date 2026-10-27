#!/usr/bin/env bash
# Refuse to commit any term listed in .denylist (one per line, case-insensitive).
#
# .denylist is gitignored on purpose: the terms it guards must never appear in the repo,
# including in this hook. Without the file the scan is a no-op.
set -uo pipefail

[ -f .denylist ] || exit 0

status=0
while IFS= read -r term; do
    [ -n "$term" ] || continue
    for file in "$@"; do
        [ -f "$file" ] || continue
        if grep -qiIF -- "$term" "$file"; then
            echo "BLOCKED: $file contains a term from .denylist" >&2
            status=1
        fi
    done
done < .denylist

exit "$status"
