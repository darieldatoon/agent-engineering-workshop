#!/usr/bin/env bash
# Refuse to commit anything that looks like a real credential.
#
# Runs over staged files only. Uses grep -E rather than rg so it works on a
# machine with nothing installed but git and bash.
set -uo pipefail

# Real-credential shapes. Deliberately NOT matching bare `sk-...` — lesson 06
# teaches PII blocking with a synthetic `sk-EXAMPLE...` string, and a pattern
# broad enough to catch that would make the hook cry wolf on teaching content.
PATTERN='lsv2_(pt|sk)_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|ghu_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,}|xox[bpsa]-[A-Za-z0-9-]{10,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|tvly-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----'

status=0
for file in "$@"; do
    [ -f "$file" ] || continue
    if matches=$(grep -EnI "$PATTERN" "$file" 2>/dev/null); then
        echo "BLOCKED: possible credential in $file" >&2
        # Print line numbers and the matched shape only — never the full line,
        # which would copy the secret into your terminal scrollback and CI logs.
        echo "$matches" | while IFS= read -r line; do
            echo "  line ${line%%:*}: $(echo "$line" | grep -Eo "$PATTERN" | head -1 | cut -c1-12)..." >&2
        done
        status=1
    fi
done

if [ "$status" -ne 0 ]; then
    echo >&2
    echo "Remove the credential, then rotate it — it was on disk, so treat it as leaked." >&2
    echo "If this is a false positive, commit with --no-verify and fix the pattern in" >&2
    echo "tools/scan_secrets.sh." >&2
fi

exit "$status"
