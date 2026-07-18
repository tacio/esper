#!/usr/bin/env bash
# Esper PreToolUse guardrail for git — enforces the local-first rule
# (CLAUDE.md §Version control): commit DIRECTLY to `master`, no feature branch,
# no PR, and NEVER push to the remote.
#
# Reads a Claude Code PreToolUse Bash payload on stdin and, when the command
# would push or would commit/branch off master, emits permissionDecision:"deny".
# Anything else passes through untouched (exit 0, no output).
set -euo pipefail

payload="$(cat)"

# Pull the command string out of the JSON payload (python3 is already required
# by the sibling mojo_lint hook, so this dependency is free).
cmd="$(printf '%s' "$payload" | python3 -c \
  'import json,sys;print((json.load(sys.stdin).get("tool_input",{}) or {}).get("command","") or "")' \
  2>/dev/null || true)"

[[ -z "$cmd" ]] && exit 0

deny() {
  # $1 = reason
  python3 - "$1" <<'PY'
import json, sys
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": sys.argv[1],
    }
}))
PY
  exit 0
}

# 1) Never push to the remote.
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])git[[:space:]]+push\b'; then
  deny "Esper is local-first (CLAUDE.md §Version control): NEVER push to the remote. Commit to master and stop."
fi

# 2) No feature branches / PRs — work happens on master.
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+(checkout[[:space:]]+-b|switch[[:space:]]+-c)\b'; then
  deny "Esper is local-first: no feature branch. Commit DIRECTLY to master (CLAUDE.md §Version control)."
fi

# 3) A commit must land on master.
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+commit\b'; then
  root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
  branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo master)"
  if [[ "$branch" != "master" ]]; then
    deny "Current branch is '$branch', not master. Esper commits go DIRECTLY to master (CLAUDE.md §Version control)."
  fi
fi

exit 0
