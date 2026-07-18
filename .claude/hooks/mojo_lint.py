#!/usr/bin/env python3
"""Esper PreToolUse guardrail for Mojo 1.0.0b2 discipline.

Reads a Claude Code PreToolUse hook payload on stdin (Edit / Write / MultiEdit).
For .mojo files under src/ or tests/ it lints only the *added* text:

  * HARD idiom violations (deterministic, low false-positive) -> BLOCK the edit
    by emitting permissionDecision: "deny".
  * FUZZY / context-dependent smells -> WARN via hookSpecificOutput.additionalContext
    (non-blocking) so Claude sees the note but the edit proceeds.

This is tooling that lives in .claude/ — it is NOT part of the pure-Mojo src/
runtime path, so it does not itself violate the "no Python in src/" rule.

The rule catalog mirrors CLAUDE.md §Toolchain and §Hard constraints.
"""

import json
import re
import sys

# --- Rule tables ------------------------------------------------------------
# Each HARD rule: (compiled regex, human message with the fix). Matched against
# comment-stripped added lines. Regexes are anchored/word-bounded to keep the
# false-positive rate near zero, because these BLOCK the edit.

HARD_RULES = [
    (
        re.compile(r"^\s*fn\s+"),
        "`fn` was removed in Mojo 1.0 — use `def` (add `raises` if it can raise).",
    ),
    (
        re.compile(r"^\s*alias\s+\w"),
        "`alias` was renamed in Mojo 1.0 — use `comptime NAME = ...` for compile-time constants.",
    ),
    (
        re.compile(r"\b__moveinit__\b"),
        "Do not hand-write `__moveinit__` — derive `(Movable)` / `(Copyable, Movable)` instead.",
    ),
    (
        re.compile(r"\bTensor\b"),
        "The stdlib `Tensor` type was removed in Mojo 1.0 — use raw `UnsafePointer` / SIMD slices.",
    ),
    # Non-std-qualified stdlib imports. `from std.memory import ...` is fine;
    # `from memory import ...` is not.
    (
        re.compile(r"^\s*from\s+(?!std\.)(memory|math|sys|collections|random|gpu)\b"),
        "Stdlib imports must be `std.`-qualified in Mojo 1.0 (e.g. `from std.memory import ...`).",
    ),
]

# src/-only HARD rules: nothing under src/ may reach into Python or an ML lib —
# src/ is pure Mojo by construction (CLAUDE.md §Hard constraints #1, #2).
SRC_ONLY_HARD_RULES = [
    (
        re.compile(r"\b(from\s+python\s+import|Python\.import_module|PythonObject)\b"),
        "No Python interop on the runtime path — Python is confined to tools/ (offline only).",
    ),
    (
        re.compile(r"^\s*(import|from)\s+(torch|tensorflow|jax|numpy|sklearn)\b"),
        "Zero external ML libraries in src/ — learning is Evolution Strategies, not an ML framework.",
    ),
    (
        re.compile(r"^\s*import\s+subprocess\b"),
        "No `subprocess` on the runtime path — src/ is pure Mojo by construction.",
    ),
]

# FUZZY rules -> WARN only (context-dependent; a regex can misjudge them).
FUZZY_RULES = [
    (
        re.compile(r"\.round\("),
        "`.round()` method form — Mojo 1.0 has no SIMD `.round()`; use the free `round()` from `std.math`.",
    ),
]

# Hot-loop files where a per-iteration heap allocation would violate the
# "no dynamic allocation in hot loops" rule (reuse ESWorkspace instead).
HOT_FILES = ("esper_evolution.mojo", "arc_io.mojo", "hope.mojo")


def strip_comment(line: str) -> str:
    """Drop a full-line comment or an inline ` # ...` tail (best-effort, so we
    don't lint text that only appears inside a comment)."""
    if line.lstrip().startswith("#"):
        return ""
    return re.sub(r"\s+#.*$", "", line)


def added_text(tool_name: str, tool_input: dict) -> str:
    """Return only the text this edit ADDS, so we never block on pre-existing
    code the user isn't touching."""
    if tool_name == "Write":
        return tool_input.get("content", "") or ""
    if tool_name == "Edit":
        return tool_input.get("new_string", "") or ""
    if tool_name == "MultiEdit":
        return "\n".join(
            (e or {}).get("new_string", "") or "" for e in tool_input.get("edits", [])
        )
    return ""


def lint(path: str, text: str):
    """Return (blocks, warns) — lists of human-readable strings."""
    in_src = "/src/" in path or path.startswith("src/")
    lines = [strip_comment(ln) for ln in text.splitlines()]

    blocks, warns = [], []
    hard = HARD_RULES + (SRC_ONLY_HARD_RULES if in_src else [])

    for i, line in enumerate(lines, 1):
        for rx, msg in hard:
            if rx.search(line):
                blocks.append(f"L{i}: {msg}")
        for rx, msg in FUZZY_RULES:
            if rx.search(line):
                warns.append(f"L{i}: {msg}")

    # --- content-level fuzzy heuristics (whole added block) ---
    # A vectorized loop stepping by `nelts` should be paired with a scalar
    # remainder loop. Heuristic: the snippet has an `nelts`-step range but only
    # one `range(` total, so no separate tail loop is in view. (Two+ ranges =
    # assume main + remainder; low-signal, advisory only.)
    if (
        re.search(r"range\([^)]*,\s*[\w.]*nelts[\w.]*\s*\)", text)
        and text.count("range(") < 2
    ):
        warns.append(
            "vectorized `range(..., nelts)` loop with no visible scalar-remainder loop — "
            "confirm the tail is handled (see the three-part SIMD shape in CLAUDE.md)."
        )

    # Heap allocation inside a hot-loop module.
    if any(path.endswith(hf) for hf in HOT_FILES) and re.search(
        r"\balloc\[|\.free\(\)|List\(", text
    ):
        warns.append(
            "heap op (`alloc[` / `.free()` / `List(`) in a hot-loop module — "
            "reuse the pre-allocated `ESWorkspace`; never alloc/free per ES iteration."
        )

    # __del__ null-guard (advisory: hard to bound scope from a snippet).
    if "def __del__" in text and re.search(r"if\s+self\.\w+\s*:", text):
        warns.append(
            "possible null-guard inside `__del__` — `UnsafePointer` is non-null by design; "
            "free unconditionally, don't guard with `if self.data:`."
        )

    return blocks, warns


def emit_allow_with_warns(warns):
    if warns:
        note = "⚠ Esper mojo-lint (advisory):\n  - " + "\n  - ".join(warns)
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "additionalContext": note,
                    }
                }
            )
        )
    sys.exit(0)


def emit_deny(blocks, warns):
    reason = (
        "Esper discipline (Mojo 1.0.0b2) — edit blocked:\n  - "
        + "\n  - ".join(blocks)
    )
    if warns:
        reason += "\nAlso note (non-blocking):\n  - " + "\n  - ".join(warns)
    reason += "\nFix the added lines and retry. See CLAUDE.md §Toolchain / §Hard constraints."
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    sys.exit(0)


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # not our concern — let the edit through

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    path = tool_input.get("file_path", "") or ""

    if tool_name not in ("Edit", "Write", "MultiEdit"):
        sys.exit(0)
    if not path.endswith(".mojo"):
        sys.exit(0)
    if not (("/src/" in path or "/tests/" in path)
            or path.startswith("src/") or path.startswith("tests/")):
        sys.exit(0)

    blocks, warns = lint(path, added_text(tool_name, tool_input))
    if blocks:
        emit_deny(blocks, warns)
    emit_allow_with_warns(warns)


if __name__ == "__main__":
    main()
