#!/usr/bin/env bash
set -euo pipefail

# Suite tier: `full` (default) runs everything; `fast` skips the heavy meta-fit
# milestone proofs (tests tagged `# suite-tier: full`) for a quick local gate.
# The `fast` tier still covers every code path at FULL budget (structural, the ES
# operator fit, and the self-mod meta-fit core via test_selfmod_memory) — it only
# defers the two large-scale proofs (grid_context ~178s, delta_selfmod ~64s), so
# it never runs a weakened threshold. See CLAUDE.md "Testing".
TIER="${1:-full}"
if [[ "$TIER" != "full" && "$TIER" != "fast" ]]; then
    echo "usage: $0 [full|fast]  (got '$TIER')" >&2
    exit 2
fi

echo "Starting Esper Test Suite (tier: ${TIER})..."

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p build

# Generate a known sample .bin (a 2x3 grid) for the IO round-trip test, and a
# sample .task bundle (2 train + 1 test flip_h pairs) for the task-loader test.
python - <<'PY'
import sys
sys.path.insert(0, "tools")
from arc_compiler import _save_grid, _save_task
_save_grid([[1, 2, 3], [4, 5, 6]], "build/sample_in.bin")
print("Generated build/sample_in.bin")
_save_task(
    [([[1, 2], [3, 4]], [[2, 1], [4, 3]]), ([[5, 6], [7, 8]], [[6, 5], [8, 7]])],
    [([[9, 0], [1, 2]], [[0, 9], [2, 1]])],
    "build/sample.task",
)
print("Generated build/sample.task")
PY

# Run every tests/test_*.mojo with src on the import path. In the `fast` tier,
# skip files that tag themselves `# suite-tier: full` (heavy milestone proofs).
for test_file in tests/test_*.mojo; do
    if [[ "$TIER" == "fast" ]] && grep -q '^# suite-tier: full' "${test_file}"; then
        echo "Skipping ${test_file} (full-tier only)..."
        continue
    fi
    echo "Running ${test_file}..."
    mojo run -I src "${test_file}"
done

# The driver must also build and run end-to-end (learns flip_h in-context and
# reports held-out generalization).
echo "Running src/main.mojo (end-to-end driver)..."
mojo run -I src src/main.mojo

# Held-out generalization driver: generate a few task bundles, fit the operator
# on each task's train pairs, and score only on the unseen test pair. This
# replaces the old src/benchmark.mojo (which memorized a known target grid).
echo "Running held-out generalization driver (src/arc_solve.mojo)..."
GEN_DIR="$(mktemp -d)"
trap 'rm -rf "$GEN_DIR"' EXIT
# The full tier adds SHAPE-CHANGING bundles so the driver's shape dispatch
# (ShapeGeomColorComposedMemory + fit_shape_color) runs end-to-end in CI: one
# pure-shape (crop1) and one colour-on-shape (recolor_crop1, Rung C) bundle. The
# fast gate keeps the same-shape-only leg (~a minute cheaper).
python - "$GEN_DIR" "$TIER" <<'PY'
import sys
sys.path.insert(0, "tools")
from synth_tasks import generate_task_groups, generate_shape_task_groups
generate_task_groups("flip_h", sys.argv[1], num_tasks=2, n_train=6, rows=4, cols=4, seed=0)
generate_task_groups("recolor", sys.argv[1], num_tasks=1, n_train=6, rows=4, cols=4, seed=1)
if sys.argv[2] == "full":
    generate_shape_task_groups("crop1", sys.argv[1], num_tasks=1, n_train=6, seed=2)
    generate_shape_task_groups("recolor_crop1", sys.argv[1], num_tasks=1, n_train=6, seed=3)
print("Generated task bundles in", sys.argv[1])
PY
mojo run -I src src/arc_solve.mojo "$GEN_DIR"/*.task

echo "All tests passed successfully."
