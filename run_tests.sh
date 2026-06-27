#!/usr/bin/env bash
set -euo pipefail

echo "Starting Esper Test Suite..."

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p build

# Generate a known sample .bin (a 2x3 grid) for the IO round-trip test.
python - <<'PY'
import sys
sys.path.insert(0, "src")
from arc_compiler import _save_grid
_save_grid([[1, 2, 3], [4, 5, 6]], "build/sample_in.bin")
print("Generated build/sample_in.bin")
PY

# Run every tests/test_*.mojo with src on the import path.
for test_file in tests/test_*.mojo; do
    echo "Running ${test_file}..."
    mojo run -I src "${test_file}"
done

# The driver must also build and run end-to-end.
echo "Running src/main.mojo (end-to-end driver)..."
mojo run -I src src/main.mojo

# Objective-reward benchmark: generate a small deterministic task set, then fit
# each target with the ES loop and report the aggregate solve rate.
echo "Running objective-reward benchmark (src/benchmark.mojo)..."
BENCH_DIR="$(mktemp -d)"
trap 'rm -rf "$BENCH_DIR"' EXIT
python - "$BENCH_DIR" <<'PY'
import sys
sys.path.insert(0, "src")
from synth_tasks import generate_tasks
generate_tasks("flip_h", sys.argv[1], count=4, rows=4, cols=4, seed=0)
print("Generated benchmark tasks in", sys.argv[1])
PY
mojo run -I src src/benchmark.mojo "$BENCH_DIR"/flip_h_*_out.bin

echo "All tests passed successfully."
