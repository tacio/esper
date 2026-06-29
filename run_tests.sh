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

# The driver must also build and run end-to-end (learns flip_h in-context and
# reports held-out generalization).
echo "Running src/main.mojo (end-to-end driver)..."
mojo run -I src src/main.mojo

# NOTE: the old objective-reward benchmark (src/benchmark.mojo) fit a known
# target grid directly — pure memorization. It is replaced by the held-out
# generalization driver (src/arc_solve.mojo) landing at milestone M7; until then
# tests/test_demo_fitness.mojo and tests/test_forward_learning.mojo cover the
# real learn-and-generalize path.

echo "All tests passed successfully."
