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

echo "All tests passed successfully."
