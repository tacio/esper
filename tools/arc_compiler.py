import json
import struct
import numpy as np
import os
import sys

def compile_arc_json(json_path: str, output_dir: str):
    """
    Parses an ARC JSON file and converts each input/output grid into a raw .bin file.
    The binary format consists of two 64-bit integers (rows, cols) followed by
    the flattened grid as 32-bit floats.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Use splitext so task ids containing dots are not truncated.
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # Process training pairs
    if 'train' in data:
        for i, pair in enumerate(data['train']):
            _save_grid(pair['input'], f"{output_dir}/{base_name}_train_{i}_in.bin")
            _save_grid(pair['output'], f"{output_dir}/{base_name}_train_{i}_out.bin")

    # Process test pairs
    if 'test' in data:
        for i, pair in enumerate(data['test']):
            _save_grid(pair['input'], f"{output_dir}/{base_name}_test_{i}_in.bin")
            if 'output' in pair:
                _save_grid(pair['output'], f"{output_dir}/{base_name}_test_{i}_out.bin")

def _write_grid(f, grid):
    """Write one grid-block to an open binary file: two little-endian int64
    (rows, cols) followed by the flattened grid as float32. This is the single
    grid encoding shared by the per-grid `.bin` format and the `.task` bundle."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    flat_data = np.array(grid, dtype=np.float32).flatten()
    f.write(struct.pack('<qq', rows, cols))
    f.write(flat_data.tobytes())


def _save_grid(grid, out_path):
    with open(out_path, 'wb') as f:
        _write_grid(f, grid)


def _save_task(train_pairs, test_pairs, out_path):
    """Write a whole task as one binary bundle (a `.task` file) the Mojo engine
    loads in a single read (see `load_arc_task` in `arc_io.mojo`):

        [n_train: int64][n_test: int64]
        then 2*n_train + 2*n_test grid-blocks in order:
        train_in, train_out, ..., test_in, test_out

    Each (input, output) pair is a 2-tuple of 2-D list-of-lists grids. Bundling
    avoids Mojo-side directory globbing and text parsing — the shell globs the
    `.task` files into the driver's argv instead.
    """
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<qq', len(train_pairs), len(test_pairs)))
        for grid_in, grid_out in train_pairs:
            _write_grid(f, grid_in)
            _write_grid(f, grid_out)
        for grid_in, grid_out in test_pairs:
            _write_grid(f, grid_in)
            _write_grid(f, grid_out)

def compile_task_to_bundle(json_path, out_path):
    """Compile one ARC-AGI-2 task JSON into a single `.task` bundle that
    `load_arc_task` reads. Each pair becomes an (input, output) 2-tuple.

    Real ARC-AGI-2 corpus tasks (training/ and evaluation/) ship the test
    output, so the held-out test pair is fully usable for scoring. A test pair
    without an `output` (a private/competition split) is skipped, so a task with
    no scorable test pair is dropped (returns False) rather than written with a
    bogus target.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    train_pairs = [(p["input"], p["output"]) for p in data.get("train", [])]
    test_pairs = [
        (p["input"], p["output"]) for p in data.get("test", []) if "output" in p
    ]
    if not test_pairs:
        return False

    _save_task(train_pairs, test_pairs, out_path)
    return True


def compile_arc_dir(in_dir, out_dir):
    """Batch-compile every `*.json` task in `in_dir` into a `{task_id}.task`
    bundle in `out_dir`. The shell then globs `out_dir/*.task` into the
    `src/arc_solve.mojo` driver's argv. Returns (written, skipped)."""
    os.makedirs(out_dir, exist_ok=True)
    written, skipped = 0, 0
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".json"):
            continue
        task_id = os.path.splitext(name)[0]
        ok = compile_task_to_bundle(
            os.path.join(in_dir, name), os.path.join(out_dir, f"{task_id}.task")
        )
        if ok:
            written += 1
        else:
            skipped += 1
    return written, skipped


if __name__ == "__main__":
    # Batch ingest: compile a directory of ARC-AGI-2 task JSON into `.task`
    # bundles. The corpus is NOT in the repo (download separately; `data_bin/`
    # is gitignored). Usage:
    #     python src/arc_compiler.py <in_dir> <out_dir>
    if len(sys.argv) != 3:
        sys.exit("usage: python src/arc_compiler.py <json_dir> <out_dir>")
    written, skipped = compile_arc_dir(sys.argv[1], sys.argv[2])
    print(f"compiled {written} task bundles -> {sys.argv[2]} ({skipped} skipped)")
