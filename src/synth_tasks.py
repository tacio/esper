"""
Synthetic, deterministic grid-transformation task generator for Esper.

Esper targets ARC-AGI, where every task has a unique, machine-checkable
ground-truth output grid -- i.e. an *objective reward* (exact pixel match).
Real ARC tasks also need a full forward pass and a primitive library, which the
engine does not have yet. This generator instead emits *single deterministic
transforms* (recolor, flip, transpose, shift). Each task therefore has an
unambiguous target grid, so the Evolution-Strategy loop can be scored objectively
(negative MSE via `calculate_fitness`, exact-match % via `exact_match`) and we can
prove the learning loop actually converges before tackling full ARC reasoning.

Grids are written in the exact `.bin` layout the Mojo engine reads
(`load_arc_grid` in `arc_io.mojo`): two little-endian int64 (rows, cols) followed
by the flattened grid as float32. We reuse `_save_grid` from `arc_compiler.py`
so the writer stays the single source of truth for that format.

Usage:
    python src/synth_tasks.py --transform flip_h --out data_bin --count 8
    python src/synth_tasks.py --transform recolor --rows 6 --cols 6 --seed 0
"""

import argparse
import os
import random as _random

# Reuse the canonical .bin writer so the on-disk format cannot drift from the
# compiler / engine contract.
from arc_compiler import _save_grid

# ARC grids use integer "colors" 0-9.
NUM_COLORS = 10


# ---------------------------------------------------------------------------
# Deterministic transforms. Each takes a 2-D list-of-lists grid and returns a
# new grid of the same shape (so input and target share dimensions, which keeps
# the engine's fixed-size fast-weight buffer valid).
# ---------------------------------------------------------------------------
def _identity(grid):
    return [row[:] for row in grid]


def _flip_h(grid):
    return [list(reversed(row)) for row in grid]


def _flip_v(grid):
    return [row[:] for row in reversed(grid)]


def _transpose(grid):
    # Square grids only keep the same shape under transpose; the generator
    # enforces rows == cols when this transform is selected.
    return [list(col) for col in zip(*grid)]


def _recolor(grid):
    # Fixed cyclic palette shift: color c -> (c + 1) % NUM_COLORS.
    return [[(c + 1) % NUM_COLORS for c in row] for row in grid]


def _shift(grid):
    # Roll every row right by one (wrap-around). Deterministic and shape-stable.
    return [[row[-1]] + row[:-1] for row in grid]


TRANSFORMS = {
    "identity": _identity,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "transpose": _transpose,
    "recolor": _recolor,
    "shift": _shift,
}


def _random_grid(rows, cols, rng):
    return [[rng.randrange(NUM_COLORS) for _ in range(cols)] for _ in range(rows)]


def generate_tasks(transform, out_dir, count, rows, cols, seed):
    """
    Generate `count` (input, target) grid pairs for `transform` and write them as
    `.bin` files into `out_dir`. Returns the list of (input_path, target_path)
    tuples written.
    """
    if transform not in TRANSFORMS:
        raise ValueError(
            "Unknown transform %r; choose from %s"
            % (transform, ", ".join(sorted(TRANSFORMS)))
        )
    if transform == "transpose" and rows != cols:
        raise ValueError("transpose requires a square grid (rows == cols)")

    os.makedirs(out_dir, exist_ok=True)
    fn = TRANSFORMS[transform]
    rng = _random.Random(seed)

    pairs = []
    for i in range(count):
        grid_in = _random_grid(rows, cols, rng)
        grid_out = fn(grid_in)

        in_path = os.path.join(out_dir, "%s_%d_in.bin" % (transform, i))
        out_path = os.path.join(out_dir, "%s_%d_out.bin" % (transform, i))
        _save_grid(grid_in, in_path)
        _save_grid(grid_out, out_path)
        pairs.append((in_path, out_path))

    return pairs


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transform",
        default="flip_h",
        choices=sorted(TRANSFORMS),
        help="Which deterministic transform maps input -> target.",
    )
    p.add_argument("--out", default="data_bin", help="Output directory for .bin files.")
    p.add_argument("--count", type=int, default=8, help="Number of task pairs to emit.")
    p.add_argument("--rows", type=int, default=6, help="Grid rows.")
    p.add_argument("--cols", type=int, default=6, help="Grid cols.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible tasks).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = generate_tasks(
        args.transform, args.out, args.count, args.rows, args.cols, args.seed
    )
    print(
        "Generated %d %s task pair(s) (%dx%d) in %s/"
        % (len(written), args.transform, args.rows, args.cols, args.out)
    )
    for in_path, out_path in written:
        print("  %s -> %s" % (os.path.basename(in_path), os.path.basename(out_path)))
