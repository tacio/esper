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

# Reuse the canonical writers so the on-disk formats cannot drift from the
# compiler / engine contract.
from arc_compiler import _save_grid, _save_task

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


# ---------------------------------------------------------------------------
# SHAPE-CHANGING transforms (Vision A / Next #1). Unlike everything above these
# return a grid whose dims DIFFER from the input — the output shape is a rule to
# be inferred in-context (never a hand-coded size heuristic). They are the
# ground truth the engine's ShapeMemory must rediscover: a closed-form shape
# rule (out = k*in + b per axis) composed with the AttnGather content gather.
# ---------------------------------------------------------------------------
def _crop1(grid):
    # Drop the 1-cell border: (r-2, c-2). Centred crop -> identity content.
    return [row[1:-1] for row in grid[1:-1]]


def _flip_h_crop1(grid):
    # Crop the 1-cell border, then reverse columns (a flip within the resize).
    return [list(reversed(row[1:-1])) for row in grid[1:-1]]


def _subsample2(grid):
    # Take every 2nd cell on each axis: (ceil(r/2), ceil(c/2)). Exactly r/2, c/2
    # for even dims (the shape rule k=1/2, b=0 is then integer-exact).
    return [
        [grid[i][j] for j in range(0, len(grid[0]), 2)]
        for i in range(0, len(grid), 2)
    ]


def _upscale2(grid):
    # Blocky replication: each cell becomes a 2x2 block -> (2r, 2c). The shape
    # rule is k=2, b=0; the content is out[r][c] = in[r//2][c//2] (a floor
    # gather, which the affine attention gather expresses exactly at sharp
    # temperature: M = I/2, t = 0).
    return [
        [grid[i // 2][j // 2] for j in range(2 * len(grid[0]))]
        for i in range(2 * len(grid))
    ]


def _tile2(grid):
    # Tile the grid 2x2 -> (2r, 2c). Same shape rule as upscale2 (k=2, b=0) but
    # the content out[r][c] = in[r % rows][c % cols] is a sawtooth — genuinely
    # NON-affine, the family that forces modular (wrapped) source addressing.
    return [
        [grid[i % len(grid)][j % len(grid[0])] for j in range(2 * len(grid[0]))]
        for i in range(2 * len(grid))
    ]


TRANSFORMS = {
    "identity": _identity,
    "flip_h": _flip_h,
    "flip_v": _flip_v,
    "transpose": _transpose,
    "recolor": _recolor,
    "shift": _shift,
}

# Shape-changing families kept in a SEPARATE table: they need their own group
# generator (input size must VARY across a task's demos so the shape rule is
# identifiable, not memorized). `subsample2` requires even input dims.
# COLOUR-ON-SHAPE families (Vision A / Next #1, Rung C): a shape change composed
# with a cellwise recolor. Colour commutes with the copy gather, so applying
# `_recolor` (the cyclic +1 palette shift) before the shape transform is the
# same as after — the ground truth the ShapeGeomColorComposedMemory must
# rediscover as (shape rule, written colour table V, geometry).
def _recolor_crop1(grid):
    return _crop1(_recolor(grid))


def _recolor_subsample2(grid):
    return _subsample2(_recolor(grid))


def _recolor_upscale2(grid):
    return _upscale2(_recolor(grid))


def _recolor_tile2(grid):
    return _tile2(_recolor(grid))


SHAPE_TRANSFORMS = {
    "crop1": _crop1,
    "flip_h_crop1": _flip_h_crop1,
    "subsample2": _subsample2,
    "upscale2": _upscale2,
    "tile2": _tile2,
    "recolor_crop1": _recolor_crop1,
    "recolor_subsample2": _recolor_subsample2,
    "recolor_upscale2": _recolor_upscale2,
    "recolor_tile2": _recolor_tile2,
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


def generate_task_groups(transform, out_dir, num_tasks, n_train, rows, cols, seed):
    """
    Emit `num_tasks` ARC-shaped *task bundles* for `transform`, one `.task` file
    each (written by `arc_compiler._save_task`). A task = `n_train` random
    (input, transform(input)) demonstration pairs plus one DISTINCT held-out test
    pair. This feeds the held-out generalization driver (`src/arc_solve.mojo`):
    the engine fits the operator on the train pairs and is scored on the unseen
    test pair. Returns the list of bundle paths written.
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

    paths = []
    for t in range(num_tasks):
        train = []
        for _ in range(n_train):
            grid_in = _random_grid(rows, cols, rng)
            train.append((grid_in, fn(grid_in)))
        test_in = _random_grid(rows, cols, rng)
        test = [(test_in, fn(test_in))]

        path = os.path.join(out_dir, "%s_%d.task" % (transform, t))
        _save_task(train, test, path)
        paths.append(path)

    return paths


def _rand_shape_size(rng, even):
    """A random grid size in [4, 8] per axis; even-only when `even` (subsample)."""
    if even:
        r = 4 + 2 * rng.randrange(3)  # {4, 6, 8}
        c = 4 + 2 * rng.randrange(3)
    else:
        r = 4 + rng.randrange(5)  # [4, 8]
        c = 4 + rng.randrange(5)
    return r, c


def generate_shape_task_groups(transform, out_dir, num_tasks, n_train, seed):
    """Emit `num_tasks` SHAPE-CHANGING task bundles (`.task`) for `transform`.

    Unlike `generate_task_groups`, each demo (and the held-out test) is drawn at
    a RANDOM input size, so the per-axis shape rule out = k*in + b is genuinely
    IDENTIFIABLE from the demos (>= 2 distinct sizes) rather than memorized, and
    the unseen-size test pair is an uncheatable generalization probe. Returns the
    bundle paths written.
    """
    if transform not in SHAPE_TRANSFORMS:
        raise ValueError(
            "Unknown shape transform %r; choose from %s"
            % (transform, ", ".join(sorted(SHAPE_TRANSFORMS)))
        )
    os.makedirs(out_dir, exist_ok=True)
    fn = SHAPE_TRANSFORMS[transform]
    even = transform in ("subsample2", "recolor_subsample2")
    rng = _random.Random(seed)

    paths = []
    for t in range(num_tasks):
        train = []
        for _ in range(n_train):
            r, c = _rand_shape_size(rng, even)
            grid_in = _random_grid(r, c, rng)
            train.append((grid_in, fn(grid_in)))
        r, c = _rand_shape_size(rng, even)
        test_in = _random_grid(r, c, rng)
        test = [(test_in, fn(test_in))]

        path = os.path.join(out_dir, "%s_%d.task" % (transform, t))
        _save_task(train, test, path)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# LOCAL-CONTENT transforms (Rung A build, Approach 1b). Same-shape per-cell rules
# keyed on a bounded local SIGNATURE (centre colour, #Moore-8 neighbours DIFFERING
# from centre) — the class the copy-gather+colour memory under-applies (the Rung A
# near-miss audit). The ground truth LocalWriteComposedMemory must rediscover as a
# written signature table composed on the gather. They need STRUCTURED (blocky)
# grids — a background with solid regions — so the local signatures are
# identifiable (uniform-random grids make every cell a border).
# ---------------------------------------------------------------------------
def _blocky_grid(rows, cols, rng):
    g = [[0] * cols for _ in range(rows)]
    for _ in range(rng.randint(2, 4)):
        col = rng.randint(1, 2)
        h = rng.randint(2, max(2, rows - 2))
        w = rng.randint(2, max(2, cols - 2))
        r0 = rng.randrange(0, rows - h + 1)
        c0 = rng.randrange(0, cols - w + 1)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                g[r][c] = col
    for _ in range(rng.randint(1, 3)):
        r = rng.randrange(1, rows - 1)
        c = rng.randrange(1, cols - 1)
        if all(
            g[r + dr][c + dc] != 0 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
        ):
            g[r][c] = 0  # a punched enclosed-background hole
    return g


def _diff_count(grid, r, c):
    rows, cols = len(grid), len(grid[0])
    ctr = grid[r][c]
    return sum(
        1
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
        and grid[(r + dr) % rows][(c + dc) % cols] != ctr
    )


def _outline(grid):
    # Any cell bordering a different colour -> 9 (edge / object-colour class).
    rows, cols = len(grid), len(grid[0])
    return [
        [9 if _diff_count(grid, r, c) >= 1 else grid[r][c] for c in range(cols)]
        for r in range(rows)
    ]


def _fill_enclosed(grid):
    # A fully-enclosed background cell (all 8 neighbours non-background) -> 8
    # (the background-fill class; a single well-covered signature, (0, 8)).
    rows, cols = len(grid), len(grid[0])
    return [
        [
            8 if grid[r][c] == 0 and _diff_count(grid, r, c) == 8 else grid[r][c]
            for c in range(cols)
        ]
        for r in range(rows)
    ]


LOCAL_TRANSFORMS = {
    "outline": _outline,
    "fill_enclosed": _fill_enclosed,
}


def generate_local_task_groups(
    transform, out_dir, num_tasks, n_train, rows, cols, seed
):
    """Emit `num_tasks` LOCAL-CONTENT task bundles on STRUCTURED blocky grids.

    Same-shape (so they route through the same-shape solver path), but the rule
    is a local-neighbourhood signature the copy-gather cannot express — the Rung A
    build's ground truth. Held-out test grid is a fresh blocky layout.
    """
    if transform not in LOCAL_TRANSFORMS:
        raise ValueError(
            "Unknown local transform %r; choose from %s"
            % (transform, ", ".join(sorted(LOCAL_TRANSFORMS)))
        )
    os.makedirs(out_dir, exist_ok=True)
    fn = LOCAL_TRANSFORMS[transform]
    rng = _random.Random(seed)

    paths = []
    for t in range(num_tasks):
        train = []
        for _ in range(n_train):
            grid_in = _blocky_grid(rows, cols, rng)
            train.append((grid_in, fn(grid_in)))
        test_in = _blocky_grid(rows, cols, rng)
        test = [(test_in, fn(test_in))]

        path = os.path.join(out_dir, "%s_%d.task" % (transform, t))
        _save_task(train, test, path)
        paths.append(path)

    return paths


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
