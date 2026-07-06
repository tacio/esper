"""Rung S audit (offline, measure-first): does the output SIZE of the
`dims-never-fit` shape tasks follow a small basis of INPUT content statistics?

The v3 shape dispatch predicts output dims with a per-axis affine over the
INPUT DIMS only (out = round(k*in_dim + b)). The v3 breakdown isolated the
tasks where that rule fits NO demo (`mem: shape`, train-fit == 0.0) — output
size is content-dependent. This tool tests, per task, whether some per-input
CONTENT STATISTIC (non-bg bbox dims, distinct-colour count, ...) linearly
predicts the output dims across a task's demos, and reports coverage so the
Rung S basis is chosen on evidence, not intuition.

    python tools/shape_from_content.py [train_dump eval_dump]

Defaults to scratch/arc2_{train,eval}_v3.txt. Reads grids from the raw ARC
JSON corpus via the `arg-agi-2-data` symlink (training/ + evaluation/).

The GATE: if one or two statistics explain a clear majority, code Rung S with
exactly that basis; if the class fragments with no dominant statistic, that is
a documented finding (the long-tail hazard) and Rung S is re-scoped, not coded.
"""

import json
import os
import sys
from collections import Counter

CORPUS = os.path.join(os.path.dirname(__file__), "..", "arg-agi-2-data")
SPLIT_DIR = {"train": "training", "eval": "evaluation"}


def task_ids_from_dump(dump_path):
    """The dims-never-fit ids: lines with `mem: shape` and train-fit == 0.0."""
    ids = []
    for line in open(dump_path):
        f = line.split()
        # `  task: <path>  held-out: H  train: T  gap: G  mem: shape`
        if len(f) < 10 or f[0] != "task:":
            continue
        if f[8] == "mem:" and f[9] == "shape" and float(f[5]) == 0.0:
            ids.append(os.path.basename(f[1]).replace(".task", ""))
    return ids


def load_task(split, tid):
    path = os.path.join(CORPUS, SPLIT_DIR[split], f"{tid}.json")
    with open(path) as fh:
        return json.load(fh)


# ---- content statistics (all computed from an INPUT grid alone) ----


def background(grid):
    """Inferred background: the most common colour on the border (falls back to
    the global plurality if the grid is 1xN). Border-majority is the standard
    ARC heuristic and robust to a large interior object."""
    rows, cols = len(grid), len(grid[0])
    border = []
    for c in range(cols):
        border.append(grid[0][c])
        border.append(grid[rows - 1][c])
    for r in range(rows):
        border.append(grid[r][0])
        border.append(grid[r][cols - 1])
    return Counter(border).most_common(1)[0][0]


def nonbg_bbox_dims(grid):
    """Height, width of the bounding box of all non-background cells."""
    bg = background(grid)
    rows, cols = len(grid), len(grid[0])
    r0, r1, c0, c1 = rows, -1, cols, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                r0, r1 = min(r0, r), max(r1, r)
                c0, c1 = min(c0, c), max(c1, c)
    if r1 < 0:
        return 0, 0
    return r1 - r0 + 1, c1 - c0 + 1


def distinct_colours(grid):
    s = set()
    for row in grid:
        s.update(row)
    return len(s)


def stats(grid):
    """The candidate per-input feature vector (row-feature, col-feature)."""
    rows, cols = len(grid), len(grid[0])
    bh, bw = nonbg_bbox_dims(grid)
    nd = distinct_colours(grid)
    # Each candidate: (name, row_feature, col_feature). Axis-symmetric ones
    # share the same scalar on both axes.
    return {
        "in_dim": (rows, cols),
        "nonbg_bbox": (bh, bw),
        "distinct_colours": (nd, nd),
    }


# ---- per-task affine fit of an output dim on a candidate feature ----


def affine_fits(xs, ys, tol=1e-6):
    """Does out = round(k*x + b) hold for ALL demos? Returns (ok, k, b).
    Underdetermined (x constant): accept only if y is also constant (k=0)."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < tol:
        # x does not vary: only a constant output is identifiable here.
        if max(ys) == min(ys):
            return True, 0.0, ys[0]
        return False, None, None
    k = (n * sxy - sx * sy) / denom
    b = (sy - k * sx) / n
    ok = all(round(k * x + b) == y for x, y in zip(xs, ys))
    return ok, k, b


def exact_fits(xs, ys):
    """out == x on every demo (k=1, b=0): the unambiguous, non-overfit rule
    (e.g. output IS the non-bg bbox). Requires the feature to actually vary
    OR to match a varying output, so a constant coincidence isn't counted."""
    return all(x == y for x, y in zip(xs, ys)) and (
        len(set(xs)) > 1 or len(set(ys)) > 1
    )


def audit_task(demos):
    """demos: list of (in_grid, out_grid). For each axis, which candidate
    features fit ALL demos? Returns {name: (affine_r, affine_c, exact_r,
    exact_c)}. `affine` = free round(k*x+b); `exact` = out==x (strong signal)."""
    feats = {}
    cand_names = stats(demos[0][0]).keys()
    for name in cand_names:
        xr = [stats(ig)[name][0] for ig, _ in demos]
        xc = [stats(ig)[name][1] for ig, _ in demos]
        yr = [len(og) for _, og in demos]
        yc = [len(og[0]) for _, og in demos]
        fr, _, _ = affine_fits(xr, yr)
        fc, _, _ = affine_fits(xc, yc)
        feats[name] = (fr, fc, exact_fits(xr, yr), exact_fits(xc, yc))
    return feats


def main(train_dump, eval_dump):
    for split, dump in (("train", train_dump), ("eval", eval_dump)):
        ids = task_ids_from_dump(dump)
        print(f"\n=== {split}: {len(ids)} dims-never-fit tasks ===")
        # Coverage: a candidate "explains" a task if it fits BOTH axes (a full
        # size rule). Also track per-axis and multi-demo identifiability.
        affine_both = Counter()
        exact_both = Counter()
        explained_affine = 0
        explained_exact = 0
        n_multi_size = 0  # tasks with >=2 distinct input sizes (identifiable)
        exact_bbox_ids = []
        unexplained = []
        for tid in ids:
            t = load_task(split, tid)
            demos = [(d["input"], d["output"]) for d in t["train"]]
            in_sizes = {(len(ig), len(ig[0])) for ig, _ in demos}
            if len(in_sizes) >= 2:
                n_multi_size += 1
            feats = audit_task(demos)
            any_affine = False
            any_exact = False
            for name, (fr, fc, er, ec) in feats.items():
                if fr and fc:
                    affine_both[name] += 1
                    any_affine = True
                if er and ec:
                    exact_both[name] += 1
                    any_exact = True
            if any_affine:
                explained_affine += 1
            else:
                unexplained.append(tid)
            if any_exact:
                explained_exact += 1
            if feats["nonbg_bbox"][2] and feats["nonbg_bbox"][3]:
                exact_bbox_ids.append(tid)

        print(f"  tasks with >=2 distinct input sizes: {n_multi_size}"
              f" (identifiable at held-out; rest underdetermined)")
        print("  EXACT rule (out == feature, k=1 b=0 — non-overfit signal):")
        for name, c in exact_both.most_common():
            print(f"    {name:18s} {c:3d}  ({100*c/len(ids):.0f}%)")
        print(f"    -> exact bbox-crop ids: {' '.join(exact_bbox_ids)}")
        print("  FREE-AFFINE rule (round(k*f+b) both axes — overfit-prone at n<=3):")
        for name, c in affine_both.most_common():
            print(f"    {name:18s} {c:3d}  ({100*c/len(ids):.0f}%)")
        print(f"  explained by SOME exact feature:  {explained_exact}"
              f"/{len(ids)}  ({100*explained_exact/len(ids):.0f}%)")
        print(f"  explained by SOME affine feature: {explained_affine}"
              f"/{len(ids)}  ({100*explained_affine/len(ids):.0f}%)")
        print(f"  unexplained by any affine ({len(unexplained)}): "
              + " ".join(unexplained[:40]))


if __name__ == "__main__":
    args = sys.argv[1:]
    td = args[0] if len(args) > 0 else "scratch/arc2_train_v3.txt"
    ed = args[1] if len(args) > 1 else "scratch/arc2_eval_v3.txt"
    main(td, ed)
