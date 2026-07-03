"""Corpus funnel diagnostics for `.task` bundle directories (offline tool).

Two modes:
  python tools/corpus_stats.py <task_dir> [...]
      Header-scan funnel: how many tasks are same-shape (reachable by the
      same-shape memories at all), the grid-size distribution, demo counts.
  python tools/corpus_stats.py --results <results.txt>
      Breakdown of an eval_parallel.sh result file: buckets tasks by
      train-fit vs held-out. `train ~ 1, held-out low` = fits demos but does
      not generalize; `train ~ 0` = expressiveness gap (the memory cannot even
      fit the demos). This breakdown, not the solve count, is what prioritizes
      the roadmap (content composition vs shape change).

Reads bundle headers only (never the float payloads) - fast on 1000 tasks.
"""

import glob
import struct
import sys


def read_task_shapes(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    n_train, n_test = struct.unpack_from("<qq", data, off)
    off += 16
    pairs = []
    for _ in range(n_train + n_test):
        pair = []
        for _ in range(2):
            r, c = struct.unpack_from("<qq", data, off)
            off += 16 + 4 * r * c
            pair.append((r, c))
        pairs.append(tuple(pair))
    return n_train, n_test, pairs


def scan_dir(task_dir):
    files = sorted(glob.glob(f"{task_dir}/*.task"))
    if not files:
        print(f"no .task bundles in {task_dir}")
        return
    same, sizes, demo_counts = 0, [], []
    buckets = {"<=100": 0, "101-400": 0, "401-900": 0, ">900": 0}
    for p in files:
        n_tr, _, pairs = read_task_shapes(p)
        demo_counts.append(n_tr)
        if all(i == o for i, o in pairs):
            same += 1
            mx = max(r * c for pr in pairs for r, c in pr)
            sizes.append(mx)
            if mx <= 100:
                buckets["<=100"] += 1
            elif mx <= 400:
                buckets["101-400"] += 1
            elif mx <= 900:
                buckets["401-900"] += 1
            else:
                buckets[">900"] += 1
    sizes.sort()
    demo_counts.sort()
    n = len(files)
    print(f"{task_dir}: {n} tasks")
    print(f"  same-shape (all pairs):  {same}  ({100 * same / n:.1f}%)")
    if sizes:
        print(
            f"  same-shape max-cells:    median {sizes[len(sizes) // 2]},"
            f" p90 {sizes[int(len(sizes) * 0.9)]}, max {sizes[-1]}"
        )
        print(f"  size buckets (cells):    {buckets}")
    print(
        f"  demos per task:          median {demo_counts[len(demo_counts) // 2]},"
        f" min {demo_counts[0]}, max {demo_counts[-1]}"
    )


def scan_results(path):
    rows = []
    for line in open(path):
        if not line.startswith("  task:"):
            continue
        f = line.split()
        rows.append((f[1], float(f[3]), float(f[5])))  # path, held-out, train
    if not rows:
        print(f"no per-task lines in {path}")
        return
    n = len(rows)
    solved = sum(1 for _, ho, _ in rows if ho >= 0.99)
    print(f"{path}: {n} tasks scored")
    print(f"  solved (held-out >= .99):        {solved}  ({100 * solved / n:.2f}%)")

    # The train-fit diagnostics only mean something on tasks the same-shape
    # memory can express AT ALL - filter via the bundle headers (shape-skipped
    # tasks print train 0.0, which would inflate the expressiveness bucket).
    same = []
    for p, ho, tr in rows:
        try:
            _, _, pairs = read_task_shapes(p)
        except OSError:
            continue
        if all(i == o for i, o in pairs):
            same.append((p, ho, tr))
    m = len(same)
    if not m:
        print("  (no same-shape tasks found for the fit diagnostics)")
        return
    expr_gap = sum(1 for _, _, tr in same if tr < 0.3)
    memorize = sum(1 for _, ho, tr in same if tr >= 0.7 and ho < 0.5)
    partial = sum(1 for _, ho, _ in same if 0.5 <= ho < 0.99)
    print(f"  same-shape subset: {m} tasks")
    print(f"    expressiveness gap (train < .3): {expr_gap}  ({100 * expr_gap / m:.1f}%)")
    print(f"    fits-but-no-generalize (train >= .7, held-out < .5): {memorize}")
    print(f"    near misses (.5 <= held-out < .99): {partial}")
    top = sorted(same, key=lambda r: -r[1])[:15]
    print("    top held-out tasks:")
    for p, ho, tr in top:
        print(f"      {p}  held-out {ho:.3f}  train {tr:.3f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    if args[0] == "--results":
        for p in args[1:]:
            scan_results(p)
    else:
        for d in args:
            scan_dir(d)
