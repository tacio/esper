"""Rung A audit (offline, measure-first): characterize WHERE the same-shape
near-miss tasks (held-out 0.90-0.99) go wrong, so the audit — not intuition —
names the mechanism.

Consumes the `DIFF` records emitted by `arc_solve --diff` (input / fitted
prediction / truth grids for each same-shape test pair), computes a per-task
failure fingerprint, and clusters the 100 near-miss tasks by dominant pattern:

  border          : wrong cells sit on the outer ring (edge/frame effect)
  colour-swap     : wrong cells are a consistent (true<->pred) colour remap
                    (the colour table is a hair off — few-demo tie territory)
  region          : wrong cells form ONE contiguous blob (localized rule)
  under-applied   : at wrong cells pred == INPUT but true != input — the memory
                    left them unchanged; the true rule is GATED to a sub-region
                    the global gather never reached  (the mask/gate candidate)
  over-applied    : at wrong cells true == INPUT but pred != input — the memory
                    changed cells that should have stayed
  scattered       : many disconnected single-cell errors (no structure)

    python tools/near_miss_audit.py [scratch/rungA/out_*.txt ...]

Reports cluster sizes + representative ids. The GATE: a dominant, coherent
cluster names one mechanism to build; fragmentation across many small clusters
is itself the (documented) finding.
"""

import glob
import sys
from collections import Counter


def parse(paths):
    """Yield (task_id, rows, cols, in_grid, pred_grid, true_grid) per DIFF."""
    cur = {}
    for path in paths:
        for line in open(path):
            line = line.strip()
            if line.startswith("DIFF task:"):
                f = line.split()
                cur = {
                    "id": f[2].split("/")[-1].replace(".task", ""),
                    "rows": int(f[4]),
                    "cols": int(f[6]),
                }
            elif line.startswith("DIFF in:"):
                cur["in"] = [int(x) for x in line[len("DIFF in:"):].split()]
            elif line.startswith("DIFF pred:"):
                cur["pred"] = [int(x) for x in line[len("DIFF pred:"):].split()]
            elif line.startswith("DIFF true:"):
                cur["true"] = [int(x) for x in line[len("DIFF true:"):].split()]
                if {"in", "pred", "true"} <= cur.keys():
                    yield cur
                cur = {}


def components(mask, R, C):
    """4-connected components of a boolean mask; returns list of sizes."""
    seen = [False] * (R * C)
    sizes = []
    for i in range(R * C):
        if mask[i] and not seen[i]:
            sz = 0
            st = [i]
            seen[i] = True
            while st:
                j = st.pop()
                sz += 1
                r, c = divmod(j, C)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C:
                        k = nr * C + nc
                        if mask[k] and not seen[k]:
                            seen[k] = True
                            st.append(k)
            sizes.append(sz)
    return sorted(sizes, reverse=True)


def classify(rec):
    R, C = rec["rows"], rec["cols"]
    ig, pg, tg = rec["in"], rec["pred"], rec["true"]
    n = R * C
    wrong = [pg[i] != tg[i] for i in range(n)]
    nw = sum(wrong)
    if nw == 0:
        return "exact", {}
    idxs = [i for i in range(n) if wrong[i]]
    # border fraction
    on_border = sum(
        1 for i in idxs
        if i // C in (0, R - 1) or i % C in (0, C - 1)
    )
    # colour-swap: distinct (true, pred) pairs on wrong cells
    swaps = Counter((tg[i], pg[i]) for i in idxs)
    n_swap_kinds = len(swaps)
    # under/over application vs the INPUT
    under = sum(1 for i in idxs if pg[i] == ig[i] and tg[i] != ig[i])
    over = sum(1 for i in idxs if tg[i] == ig[i] and pg[i] != ig[i])
    # contiguity
    comps = components(wrong, R, C)
    biggest = comps[0] / nw if comps else 0.0
    feat = {
        "nw": nw,
        "frac": nw / n,
        "border_frac": on_border / nw,
        "n_swap_kinds": n_swap_kinds,
        "under_frac": under / nw,
        "over_frac": over / nw,
        "n_comps": len(comps),
        "biggest_comp_frac": biggest,
    }
    # ---- dominant-pattern label (priority order: most specific first) ----
    if feat["border_frac"] >= 0.9:
        return "border", feat
    if n_swap_kinds <= 2 and feat["under_frac"] < 0.5 and feat["over_frac"] < 0.5:
        # a coherent colour remap that isn't just "left as input"
        return "colour-swap", feat
    if feat["under_frac"] >= 0.7:
        return "under-applied", feat
    if feat["over_frac"] >= 0.7:
        return "over-applied", feat
    if feat["n_comps"] <= 2 and feat["biggest_comp_frac"] >= 0.6:
        return "region", feat
    return "scattered", feat


def main(paths):
    recs = list(parse(paths))
    print(f"parsed {len(recs)} DIFF records")
    clusters = {}
    for rec in recs:
        label, feat = classify(rec)
        clusters.setdefault(label, []).append((rec["id"], feat))
    print("\ncluster sizes:")
    for label, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  {label:14s} {len(items):3d}")
    print("\nper-cluster detail (id: frac_wrong border under over comps big):")
    for label, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"\n[{label}] ({len(items)})")
        for tid, f in sorted(items, key=lambda t: -t[1].get("frac", 0))[:20]:
            if not f:
                print(f"  {tid}")
                continue
            print(f"  {tid}  frac {f['frac']:.3f}  border {f['border_frac']:.2f}"
                  f"  under {f['under_frac']:.2f}  over {f['over_frac']:.2f}"
                  f"  comps {f['n_comps']:2d}  big {f['biggest_comp_frac']:.2f}"
                  f"  swaps {f['n_swap_kinds']}")


if __name__ == "__main__":
    args = sys.argv[1:] or glob.glob("scratch/rungA/out_*.txt")
    main(args)
