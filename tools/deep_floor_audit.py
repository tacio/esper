"""Rung CMS audit (offline, measure-first): are the same-shape DEEP-FLOOR
tasks (held-out < 0.4 at v3 — can't even fit the demos, train-fit ~0.34)
CHAIN-OF-PROVEN-FACTORS shaped?

The CMS rung's mechanism is the twice-proven composition pattern chained in
DEPTH: written content/count factor -> geometry gather (ES) -> written local
override on the residual. Before building it, this tool tests — per task,
from the RAW demo pairs — whether such a chain would even express the rule,
by direct residual testing of progressively richer decompositions:

  depth-2 baseline : out[i] = g(pi(in)[i])           (geometry + colour map —
                     what GeomColor already expresses; a high score here is a
                     FIT failure, not an expressiveness failure)
  chain-local      : out[i] = f(pi(in)[i], ndiff8)   (+ LocalWrite-class
                     signature on the PREMAPPED grid — the depth-3 chain)
  chain-count      : out[i] = f(pi(in)[i], count_P)  (+ GridCountMap-class
                     count key — the other depth-3 chain)

Neighbourhood keys are TOROIDAL Moore-8 (the engine's and synth generator's
convention). Every score is LEAVE-ONE-DEMO-OUT: the keyed table is written
from n-1 demos and scored on the held-out demo, so a rich key only wins if it
GENERALIZES across demos (the Schug over-capacity hazard is controlled inside
the audit — in-sample consistency would let a 90-key table memorize).

Because a load-bearing chain factor may touch few cells (fill/outline classes:
~1-5%), a GLOBAL per-cell threshold cannot separate "depth-2 plus a small,
essential residual" from "pure depth-2 with LOO noise" (measured on synth:
0.989 vs 0.994). The discriminator is therefore the PAIRED RESIDUAL FIX at
the same geometry: of the cells the depth-2 map misses (LOO), the net
fraction the chain key fixes (fixes minus breaks). Calibrated on synth
ground truth (known depth-3 chains vs pure depth-2 controls, n_train=3).

    python tools/deep_floor_audit.py [train_dump]

Defaults to scratch/arc2_train_v3.txt. Reads grids from the raw ARC JSON
corpus (arg-agi-2-data/training/).

THE GATE (pre-registered, ROADMAP rung 5 / plan 2026-07-08): GO to CMS-1 iff
>= 25 of the ~146 deep-floor ids land in coherent chain-shaped clusters
(chain-local + chain-count at LOO >= 0.95 with a real margin over depth-2).
If the floor is dominated by object-level rules (per-object selection /
movement / counting-as-control-flow) that no chain of cellwise/positional
factors expresses, STOP is the documented outcome (Rung S precedent) and the
class re-scopes to the meta-trained factor (rung #6).
"""

import json
import os
import sys
from collections import Counter

CORPUS = os.path.join(os.path.dirname(__file__), "..", "arg-agi-2-data")

# Gate thresholds. GO_COUNT is pre-registered (plan 2026-07-08); the label
# thresholds are calibrated on synth ground truth (see module docstring and
# the calibration table in docs/JOURNAL.md).
NET_FIX_BAR = 0.5  # net fraction of the depth-2 residual the chain key fixes
RESIDUAL_FLOOR = 0.005  # depth-2 residual below this = depth-2 expressible
CHAIN_LOO_BAR = 0.9  # the chain key must also be globally consistent
GO_COUNT = 25


def deep_floor_ids(dump_path):
    """Same-shape (mem: same) train tasks with held-out < 0.4 at v3."""
    ids = []
    for line in open(dump_path):
        f = line.split()
        # `  task: <path>  held-out: H  train: T  gap: G  mem: same`
        if len(f) < 10 or f[0] != "task:":
            continue
        if f[8] == "mem:" and f[9] == "same" and float(f[3]) < 0.4:
            ids.append(os.path.basename(f[1]).replace(".task", ""))
    return ids


def load_demos(tid):
    with open(os.path.join(CORPUS, "training", f"{tid}.json")) as fh:
        t = json.load(fh)
    return [(d["input"], d["output"]) for d in t["train"]]


# ---- geometry candidates (the permutation class AttnGather expresses) ----


def apply_pi(grid, name):
    R, C = len(grid), len(grid[0])
    if name == "identity":
        return grid
    if name == "flip_h":
        return [list(reversed(row)) for row in grid]
    if name == "flip_v":
        return list(reversed(grid))
    if name == "rot180":
        return [list(reversed(row)) for row in reversed(grid)]
    if R != C:
        return None
    if name == "transpose":
        return [[grid[c][r] for c in range(C)] for r in range(R)]
    if name == "anti_transpose":
        return [[grid[C - 1 - c][R - 1 - r] for c in range(C)] for r in range(R)]
    if name == "rot90":
        return [[grid[R - 1 - c][r] for c in range(C)] for r in range(R)]
    if name == "rot270":
        return [[grid[c][C - 1 - r] for c in range(C)] for r in range(R)]
    return None


PI_NAMES = (
    "identity", "flip_h", "flip_v", "rot180",
    "transpose", "anti_transpose", "rot90", "rot270",
)


# ---- per-cell key features on the PREMAPPED grid q = pi(in) ----


def ndiff8(q, r, c):
    """# toroidal Moore-8 neighbours differing from the centre (the engine's
    and the synth generator's signature convention)."""
    R, C = len(q), len(q[0])
    n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if q[(r + dr) % R][(c + dc) % C] != q[r][c]:
                n += 1
    return n


def count_p(q, r, c, p):
    """# toroidal Moore-8 neighbours of colour p."""
    R, C = len(q), len(q[0])
    n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if q[(r + dr) % R][(c + dc) % C] == p:
                n += 1
    return n


def keyed_cells(demos, pi, keyfn):
    """Per demo: list of ((d2_key, chain_key), out_colour) for every cell —
    d2_key is the centre colour, chain_key the enriched key — or None if pi
    is inapplicable to any demo (non-square)."""
    per_demo = []
    for ig, og in demos:
        q = apply_pi(ig, pi)
        if q is None:
            return None
        R, C = len(q), len(q[0])
        cells = []
        for r in range(R):
            for c in range(C):
                cells.append(((q[r][c], keyfn(q, r, c)), og[r][c]))
        per_demo.append(cells)
    return per_demo


def loo_paired(per_demo):
    """Leave-one-demo-out, PAIRED at the same geometry: majority tables for
    both the depth-2 key (centre colour) and the chain key, built from the
    other demos, scored on the held-out demo. The chain predictor mirrors the
    actual mechanism (a gated OVERRIDE on the base map, not a replacement):
    it answers from the chain table when the key was seen, else FALLS BACK to
    the depth-2 table. Unseen on both = miss (honest — no identity fallback).
    Returns:

      d2_loo, chain_loo : per-cell LOO accuracy of each predictor
      net_fix           : (chain fixes of d2 misses - chain breaks of d2
                          hits) / d2 misses  — the load-bearing signal
      residual          : d2 miss fraction (how much the chain must explain)
    """
    n = len(per_demo)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0
    total = d2_hit = ch_hit = fixed = broken = 0
    for held in range(n):
        t2 = {}
        tc = {}
        for d in range(n):
            if d == held:
                continue
            for (k2, kc), out in per_demo[d]:
                t2.setdefault(k2, Counter())[out] += 1
                tc.setdefault(kc, Counter())[out] += 1
        for (k2, kc), out in per_demo[held]:
            total += 1
            h2 = k2 in t2 and t2[k2].most_common(1)[0][0] == out
            if kc in tc:
                hc = tc[kc].most_common(1)[0][0] == out
            else:
                hc = h2  # gated override absent -> base prediction
            d2_hit += h2
            ch_hit += hc
            if hc and not h2:
                fixed += 1
            elif h2 and not hc:
                broken += 1
    miss2 = total - d2_hit
    net_fix = (fixed - broken) / miss2 if miss2 else 0.0
    return (d2_hit / total, ch_hit / total, net_fix, miss2 / total)


# ---- structural fingerprints (context for the clusters) ----


def components(grid, bg):
    R, C = len(grid), len(grid[0])
    seen = [[False] * C for _ in range(R)]
    n = 0
    for r0 in range(R):
        for c0 in range(C):
            if grid[r0][c0] == bg or seen[r0][c0]:
                continue
            n += 1
            st = [(r0, c0)]
            seen[r0][c0] = True
            while st:
                r, c = st.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < R and 0 <= cc < C and not seen[rr][cc]
                            and grid[rr][cc] != bg):
                        seen[rr][cc] = True
                        st.append((rr, cc))
    return n


def fingerprint(demos):
    """Task-level structure hints: object-count delta, generated colours."""
    obj_deltas = []
    gen = 0
    for ig, og in demos:
        flat_i = [x for row in ig for x in row]
        flat_o = [x for row in og for x in row]
        bg = Counter(flat_i).most_common(1)[0][0]
        obj_deltas.append(components(og, bg) - components(ig, bg))
        if set(flat_o) - set(flat_i):
            gen += 1
    return {
        "obj_delta_mean": sum(obj_deltas) / len(obj_deltas),
        "new_colour_frac": gen / len(demos),
    }


# ---- per-task audit ----


def audit_task(demos):
    colours = set()
    for ig, _ in demos:
        for row in ig:
            colours.update(row)

    # Per key family, keep the pi with the best chain LOO; d2 is always
    # scored PAIRED at the same pi (plus a global best-d2 across pis).
    best = {
        "depth2": 0.0,
        "chain-local": (0.0, 0.0, 0.0, 1.0),  # (d2, chain, net_fix, residual)
        "chain-count": (0.0, 0.0, 0.0, 1.0),
    }
    best_pi = {}

    def key_local(q, r, c):
        return (q[r][c], ndiff8(q, r, c))

    for pi in PI_NAMES:
        cells = keyed_cells(demos, pi, key_local)
        if cells is None:
            continue
        d2, ch, fx, res = loo_paired(cells)
        if d2 > best["depth2"]:
            best["depth2"] = d2
            best_pi["depth2"] = pi
        if ch > best["chain-local"][1]:
            best["chain-local"] = (d2, ch, fx, res)
            best_pi["chain-local"] = pi
        for p in colours:
            stats = loo_paired(
                keyed_cells(
                    demos, pi,
                    lambda q, r, c, p=p: (q[r][c], count_p(q, r, c, p)),
                )
            )
            if stats[1] > best["chain-count"][1]:
                best["chain-count"] = stats
                best_pi["chain-count"] = pi

    # ---- label (priority: most specific first) ----
    d2 = best["depth2"]
    if 1.0 - d2 < RESIDUAL_FLOOR:
        return "depth2-fit-failure", best, best_pi

    def is_chain(stats):
        _, ch, fx, res = stats
        return fx >= NET_FIX_BAR and ch >= CHAIN_LOO_BAR and res >= RESIDUAL_FLOOR

    loc, cnt = best["chain-local"], best["chain-count"]
    if is_chain(loc) or is_chain(cnt):
        pick_local = loc[2] >= cnt[2] if (is_chain(loc) and is_chain(cnt)) \
            else is_chain(loc)
        label = "chain-local" if pick_local else "chain-count"
    elif max(loc[2], cnt[2]) >= 0.25:
        label = "chain-partial"
    else:
        label = "unexplained"
    return label, best, best_pi


def main(train_dump):
    ids = deep_floor_ids(train_dump)
    print(f"deep floor: {len(ids)} same-shape train tasks at held-out < 0.4")
    clusters = {}
    for tid in ids:
        demos = load_demos(tid)
        label, best, best_pi = audit_task(demos)
        fp = fingerprint(demos)
        # object-level annotation on the residual class
        if label == "unexplained" and abs(fp["obj_delta_mean"]) >= 3:
            label = "object-level"
        clusters.setdefault(label, []).append((tid, best, best_pi, fp))

    print("\ncluster sizes:")
    for label, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  {label:20s} {len(items):3d}")

    go = sum(len(v) for k, v in clusters.items()
             if k in ("chain-local", "chain-count"))
    print(f"\nGATE: chain-shaped = {go}/{len(ids)}"
          f"  (pre-registered GO >= {GO_COUNT})  ->"
          f"  {'GO' if go >= GO_COUNT else 'STOP'}")

    print("\nper-cluster detail (id: d2 | chain loo/net_fix/residual for "
          "local + count | pi obj_d newcol):")
    for label, items in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"\n[{label}] ({len(items)})")
        for tid, best, best_pi, fp in sorted(
                items,
                key=lambda t: -max(t[1]["chain-local"][2],
                                   t[1]["chain-count"][2]))[:25]:
            loc, cnt = best["chain-local"], best["chain-count"]
            pi = best_pi.get("chain-local", best_pi.get("depth2", "?"))
            print(f"  {tid}  d2 {best['depth2']:.3f}"
                  f"  loc {loc[1]:.3f}/{loc[2]:+.2f}/{loc[3]:.3f}"
                  f"  cnt {cnt[1]:.3f}/{cnt[2]:+.2f}/{cnt[3]:.3f}"
                  f"  pi {pi:14s}"
                  f"  obj_d {fp['obj_delta_mean']:+.1f}"
                  f"  newcol {fp['new_colour_frac']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch/arc2_train_v3.txt")
