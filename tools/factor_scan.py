"""Factor-coverage scan (post-CMS-0): WHICH missing read would explain the
deep floor?

CMS-0 (tools/deep_floor_audit.py) returned STOP: the 146 deep-floor tasks are
not chain-of-PROVEN-factors shaped — the missing capability is the factors,
not composition depth. This scan measures candidate NEW factor families
against the same 146 ids using the same calibrated harness (LOO paired
residual-fix on the premapped grid, gated-override fallback), so the next
rung's mechanism — and the literature pass aimed at it — is chosen on
coverage evidence, not intuition.

Each candidate family is a per-cell KEY over grid-global / object-level
context that no current memory can read. A task counts as COVERED by a
family when the family's keyed table (LOO) net-fixes >= 50% of the depth-2
residual with global consistency >= 0.9 — the same bar the audit's chain
labels used, so numbers are comparable. Emergence note: connected
components / mirrors / histograms here are SUBSTRATE (representations a
learned read could operate over), not hand-coded transforms — the scan asks
what a factor must EXPRESS, not how it is learned.

Extension (post-literature-pass, RESEARCH-NOTES 2026-07-08): the object-level
per-cell families covered 4/146 — the floor lies outside per-cell functions
over POSITION-ALIGNED context. The CONTENT_FAMILIES below test the next
class: per-cell keys whose fetch position is selected by CONTENT (rays,
nearest-object, global registers, content-defined anchors, object-local
frames) — the class a content-keyed AttnGather (position-query -> position +
content-match terms) would express. Pre-registered gate (user decision
2026-07-08): content-family union >= 20/146 => GO to building the Mojo
mechanism; below => documented STOP, floor re-scopes to the constructive
self-mod editor (rung #6).

    python tools/factor_scan.py [train_dump]

Reports per-family coverage, group unions (object-level vs content), a
greedy set cover, and per-task detail. ~5-10 min for the 146 ids.
"""

import sys
from collections import Counter, deque

from deep_floor_audit import (
    PI_NAMES,
    apply_pi,
    count_p,
    deep_floor_ids,
    load_demos,
    loo_paired,
    ndiff8,
)

# Same bars as the audit's chain label (comparability).
NET_FIX_BAR = 0.5
CHAIN_LOO_BAR = 0.9
RESIDUAL_FLOOR = 0.005
DIST_CAP = 5
SIZE_CAP = 30


# ---- per-grid precomputation (substrate: components, histogram, mirrors) --


def precompute(q):
    R, C = len(q), len(q[0])
    flat = [x for row in q for x in row]
    hist = Counter(flat)
    bg = hist.most_common(1)[0][0]
    # frequency rank of each colour (0 = most common)
    rank = {col: i for i, (col, _) in enumerate(hist.most_common())}

    # 4-connected same-colour components over ALL cells
    comp = [[-1] * C for _ in range(R)]
    sizes = []  # per component id
    bboxes = []  # (r0, r1, c0, c1)
    colours = []  # component colour
    cid = 0
    for r0 in range(R):
        for c0 in range(C):
            if comp[r0][c0] != -1:
                continue
            col = q[r0][c0]
            st = [(r0, c0)]
            comp[r0][c0] = cid
            cells = []
            while st:
                r, c = st.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < R and 0 <= cc < C and comp[rr][cc] == -1
                            and q[rr][cc] == col):
                        comp[rr][cc] = cid
                        st.append((rr, cc))
            sizes.append(len(cells))
            bboxes.append((
                min(r for r, _ in cells), max(r for r, _ in cells),
                min(c for _, c in cells), max(c for _, c in cells),
            ))
            colours.append(col)
            cid += 1

    # nonbg component size order -> largest/smallest flags per component
    nonbg_sizes = sorted(
        (sizes[i] for i in range(cid) if colours[i] != bg), reverse=True)
    big = nonbg_sizes[0] if nonbg_sizes else -1
    small = nonbg_sizes[-1] if nonbg_sizes else -1
    # size rank among nonbg components (distinct sizes, 0 = largest)
    distinct = sorted(set(nonbg_sizes), reverse=True)
    size_rank = {s: i for i, s in enumerate(distinct)}

    # multi-source BFS distance to the OTHER set (bg <-> nonbg), 8-connected
    dist = [[DIST_CAP] * C for _ in range(R)]
    dq = deque()
    for r in range(R):
        for c in range(C):
            other = any(
                0 <= r + dr < R and 0 <= c + dc < C
                and (q[r + dr][c + dc] == bg) != (q[r][c] == bg)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            )
            if other:
                dist[r][c] = 1
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        if dist[r][c] >= DIST_CAP:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < R and 0 <= cc < C and dist[rr][cc] > dist[r][c] + 1:
                    dist[rr][cc] = dist[r][c] + 1
                    dq.append((rr, cc))

    # ---- content-addressed substrate (fetch position selected by content) --

    # nearest-nonbg colour + distance per cell (multi-source BFS, 8-connected)
    near_col = [[None] * C for _ in range(R)]
    near_d = [[DIST_CAP] * C for _ in range(R)]
    dq = deque()
    for r in range(R):
        for c in range(C):
            if q[r][c] != bg:
                near_col[r][c] = q[r][c]
                near_d[r][c] = 0
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        if near_d[r][c] >= DIST_CAP:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (0 <= rr < R and 0 <= cc < C
                        and near_d[rr][cc] > near_d[r][c] + 1):
                    near_d[rr][cc] = near_d[r][c] + 1
                    near_col[rr][cc] = near_col[r][c]
                    dq.append((rr, cc))

    # first nonbg colour strictly along each ray (up/down/left/right)
    ray_u = [[None] * C for _ in range(R)]
    ray_d = [[None] * C for _ in range(R)]
    ray_l = [[None] * C for _ in range(R)]
    ray_r = [[None] * C for _ in range(R)]
    for c in range(C):
        last = None
        for r in range(R):
            ray_u[r][c] = last
            if q[r][c] != bg:
                last = q[r][c]
        last = None
        for r in range(R - 1, -1, -1):
            ray_d[r][c] = last
            if q[r][c] != bg:
                last = q[r][c]
    for r in range(R):
        last = None
        for c in range(C):
            ray_l[r][c] = last
            if q[r][c] != bg:
                last = q[r][c]
        last = None
        for c in range(C - 1, -1, -1):
            ray_r[r][c] = last
            if q[r][c] != bg:
                last = q[r][c]

    # global content registers + anchors (bbox top-left of the register comp)
    nonbg_ids = [i for i in range(cid) if colours[i] != bg]
    large_i = max(nonbg_ids, key=lambda i: sizes[i], default=None)
    small_i = min(nonbg_ids, key=lambda i: sizes[i], default=None)
    col_comp_n = Counter(colours[i] for i in nonbg_ids)
    uniq_cols = [col for col, n in col_comp_n.items() if n == 1]
    uniq_col = uniq_cols[0] if len(uniq_cols) == 1 else None
    uniq_i = next((i for i in nonbg_ids if colours[i] == uniq_col),
                  None) if uniq_col is not None else None
    large_col = colours[large_i] if large_i is not None else None
    small_col = colours[small_i] if small_i is not None else None
    nonbg_hist = Counter(x for x in flat if x != bg)
    major_col = nonbg_hist.most_common(1)[0][0] if nonbg_hist else None
    anchors = {}
    for name, i in (("large", large_i), ("uniq", uniq_i)):
        anchors[name] = (bboxes[i][0], bboxes[i][2]) if i is not None else None

    return {
        "q": q, "R": R, "C": C, "bg": bg, "rank": rank,
        "comp": comp, "sizes": sizes, "bboxes": bboxes, "colours": colours,
        "big": big, "small": small, "size_rank": size_rank, "dist": dist,
        "near_col": near_col, "near_d": near_d,
        "ray": {"u": ray_u, "d": ray_d, "l": ray_l, "r": ray_r},
        "large_col": large_col, "small_col": small_col, "uniq_col": uniq_col,
        "major_col": major_col, "anchors": anchors,
    }


# ---- candidate factor families (key = f(cell, substrate)) ----


def k_comp_size(P, r, c):
    i = P["comp"][r][c]
    col = P["q"][r][c]
    if col == P["bg"]:
        return (col, -1)
    return (col, min(P["sizes"][i], SIZE_CAP))


def k_comp_rank(P, r, c):
    i = P["comp"][r][c]
    col = P["q"][r][c]
    if col == P["bg"]:
        return (col, -1, False, False)
    s = P["sizes"][i]
    return (col, min(P["size_rank"].get(s, 9), 9), s == P["big"],
            s == P["small"])


def k_bbox_pos(P, r, c):
    i = P["comp"][r][c]
    col = P["q"][r][c]
    if col == P["bg"]:
        return (col, "bg")
    r0, r1, c0, c1 = P["bboxes"][i]
    on_r = r in (r0, r1)
    on_c = c in (c0, c1)
    pos = "corner" if (on_r and on_c) else "edge" if (on_r or on_c) else "in"
    return (col, pos)


def k_sym_h(P, r, c):
    return (P["q"][r][c], P["q"][r][P["C"] - 1 - c])


def k_sym_v(P, r, c):
    return (P["q"][r][c], P["q"][P["R"] - 1 - r][c])


def k_sym_180(P, r, c):
    return (P["q"][r][c], P["q"][P["R"] - 1 - r][P["C"] - 1 - c])


def k_freq_rank(P, r, c):
    col = P["q"][r][c]
    return (col, min(P["rank"][col], 5))


def k_dist(P, r, c):
    return (P["q"][r][c], P["q"][r][c] == P["bg"], P["dist"][r][c])


def k_comp_dims(P, r, c):
    i = P["comp"][r][c]
    col = P["q"][r][c]
    if col == P["bg"]:
        return (col, -1, -1)
    r0, r1, c0, c1 = P["bboxes"][i]
    return (col, min(r1 - r0 + 1, 15), min(c1 - c0 + 1, 15))


FAMILIES = {
    "comp-size": k_comp_size,
    "comp-rank": k_comp_rank,
    "bbox-pos": k_bbox_pos,
    "comp-dims": k_comp_dims,
    "sym-h": k_sym_h,
    "sym-v": k_sym_v,
    "sym-180": k_sym_180,
    "freq-rank": k_freq_rank,
    "dist": k_dist,
}


# ---- content-addressed families: the fetch position is selected by CONTENT
# (the class a content-keyed AttnGather would express: "read from the cell
# nearest/relative to a content-defined anchor having property P") ----


def k_ray4(P, r, c):
    ray = P["ray"]
    return (P["q"][r][c], ray["u"][r][c], ray["d"][r][c],
            ray["l"][r][c], ray["r"][r][c])


def k_nearest(P, r, c):
    return (P["q"][r][c], P["q"][r][c] == P["bg"],
            P["near_col"][r][c], P["near_d"][r][c])


def k_registers(P, r, c):
    return (P["q"][r][c], P["large_col"], P["small_col"], P["uniq_col"])


def k_objlocal(P, r, c):
    i = P["comp"][r][c]
    r0, r1, c0, c1 = P["bboxes"][i]
    return (P["q"][r][c], P["q"][r][c0 + c1 - c], P["q"][r0 + r1 - r][c])


CONTENT_FAMILIES = {
    "fetch-ray4": k_ray4,
    "fetch-nearest": k_nearest,
    "fetch-registers": k_registers,
    "fetch-objlocal": k_objlocal,
}
# fetch-anchor: colour at the position displaced by a content-defined anchor
# (largest / unique-colour component bbox corner), both signs — variants kept
# best-of under one name, like count-P.
ANCHOR_VARIANTS = tuple((a, s) for a in ("large", "uniq") for s in (1, -1))


# ---- copy-capable families (the sharp content-keyed-gather semantics) ----
#
# A keyed TABLE can only emit colours it has seen for a key, so copy-through
# rules (out = the fetched cell's value, colours varying across demos) are
# invisible to it. A sharp content-keyed gather emits the ATTENDED CELL'S
# VALUE. These families pair a colour-ABSTRACT relational key with a fetched
# value; the table's votes are the abstract actions KEEP (out == centre) /
# COPY (out == fetched) plus constant colours (loo_paired_fetch below).
# Each returns (key, fetched); variants are separate runs, best-of per name.


def fetch_ray(P, r, c, d):
    f = P["ray"][d][r][c]
    return ((P["q"][r][c] == P["bg"], f is not None), f)


def fetch_nearest(P, r, c):
    return ((P["q"][r][c] == P["bg"], P["near_d"][r][c]),
            P["near_col"][r][c])


def fetch_register(P, r, c, reg):
    f = P[reg]
    return ((P["q"][r][c] == P["bg"], P["q"][r][c] == f), f)


def fetch_anchor(P, r, c, a, s):
    anc = P["anchors"][a]
    if anc is None:
        return ((P["q"][r][c] == P["bg"], None), None)
    f = P["q"][(r + s * anc[0]) % P["R"]][(c + s * anc[1]) % P["C"]]
    return ((P["q"][r][c] == P["bg"], f == P["q"][r][c]), f)


def fetch_objlocal(P, r, c, axis):
    r0, r1, c0, c1 = P["bboxes"][P["comp"][r][c]]
    f = P["q"][r][c0 + c1 - c] if axis == "h" else P["q"][r0 + r1 - r][c]
    return ((P["q"][r][c] == P["bg"], f == P["q"][r][c]), f)


COPY_FAMILIES = {
    "copy-ray": [lambda P, r, c, d=d: fetch_ray(P, r, c, d)
                 for d in ("u", "d", "l", "r")],
    "copy-nearest": [fetch_nearest],
    "copy-registers": [
        lambda P, r, c, g=g: fetch_register(P, r, c, g)
        for g in ("large_col", "small_col", "uniq_col", "major_col")],
    "copy-anchor": [lambda P, r, c, a=a, s=s: fetch_anchor(P, r, c, a, s)
                    for a, s in ANCHOR_VARIANTS],
    "copy-objlocal": [lambda P, r, c, x=x: fetch_objlocal(P, r, c, x)
                      for x in ("h", "v")],
}

KEEP = "KEEP"
COPY = "COPY"


def loo_paired_fetch(per_demo):
    """LOO paired exactly like deep_floor_audit.loo_paired (same gated
    override + d2 fallback, same return tuple), but cells are
    ((d2_key, chain_key, fetched), out) and the chain table votes over
    abstract actions: KEEP -> centre (= the d2 key), COPY -> fetched value,
    else a constant colour."""
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
            for (k2, kc, f), out in per_demo[d]:
                t2.setdefault(k2, Counter())[out] += 1
                if out == k2:
                    v = KEEP
                elif f is not None and out == f:
                    v = COPY
                else:
                    v = out
                tc.setdefault(kc, Counter())[v] += 1
        for (k2, kc, f), out in per_demo[held]:
            total += 1
            h2 = k2 in t2 and t2[k2].most_common(1)[0][0] == out
            if kc in tc:
                v = tc[kc].most_common(1)[0][0]
                pred = k2 if v == KEEP else f if v == COPY else v
                hc = h2 if pred is None else pred == out
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


# the audit's proven-factor keys, re-scored here for a comparable baseline row
BASELINES = ("ndiff8", "count-P")


def scan_task(demos):
    """Best (chain_loo, net_fix, residual, pi) per family across geometries."""
    colours = set()
    for ig, _ in demos:
        for row in ig:
            colours.update(row)

    best = {}  # family -> (d2, chain, fix, residual, pi)

    def consider(fam, stats, pi):
        if fam not in best or stats[1] > best[fam][1]:
            best[fam] = (*stats, pi)

    for pi in PI_NAMES:
        qs = [apply_pi(ig, pi) for ig, _ in demos]
        if any(q is None for q in qs):
            continue
        pres = [precompute(q) for q in qs]
        # families with precomputed substrate
        for fam, keyfn in {**FAMILIES, **CONTENT_FAMILIES}.items():
            per_demo = []
            for P, (_, og) in zip(pres, demos):
                cells = []
                for r in range(P["R"]):
                    for c in range(P["C"]):
                        cells.append(
                            ((P["q"][r][c], keyfn(P, r, c)), og[r][c]))
                per_demo.append(cells)
            consider(fam, loo_paired(per_demo), pi)
        # fetch-anchor variants (anchor x sign), best-of under one name
        for a, s in ANCHOR_VARIANTS:
            per_demo = []
            for P, (_, og) in zip(pres, demos):
                anc = P["anchors"][a]
                q, R, C = P["q"], P["R"], P["C"]
                cells = []
                for r in range(R):
                    for c in range(C):
                        if anc is None:
                            f = None
                        else:
                            f = q[(r + s * anc[0]) % R][(c + s * anc[1]) % C]
                        cells.append(((q[r][c], (q[r][c], a, s, f)),
                                      og[r][c]))
                per_demo.append(cells)
            consider("fetch-anchor", loo_paired(per_demo), pi)
        # copy-capable families (KEEP/COPY votes), variants best-of per name
        for fam, variants in COPY_FAMILIES.items():
            for fn in variants:
                per_demo = []
                for P, (_, og) in zip(pres, demos):
                    cells = []
                    for r in range(P["R"]):
                        for c in range(P["C"]):
                            key, f = fn(P, r, c)
                            cells.append(
                                ((P["q"][r][c], key, f), og[r][c]))
                    per_demo.append(cells)
                consider(fam, loo_paired_fetch(per_demo), pi)
        # proven-factor baselines (same numbers as the audit)
        per_demo = []
        for q, (_, og) in zip(qs, demos):
            R, C = len(q), len(q[0])
            per_demo.append([
                ((q[r][c], (q[r][c], ndiff8(q, r, c))), og[r][c])
                for r in range(R) for c in range(C)
            ])
        consider("ndiff8", loo_paired(per_demo), pi)
        for p in colours:
            per_demo = []
            for q, (_, og) in zip(qs, demos):
                R, C = len(q), len(q[0])
                per_demo.append([
                    ((q[r][c], (q[r][c], count_p(q, r, c, p))), og[r][c])
                    for r in range(R) for c in range(C)
                ])
            consider("count-P", loo_paired(per_demo), pi)
    return best


def covered(stats):
    d2, ch, fx, res, _pi = stats
    return fx >= NET_FIX_BAR and ch >= CHAIN_LOO_BAR and res >= RESIDUAL_FLOOR


def main(train_dump):
    ids = deep_floor_ids(train_dump)
    print(f"factor scan over {len(ids)} deep-floor ids; "
          f"cover = net_fix >= {NET_FIX_BAR} & loo >= {CHAIN_LOO_BAR}")
    content_names = (list(CONTENT_FAMILIES) + ["fetch-anchor"]
                     + list(COPY_FAMILIES))
    all_new = list(FAMILIES) + content_names
    cover_sets = {f: set() for f in all_new + list(BASELINES)}
    per_task = {}
    for n, tid in enumerate(ids):
        best = scan_task(load_demos(tid))
        per_task[tid] = best
        for fam, stats in best.items():
            if covered(stats):
                cover_sets[fam].add(tid)
        if (n + 1) % 25 == 0:
            print(f"  ... {n + 1}/{len(ids)}")

    print("\nper-family coverage (tasks where the read is load-bearing "
          "and consistent):")
    for fam, s in sorted(cover_sets.items(), key=lambda kv: -len(kv[1])):
        tag = " (proven baseline)" if fam in BASELINES else ""
        print(f"  {fam:12s} {len(s):3d}{tag}")

    new_sets = {f: s for f, s in cover_sets.items() if f not in BASELINES}
    union = set().union(*new_sets.values())
    base_union = set().union(*(cover_sets[f] for f in BASELINES))
    obj_union = set().union(*(cover_sets[f] for f in FAMILIES))
    content_union = set().union(*(cover_sets[f] for f in content_names))
    print(f"\nunion of NEW families: {len(union)}/{len(ids)}"
          f"   (proven baselines: {len(base_union)};"
          f" new-only: {len(union - base_union)})")
    print(f"  object-level per-cell group: {len(obj_union)}"
          f"   CONTENT-ADDRESSED group: {len(content_union)}"
          f"   (gate: content union >= 20)")

    print("\ngreedy set cover (new families):")
    remaining = set(union)
    while remaining:
        fam = max(new_sets, key=lambda f: len(new_sets[f] & remaining))
        got = new_sets[fam] & remaining
        if not got:
            break
        print(f"  + {fam:12s} covers {len(got):3d}  (cum "
              f"{len(union) - len(remaining - got)})")
        remaining -= got

    print("\nper-task best NEW family (covered tasks):")
    for tid in sorted(union):
        fam, stats = max(
            ((f, per_task[tid][f]) for f in new_sets if tid in new_sets[f]),
            key=lambda kv: kv[1][2],
        )
        d2, ch, fx, res, pi = stats
        print(f"  {tid}  {fam:12s} d2 {d2:.3f}  loo {ch:.3f}"
              f"  fix {fx:+.2f}  res {res:.3f}  pi {pi}")

    print("\nnear misses (best new family fix in 0.25-0.5, not covered):")
    for tid in sorted(set(ids) - union):
        fam, stats = max(
            ((f, per_task[tid][f]) for f in new_sets),
            key=lambda kv: kv[1][2],
        )
        if stats[2] >= 0.25:
            print(f"  {tid}  {fam:12s} d2 {stats[0]:.3f}  loo {stats[1]:.3f}"
                  f"  fix {stats[2]:+.2f}  res {stats[3]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scratch/arc2_train_v3.txt")
