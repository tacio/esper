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

Extension (post-CF, partial-fix-band build 2026-07-08): CF's hard content
table converts only 2 of the 22 to exact and leaves a ~72-id partial-fix band
(~0.9 held-out). The SOFTSCORE_FAMILIES below test the SOFT content-keyed
gather's incremental class over the fixed copy-* reads — a centre-RELATIVE
relation (nearest cell whose component is strictly larger/smaller than the
centre's; the Slot-Abstractors relational bottleneck), faithful to the Mojo
gather's `argmax_j[-beta|q_i-x_j|^2 + w.feat]` read (nearest-of-relation, emit
value). PRE-REGISTERED gate (plan, before the run): soft-gather class newly
COVERS >= 15 deep-floor ids beyond the hard content union => GO to building
`ContentGatherComposedMemory`; below => documented STOP for the soft gather
(Phase 2 rung #6 proceeds independently). RESULT 2026-07-09: soft incremental =
3 (25094a63 52364a65 7d1f7ee8) < 15 => STOP; the soft gather does not open new
band territory beyond the sharp CF table.

Extension (rung #6 constructive editor, 2026-07-09): the EDITOR_FAMILIES +
scan_editor below simulate the TRM-style iterated-edit loop (materialized
answer grid, one colour-abstract local relational rule read over the EVOLVING
grid, up to EDITOR_T passes, writes become evidence — positions-written !=
positions-read, the content-addressed construction no single per-cell pass
expresses). PRE-REGISTERED gate (plan, before the run): the iterated editor
newly COVERS >= 15 deep-floor ids that the ENTIRE per-cell/content new-family
union does NOT reach at the 0.9 bar => GO to building `GridEditorSelfModMemory`;
below => documented STOP.

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


def _bfs_nearest_col(q, R, C, sources):
    """8-connected multi-source BFS (capped at DIST_CAP): nearest SOURCE
    cell's colour per cell (None if unreached). Sources carry their own
    colour, so the frontier propagates the nearest source colour — the same
    shape as near_col, restricted to an arbitrary source subset."""
    col = [[None] * C for _ in range(R)]
    d = [[DIST_CAP + 1] * C for _ in range(R)]
    dq = deque()
    for (r, c) in sources:
        col[r][c] = q[r][c]
        d[r][c] = 0
        dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        if d[r][c] >= DIST_CAP:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < R and 0 <= cc < C and d[rr][cc] > d[r][c] + 1:
                    d[rr][cc] = d[r][c] + 1
                    col[rr][cc] = col[r][c]
                    dq.append((rr, cc))
    return col


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

    # centre-RELATIVE relational reads (the soft content-keyed gather's
    # incremental class over the fixed copy-* families: an additive content
    # score biases the position-nearest gather toward a source satisfying a
    # relation to the CENTRE — nearest cell whose component is strictly
    # larger / smaller than the centre's, emit its colour. Faithful to the
    # Mojo `argmax_j[-beta|q_i-x_j|^2 + w.feat]` read: nearest-of-relation,
    # value = that cell's colour; NON-collapsing and distinct from the GLOBAL
    # largest/smallest registers copy-registers already expresses).
    csize = [[sizes[comp[r][c]] for c in range(C)] for r in range(R)]
    larger_near = [[None] * C for _ in range(R)]
    smaller_near = [[None] * C for _ in range(R)]
    distinct_sz = sorted({csize[r][c] for r in range(R) for c in range(C)
                          if q[r][c] != bg})
    for t in distinct_sz:
        bigger = _bfs_nearest_col(
            q, R, C, [(r, c) for r in range(R) for c in range(C)
                      if q[r][c] != bg and csize[r][c] > t])
        smaller = _bfs_nearest_col(
            q, R, C, [(r, c) for r in range(R) for c in range(C)
                      if q[r][c] != bg and csize[r][c] < t])
        for r in range(R):
            for c in range(C):
                if q[r][c] != bg and csize[r][c] == t:
                    larger_near[r][c] = bigger[r][c]
                    smaller_near[r][c] = smaller[r][c]

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
        "larger_near": larger_near, "smaller_near": smaller_near,
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

# ---- soft content-keyed gather families (the ES-fit score's incremental
# class over the fixed copy-* reads) --------------------------------------
#
# The soft gather selects its source by `argmax_j[-beta|q_i-x_j|^2 + w.feat]`:
# a position-nearest read biased by a fitted content-match term, emitting the
# won cell's VALUE (sharp beta => copy-through). Its incremental power over
# copy-* is a CENTRE-RELATIVE relation: the source is the nearest cell whose
# component is strictly larger / smaller than the centre's (Slot-Abstractors'
# larger-than/smaller-than relations) — distinct from copy-registers' GLOBAL
# largest/smallest, and non-collapsing (the fetched colour genuinely varies).
# Same (key, fetched) shape + loo_paired_fetch scoring as COPY_FAMILIES so the
# coverage numbers are directly comparable; each returns (chain_key, fetched).


def fetch_larger_near(P, r, c):
    f = P["larger_near"][r][c]
    return ((P["q"][r][c] == P["bg"], f is not None, f == P["q"][r][c]), f)


def fetch_smaller_near(P, r, c):
    f = P["smaller_near"][r][c]
    return ((P["q"][r][c] == P["bg"], f is not None, f == P["q"][r][c]), f)


SOFTSCORE_FAMILIES = {
    "soft-larger": [fetch_larger_near],
    "soft-smaller": [fetch_smaller_near],
}

KEEP = "KEEP"
COPY = "COPY"


# ---- CF-read probe: deterministic tie-break + finer keys ------------------
#
# The band evidence (scratch/content_scan_v1.txt): the ~72 partial-fix ids are
# dominated by copy-* families at LOO 0.70-0.90 — the right SOURCE is grazed
# but the closed-form voted table isn't consistent enough to cross the 0.9
# cover bar. Two deterministic sharpenings, measured here in Python BEFORE any
# Mojo is touched (the pre-rung-#6 probe, ROADMAP rung #6):
#
#   (a) TIE-BREAK. loo_paired_fetch uses Counter.most_common(1), whose tie
#       resolution is insertion-order (nondeterministic w.r.t. demo order).
#       loo_paired_fetch_det resolves ties by an explicit action precedence,
#       so a near-tied band bucket becomes a stable, portable decision.
#   (b) KEY GRANULARITY. The copy-* chain keys are coarse (e.g. copy-ray keys
#       only on `f is not None`), so one bucket mixes cells whose true action
#       is KEEP with cells whose action is COPY -> impure majority. The finer
#       variants add the KEEP/COPY-disambiguating `f == centre` bit (and, where
#       it fits, register identity / distance band), splitting the impure
#       bucket. Each finer key's non-is_bg part stays < SUB_REL_K (8) buckets so
#       it ports to Mojo's `fetch` rel bucket without widening the 16-entry
#       table (CONTENTFETCH_KEYS = 2 * SUB_REL_K).
#
# Both are scored on the EXACT copy-* fetch sources (same reads), so any gain is
# attributable to sharper selection, not a new capability. Honesty guard:
# scratch/calib_cfprobe.py (positives covered, support-starvation negatives 0).

# action precedence for deterministic tie resolution (higher wins a vote tie).
# COPY/KEEP (relational, generalizing) beat a memorized constant; among
# constants the lower colour index wins (stable, mirrors a lowest-index rule).
PREC = {"COPY": (3, 0), "KEEP": (2, 0)}


def _det_argmax(counter):
    """Deterministic most-common: max count, then action precedence, then a
    stable key. Constants (int colours) rank below KEEP/COPY; lower colour
    wins among constants."""
    return max(
        counter.items(),
        key=lambda kv: (kv[1], PREC.get(kv[0], (1, -kv[0] if isinstance(
            kv[0], int) else 0))),
    )[0]


def loo_paired_fetch_det(per_demo):
    """loo_paired_fetch with DETERMINISTIC tie resolution (via _det_argmax) at
    both the base (t2) and abstract-action (tc) decision sites. Identical
    otherwise — same tuple shape, same net_fix accounting."""
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
            h2 = k2 in t2 and _det_argmax(t2[k2]) == out
            if kc in tc:
                v = _det_argmax(tc[kc])
                pred = k2 if v == KEEP else f if v == COPY else v
                hc = h2 if pred is None else pred == out
            else:
                hc = h2
            d2_hit += h2
            ch_hit += hc
            if hc and not h2:
                fixed += 1
            elif h2 and not hc:
                broken += 1
    miss2 = total - d2_hit
    net_fix = (fixed - broken) / miss2 if miss2 else 0.0
    return (d2_hit / total, ch_hit / total, net_fix, miss2 / total)


# Finer-key copy fetches: same SOURCE as the committed copy-* fetch, but the
# chain key adds the KEEP/COPY-disambiguating `f == centre` bit (+ small,
# portable extras). Each returns (finer_key, fetched).
def fetch_ray_fine(P, r, c, d):
    f = P["ray"][d][r][c]
    ctr = P["q"][r][c]
    # non-is_bg part in {0,1,2}: absent / present!=centre / present==centre
    rel = 0 if f is None else (2 if f == ctr else 1)
    return ((ctr == P["bg"], rel), f)


def fetch_nearest_fine(P, r, c):
    f = P["near_col"][r][c]
    ctr = P["q"][r][c]
    band = min(P["near_d"][r][c], 2)  # {0,1,>=2} -> banded distance (3)
    same = 1 if f == ctr else 0
    # non-is_bg part in {0..5} = band*2 + same  (< SUB_REL_K)
    return ((ctr == P["bg"], band * 2 + same), f)


def fetch_register_fine(P, r, c, reg):
    f = P[reg]
    ctr = P["q"][r][c]
    # non-is_bg part in {0,1,2,3}: (f==centre) and (f==bg) both encoded
    rel = (1 if f == ctr else 0) + (2 if f == P["bg"] else 0)
    return ((ctr == P["bg"], rel), f)


def fetch_anchor_fine(P, r, c, a, s):
    anc = P["anchors"][a]
    ctr = P["q"][r][c]
    if anc is None:
        return ((ctr == P["bg"], 0), None)
    f = P["q"][(r + s * anc[0]) % P["R"]][(c + s * anc[1]) % P["C"]]
    rel = 2 if f == ctr else 1  # present==centre / present!=centre
    return ((ctr == P["bg"], rel), f)


def fetch_objlocal_fine(P, r, c, axis):
    r0, r1, c0, c1 = P["bboxes"][P["comp"][r][c]]
    ctr = P["q"][r][c]
    f = P["q"][r][c0 + c1 - c] if axis == "h" else P["q"][r0 + r1 - r][c]
    rel = 2 if f == ctr else 1
    return ((ctr == P["bg"], rel), f)


PROBE_COPY_FAMILIES = {
    "copy-ray": [lambda P, r, c, d=d: fetch_ray_fine(P, r, c, d)
                 for d in ("u", "d", "l", "r")],
    "copy-nearest": [fetch_nearest_fine],
    "copy-registers": [
        lambda P, r, c, g=g: fetch_register_fine(P, r, c, g)
        for g in ("large_col", "small_col", "uniq_col", "major_col")],
    "copy-anchor": [lambda P, r, c, a=a, s=s: fetch_anchor_fine(P, r, c, a, s)
                    for a, s in ANCHOR_VARIANTS],
    "copy-objlocal": [lambda P, r, c, x=x: fetch_objlocal_fine(P, r, c, x)
                      for x in ("h", "v")],
}


def _score_copy(pres, demos, fn):
    per_demo = []
    for P, (_, og) in zip(pres, demos):
        cells = []
        for r in range(P["R"]):
            for c in range(P["C"]):
                key, f = fn(P, r, c)
                cells.append(((P["q"][r][c], key, f), og[r][c]))
        per_demo.append(cells)
    return per_demo


def scan_task_probe(demos):
    """Two probe groups, each best-(chain_loo,net_fix,res,pi) per family across
    geometries — same shape as scan_task so `covered` applies unchanged:
      tb   = committed copy-* sources, DETERMINISTIC voting (isolates how much
             of the band is pure tie nondeterminism).
      fine = finer keys (KEEP/COPY-disambiguating bit) + deterministic voting
             (the full sharpening)."""
    tb = {}
    fine = {}

    def consider(store, fam, stats, pi):
        if fam not in store or stats[1] > store[fam][1]:
            store[fam] = (*stats, pi)

    for pi in PI_NAMES:
        qs = [apply_pi(ig, pi) for ig, _ in demos]
        if any(q is None for q in qs):
            continue
        pres = [precompute(q) for q in qs]
        for fam, variants in COPY_FAMILIES.items():  # committed coarse sources
            for fn in variants:
                consider(tb, fam,
                         loo_paired_fetch_det(_score_copy(pres, demos, fn)), pi)
        for fam, variants in PROBE_COPY_FAMILIES.items():  # finer keys
            for fn in variants:
                consider(fine, fam,
                         loo_paired_fetch_det(_score_copy(pres, demos, fn)), pi)
    return tb, fine


# ---- rung #6 constructive editor: iterated-edit simulation ----------------
#
# The per-cell scan (single pass over the INPUT substrate) cannot express an
# iterated editor. scan_editor simulates the TRM-style loop faithfully:
# maintain a materialized answer grid y (init = input); for up to T passes
# apply ONE colour-abstract local relational rule reading the CURRENT y, and
# write where it fires. Because y evolves, a write becomes evidence for the
# next pass (propagate / flood / extend) — the content-addressed construction
# a single per-cell pass cannot reach (positions written != positions read).
#
# The rule is a LOO-fit abstract action table (KEEP/COPY only — pure-constant
# votes are DROPPED so palette memorization can't fake coverage; the editor
# earns coverage only through RELATIONAL propagation). Substrate is the cheap
# _light grid (bg + 4 rays, O(RC)) so T passes stay tractable.

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
EDITOR_T = 16


def _light(q):
    """Cheap per-pass substrate for the editor: bg + first-nonbg rays. No
    components / BFS / larger_near (the editor's fetch fns need only q's
    neighbours and directional scans over the EVOLVING grid)."""
    R, C = len(q), len(q[0])
    bg = Counter(x for row in q for x in row).most_common(1)[0][0]
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
    return {"q": q, "R": R, "C": C, "bg": bg,
            "ray": {"u": ray_u, "d": ray_d, "l": ray_l, "r": ray_r}}


def ed_flood(P, r, c):
    """Majority non-bg 4-neighbour colour in the current grid (region
    thicken / flood front). key = (centre is bg, has a non-bg neighbour)."""
    q, R, C, bg = P["q"], P["R"], P["C"], P["bg"]
    nb = [q[r + dr][c + dc] for dr, dc in DIRS4
          if 0 <= r + dr < R and 0 <= c + dc < C and q[r + dr][c + dc] != bg]
    f = Counter(nb).most_common(1)[0][0] if nb else None
    return ((q[r][c] == bg, f is not None), f)


def _ed_dir(dr, dc):
    def fn(P, r, c):
        q, R, C, bg = P["q"], P["R"], P["C"], P["bg"]
        rr, cc = r + dr, c + dc
        f = q[rr][cc] if 0 <= rr < R and 0 <= cc < C else None
        if f == bg:
            f = None
        return ((q[r][c] == bg, f is not None), f)
    return fn


def _ed_ray(d):
    def fn(P, r, c):
        f = P["ray"][d][r][c]
        return ((P["q"][r][c] == P["bg"], f is not None), f)
    return fn


EDITOR_FAMILIES = {
    "ed-flood": [ed_flood],
    "ed-dir": [_ed_dir(dr, dc) for dr, dc in DIRS4],
    "ed-ray": [_ed_ray(d) for d in ("u", "d", "l", "r")],
}


def loo_editor(demos, keyfn, T=EDITOR_T):
    """LOO: fit the abstract KEEP/COPY table from the other demos (input
    substrate -> output value, constant votes dropped), then ITERATE the held
    demo's grid from input for up to T passes over the evolving substrate;
    stop at a fixed point. Returns (d2, final_loo, net_fix, residual) vs the
    identity baseline — same tuple shape / bars as loo_paired_fetch."""
    n = len(demos)
    if n < 2:
        return 0.0, 0.0, 0.0, 1.0
    total = d2_hit = final_hit = fixed = broken = 0
    for held in range(n):
        tc = {}
        for d in range(n):
            if d == held:
                continue
            ig, og = demos[d]
            P = _light(ig)
            for r in range(P["R"]):
                for c in range(P["C"]):
                    key, f = keyfn(P, r, c)
                    out, cen = og[r][c], ig[r][c]
                    if out == cen:
                        v = KEEP
                    elif f is not None and out == f:
                        v = COPY
                    else:
                        continue  # abstract-only: drop constant/undetermined
                    tc.setdefault(key, Counter())[v] += 1
        ig, og = demos[held]
        R, C = len(ig), len(ig[0])
        y = [row[:] for row in ig]
        for _ in range(T):
            P = _light(y)
            ny = [row[:] for row in y]
            changed = False
            for r in range(R):
                for c in range(C):
                    key, f = keyfn(P, r, c)
                    if key in tc and tc[key].most_common(1)[0][0] == COPY \
                            and f is not None and ny[r][c] != f:
                        ny[r][c] = f
                        changed = True
            y = ny
            if not changed:
                break
        for r in range(R):
            for c in range(C):
                total += 1
                h2 = ig[r][c] == og[r][c]
                hf = y[r][c] == og[r][c]
                d2_hit += h2
                final_hit += hf
                if hf and not h2:
                    fixed += 1
                elif h2 and not hf:
                    broken += 1
    miss2 = total - d2_hit
    net_fix = (fixed - broken) / miss2 if miss2 else 0.0
    return (d2_hit / total, final_hit / total, net_fix, miss2 / total)


def scan_editor(demos):
    """Best (d2, final_loo, net_fix, residual) per editor family (best-of its
    direction variants), identity premap only (conservative: the band is
    identity-dominant; PI premaps would only add coverage, never remove)."""
    best = {}
    for fam, variants in EDITOR_FAMILIES.items():
        for fn in variants:
            stats = loo_editor(demos, fn)
            if fam not in best or stats[1] > best[fam][1]:
                best[fam] = stats
    return best


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
        # soft content-keyed gather families (centre-relative relational reads,
        # same KEEP/COPY voting as copy-*; measures the soft gather's
        # incremental class over the fixed copy families)
        for fam, variants in SOFTSCORE_FAMILIES.items():
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


# ---- rung #6 probe: does a CROSS-TASK (meta-learned) content read separate
# the copy-* band that the per-task closed-form vote cannot? ----------------
#
# The CF-read probe (--probe) showed the band does not yield to DETERMINISTIC
# sharpening: the only KEEP/COPY-discriminating bits are colour/object-IDENTITY
# keys, which the PER-TASK 3-demo vote finds near-0 (voting starves at 3 demos).
# Rung #6's bet: those identity features become informative when their WEIGHTING
# is CONSOLIDATED ACROSS the band (one shared "slow" read vector). This block
# measures exactly that, offline in Python, before any Mojo:
#   1. select the ~72-id copy-* band (best copy-* at LOO 0.70-0.90, fix
#      0.25-0.5, res>=floor, NOT covered by any committed family);
#   2. build a per-cell IDENTITY feature vector (rank/symmetry-normalized, never
#      raw colour) the colour-abstract vote discards, + the KEEP/COPY/CONST
#      action label loo_paired_fetch votes over;
#   3. fit ONE shared multinomial-logistic SELECTION read over cells pooled
#      ACROSS a train split of band tasks (mirrors the Mojo SelfModMemory
#      softmax head a=softmax(tau*(W.feat+b)); W is pure selection, the CONST
#      value is the per-task fast write, never in W);
#   4. freeze W, score each held-out band task with per-task LOO (only the CONST
#      value is written from that task's OWN demos), count newly `covered` ids;
#   5. GATE CF6: held-out covered >= 15 => GO (port to a Mojo ContentFetchSelfMod
#      + stream driver); < 15 => STOP (consolidation ALSO fails to separate ->
#      rung #6 re-scopes to per-family/emergent structure). Guards printed:
#      rank/sym colour only; portable-feature-only; CONST-disabled; ES-vs-logistic
#      (a clean GO wants logistic>=15 AND ES>=10, else optimizer-dependent).
#      Honesty guard: scratch/calib_cf6probe.py (shared fit, held-out scored,
#      0 false-covers incl. a NO-SHARED-STRUCTURE negative that must not be
#      memorized across the train/test split).

META_RANK_CAP = 4  # colour freq-rank buckets 0..3, >=4
META_SIZE_CAP = 3  # component size-rank buckets 0..2, >=3
KEEP_I, COPY_I, CONST_I = 0, 1, 2

# feature layout: (name, portable-to-Mojo-ContentFetch-substrate?). The
# non-portable dims (raw colour freq-rank one-hots, global register agreement)
# are the ones a per-task Mojo read + GridSubstrate cannot cleanly express; the
# portable-only ablation must itself clear the gate for a trustworthy GO.
META_FEATURES = (
    ("bias", True), ("is_bg", True), ("f_present", True),
    ("f_eq_centre", True), ("f_eq_bg", True),
    ("crank0", False), ("crank1", False), ("crank2", False),
    ("crank3", False), ("crank4", False),
    ("frank0", False), ("frank1", False), ("frank2", False),
    ("frank3", False), ("frank4", False),
    ("srank0", True), ("srank1", True), ("srank2", True), ("srank3", True),
    ("is_largest", True), ("is_smallest", True),
    ("pos_corner", True), ("pos_edge", True), ("pos_in", True), ("pos_bg", True),
    ("nd0", True), ("nd1", True), ("nd2", True),
    ("ray_agree", True), ("reg_agree", False),
)
META_DIM = len(META_FEATURES)
META_PORTABLE = [i for i, (_, p) in enumerate(META_FEATURES) if p]


def _meta_vec(P, r, c, f):
    """Per-cell identity/relational features (rank/symmetry-normalized, NEVER a
    raw colour index — the #1 false-GO leak). f = the copy-* fetched colour."""
    q, bg = P["q"], P["bg"]
    ctr = q[r][c]
    is_bg = ctr == bg
    v = [0.0] * META_DIM
    v[0] = 1.0  # bias
    v[1] = 1.0 if is_bg else 0.0
    v[2] = 1.0 if f is not None else 0.0
    v[3] = 1.0 if (f is not None and f == ctr) else 0.0
    v[4] = 1.0 if (f is not None and f == bg) else 0.0
    v[5 + min(P["rank"][ctr], META_RANK_CAP)] = 1.0
    if f is not None:
        v[10 + min(P["rank"].get(f, META_RANK_CAP), META_RANK_CAP)] = 1.0
    if not is_bg:
        i = P["comp"][r][c]
        s = P["sizes"][i]
        v[15 + min(P["size_rank"].get(s, META_SIZE_CAP), META_SIZE_CAP)] = 1.0
        v[19] = 1.0 if s == P["big"] else 0.0
        v[20] = 1.0 if s == P["small"] else 0.0
        r0, r1, c0, c1 = P["bboxes"][i]
        on_r, on_c = r in (r0, r1), c in (c0, c1)
        v[21 if (on_r and on_c) else 22 if (on_r or on_c) else 23] = 1.0
    else:
        v[24] = 1.0  # pos_bg
    v[25 + min(P["near_d"][r][c], 2)] = 1.0
    if f is not None:
        rays = P["ray"]
        v[28] = sum(rays[d][r][c] == f for d in ("u", "d", "l", "r")) / 4.0
        regs = (P["large_col"], P["small_col"], P["uniq_col"], P["major_col"])
        v[29] = sum(x is not None and x == f for x in regs) / 4.0
    return v


def _meta_label(k2, f, out):
    """The abstract action loo_paired_fetch votes over. f==centre & out==centre
    collapses KEEP==COPY -> labelled KEEP (both predict centre anyway)."""
    if out == k2:
        return KEEP_I
    if f is not None and out == f:
        return COPY_I
    return CONST_I


def _best_copy(demos):
    """Best copy-* family by chain_loo across PIs (the winning source the probe
    fits its shared read over). Returns (stats4, fam, fn, pi) or None."""
    best = None
    for pi in PI_NAMES:
        qs = [apply_pi(ig, pi) for ig, _ in demos]
        if any(q is None for q in qs):
            continue
        pres = [precompute(q) for q in qs]
        for fam, variants in COPY_FAMILIES.items():
            for fn in variants:
                stats = loo_paired_fetch(_score_copy(pres, demos, fn))
                if best is None or stats[1] > best[0][1]:
                    best = (stats, fam, fn, pi)
    return best


def _meta_task_cells(demos, fn, pi):
    """Per-demo list of (feature-vec, centre, kc-bucket, fetched, out) on the
    winning copy-* source, PI-mapped exactly as the source was scored."""
    qs = [apply_pi(ig, pi) for ig, _ in demos]
    pres = [precompute(q) for q in qs]
    per_demo = []
    for P, (_, og) in zip(pres, demos):
        cells = []
        for r in range(P["R"]):
            for c in range(P["C"]):
                key, f = fn(P, r, c)
                cells.append(
                    (_meta_vec(P, r, c, f), P["q"][r][c], key, f, og[r][c]))
        per_demo.append(cells)
    return per_demo


def _balanced_w(y):
    import numpy as np
    freq = np.bincount(y, minlength=3).astype(float)
    freq[freq == 0] = 1.0
    w = 1.0 / freq
    return w[y] / w[y].mean()


def _fit_softmax(X, y, sw, l2=1.0, steps=400, lr=0.5):
    """L2-regularized multinomial logistic (the Mojo softmax head's convex
    surrogate). Deterministic full-batch GD."""
    import numpy as np
    n, D = X.shape
    W = np.zeros((3, D))
    Y = np.zeros((n, 3))
    Y[np.arange(n), y] = 1.0
    swc = sw[:, None]
    for _ in range(steps):
        Z = X @ W.T
        Z -= Z.max(1, keepdims=True)
        Pr = np.exp(Z)
        Pr /= Pr.sum(1, keepdims=True)
        g = ((Pr - Y) * swc).T @ X / n + l2 * W / n
        W -= lr * g
    return W


def _fit_es(X, y, sw, restarts=5, steps=120, pop=24, sigma=0.6, seed=0):
    """Random-restart antithetic ES over the SAME linear read, maximizing
    train weighted action-accuracy — the optimizer-fragility guard (a separator
    the convex logistic finds should survive a derivative-free search)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n, D = X.shape

    def acc(W):
        return float((sw * (np.argmax(X @ W.T, 1) == y)).sum())

    best_W, best = None, -1.0
    for _ in range(restarts):
        W = rng.standard_normal((3, D)) * 0.3
        cur = acc(W)
        s = sigma
        for _ in range(steps):
            E = rng.standard_normal((pop, 3, D))
            grad = np.zeros((3, D))
            for e in E:
                grad += (acc(W + s * e) - acc(W - s * e)) * e
            W = W + (0.1 / (pop * s)) * grad
            s *= 0.985
            cur = acc(W)
        if cur > best:
            best, best_W = cur, W.copy()
    return best_W


def covered4(stats):
    _d2, ch, fx, res = stats
    return fx >= NET_FIX_BAR and ch >= CHAIN_LOO_BAR and res >= RESIDUAL_FLOOR


def _score_meta(demos, predict, no_const=False):
    """Held-out band scoring: per-task LOO exactly like loo_paired_fetch, but
    the KEEP/COPY/CONST DECISION is the frozen shared `predict(X)`; only the
    CONST VALUE (tcout) and the identity baseline (t2) are written from the
    task's OWN demos (matching Mojo slow-frozen / fast-from-demos). demos =
    list of (X, k2s, kcs, fs, outs)."""
    n = len(demos)
    if n < 2:
        return (0.0, 0.0, 0.0, 1.0)
    total = d2 = ch = fixed = broken = 0
    for held in range(n):
        t2, tcout = {}, {}
        for d in range(n):
            if d == held:
                continue
            _X, k2s, kcs, _fs, outs = demos[d]
            for k2, kc, out in zip(k2s, kcs, outs):
                t2.setdefault(k2, Counter())[out] += 1
                tcout.setdefault(kc, Counter())[out] += 1
        Xh, k2s, kcs, fs, outs = demos[held]
        acts = predict(Xh)
        for i in range(len(outs)):
            total += 1
            k2, kc, f, out = k2s[i], kcs[i], fs[i], outs[i]
            h2 = k2 in t2 and _det_argmax(t2[k2]) == out
            a = acts[i]
            if a == KEEP_I:
                pred = k2
            elif a == COPY_I:
                pred = f
            elif no_const:
                pred = None
            else:
                pred = _det_argmax(tcout[kc]) if kc in tcout else None
            hc = h2 if pred is None else pred == out
            d2 += h2
            ch += hc
            if hc and not h2:
                fixed += 1
            elif h2 and not hc:
                broken += 1
    miss2 = total - d2
    net_fix = (fixed - broken) / miss2 if miss2 else 0.0
    return (d2 / total, ch / total, net_fix, miss2 / total)


def meta_probe_run(tasks, cols, fitter, no_const=False, folds=4):
    """k-fold over the band tasks: fit ONE shared read on the train folds'
    pooled cells, freeze it, score each held-out band task. Returns the set of
    held-out tids newly `covered4`."""
    import numpy as np
    covered_ids = set()
    for fold in range(folds):
        train = [t for i, t in enumerate(tasks) if i % folds != fold]
        test = [t for i, t in enumerate(tasks) if i % folds == fold]
        if not train or not test:
            continue
        Xs, ys = [], []
        for t in train:
            for (X, k2s, _kcs, fs, outs) in t["demos"]:
                Xs.append(X[:, cols])
                ys.append(np.fromiter(
                    (_meta_label(k2s[i], fs[i], outs[i])
                     for i in range(len(outs))), int, len(outs)))
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        W = fitter(X, y, _balanced_w(y))

        def predict(Xh, W=W):
            return np.argmax(Xh[:, cols] @ W.T, 1)

        for t in test:
            if covered4(_score_meta(t["demos"], predict, no_const=no_const)):
                covered_ids.add(t["tid"])
    return covered_ids


def meta_probe(train_dump):
    import numpy as np
    np.random.seed(0)
    ids = deep_floor_ids(train_dump)
    print(f"rung #6 cross-task meta-read probe over {len(ids)} deep-floor ids")

    # 1. band = best copy-* in the partial-fix range AND not committed-covered.
    band = []
    for n, tid in enumerate(ids):
        demos = load_demos(tid)
        committed_cov = any(covered(s) for s in scan_task(demos).values())
        bc = _best_copy(demos)
        if bc is not None and not committed_cov:
            (_d2, ch, fx, res), fam, fn, pi = bc
            if 0.70 <= ch < 0.90 and 0.25 <= fx < 0.50 and res >= RESIDUAL_FLOOR:
                band.append((tid, fam, fn, pi))
        if (n + 1) % 25 == 0:
            print(f"  ... band scan {n + 1}/{len(ids)}  (band so far {len(band)})")
    print(f"\nband = {len(band)} copy-* near-miss ids (LOO 0.70-0.90, "
          f"fix 0.25-0.50, uncovered); baseline covered = 0/band by construction")
    if len(band) < 8:
        print("band too small to k-fold; GATE CF6 => STOP (no consolidation "
              "signal to measure)")
        return

    # 2. materialize per-cell features/labels on each band task's winning source.
    tasks = []
    for tid, _fam, fn, pi in band:
        dm = []
        for cells in _meta_task_cells(load_demos(tid), fn, pi):
            dm.append((
                np.array([c[0] for c in cells]),
                [c[1] for c in cells], [c[2] for c in cells],
                [c[3] for c in cells], [c[4] for c in cells]))
        tasks.append({"tid": tid, "demos": dm})

    allc = list(range(META_DIM))
    # 3-4. the primary run + the four false-GO guard ablations.
    full = meta_probe_run(tasks, allc, _fit_softmax)
    portable = meta_probe_run(tasks, META_PORTABLE, _fit_softmax)
    noconst = meta_probe_run(tasks, allc, _fit_softmax, no_const=True)
    es = meta_probe_run(tasks, allc, _fit_es)

    def pct(s):
        return f"{len(s):3d}/{len(band)}"

    print("\n== GATE CF6 (cross-task meta-learned selection read) ==")
    print(f"  PRIMARY  logistic, all features:   held-out covered {pct(full)}")
    print(f"  guard    portable features only:   held-out covered {pct(portable)}"
          f"   (must also clear 15)")
    print(f"  guard    CONST disabled (KEEP/COPY):held-out covered {pct(noconst)}"
          f"   (most of PRIMARY must survive)")
    print(f"  guard    ES over same linear read: held-out covered {pct(es)}"
          f"   (clean GO wants >=10)")
    go = len(full) >= 15 and len(portable) >= 15 and len(es) >= 10
    print(f"\n  GATE CF6: PRIMARY >= 15 AND portable >= 15 AND ES >= 10  =>  "
          f"{'GO' if go else 'STOP'}   (PRIMARY={len(full)}, "
          f"portable={len(portable)}, ES={len(es)})")
    if len(full) >= 15 and not go:
        print("  NOTE: PRIMARY cleared but a guard did not -> optimizer- or "
              "leak-dependent; treat as STOP (untrustworthy GO).")
    if full:
        print("  held-out covered ids: " + " ".join(sorted(full)))


def main(train_dump, probe=False):
    ids = deep_floor_ids(train_dump)
    print(f"factor scan over {len(ids)} deep-floor ids; "
          f"cover = net_fix >= {NET_FIX_BAR} & loo >= {CHAIN_LOO_BAR}"
          + ("   [+CF-READ PROBE]" if probe else ""))
    hard_content_names = (list(CONTENT_FAMILIES) + ["fetch-anchor"]
                          + list(COPY_FAMILIES))
    soft_names = list(SOFTSCORE_FAMILIES)
    content_names = hard_content_names + soft_names
    all_new = list(FAMILIES) + content_names
    cover_sets = {f: set() for f in all_new + list(BASELINES)}
    editor_sets = {f: set() for f in EDITOR_FAMILIES}
    tb_sets = {f: set() for f in COPY_FAMILIES}
    fine_sets = {f: set() for f in PROBE_COPY_FAMILIES}
    per_task = {}
    for n, tid in enumerate(ids):
        demos = load_demos(tid)
        best = scan_task(demos)
        per_task[tid] = best
        for fam, stats in best.items():
            if covered(stats):
                cover_sets[fam].add(tid)
        # rung #6 iterated-edit simulation (identity premap, best-of variants)
        for fam, (d2, ch, fx, res) in scan_editor(demos).items():
            if fx >= NET_FIX_BAR and ch >= CHAIN_LOO_BAR and res >= RESIDUAL_FLOOR:
                editor_sets[fam].add(tid)
        if probe:
            tb_best, fine_best = scan_task_probe(demos)
            for fam, stats in tb_best.items():
                if covered(stats):
                    tb_sets[fam].add(tid)
            for fam, stats in fine_best.items():
                if covered(stats):
                    fine_sets[fam].add(tid)
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
    hard_content_union = set().union(
        *(cover_sets[f] for f in hard_content_names))
    soft_union = set().union(*(cover_sets[f] for f in soft_names))
    soft_incremental = soft_union - hard_content_union
    print(f"\nunion of NEW families: {len(union)}/{len(ids)}"
          f"   (proven baselines: {len(base_union)};"
          f" new-only: {len(union - base_union)})")
    print(f"  object-level per-cell group: {len(obj_union)}"
          f"   CONTENT-ADDRESSED group: {len(content_union)}"
          f"   (gate CF: content union >= 20)")
    # PRE-REGISTERED soft-gather gate (plan, before this run): the soft
    # content-keyed gather is built in Mojo iff its centre-relative relational
    # class newly COVERS >= 15 deep-floor ids the hard copy-*/fetch-* families
    # do NOT already cover. Below => documented STOP for the soft gather;
    # Phase 2 (rung #6 constructive editor) proceeds independently.
    print(f"  SOFT-GATHER group: {len(soft_union)}"
          f"   incremental over hard-content: {len(soft_incremental)}"
          f"   (gate SOFT: incremental >= 15  =>  "
          f"{'GO' if len(soft_incremental) >= 15 else 'STOP'})")
    if soft_incremental:
        print("  soft-incremental ids: "
              + " ".join(sorted(soft_incremental)))

    # rung #6 constructive editor: coverage of the iterated-edit simulation.
    editor_union = set().union(*editor_sets.values()) if editor_sets else set()
    editor_incremental = editor_union - union
    print("\nrung #6 iterated-editor coverage (colour-abstract local rule, "
          "evolving grid, <=%d passes):" % EDITOR_T)
    for fam, s in sorted(editor_sets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {fam:12s} {len(s):3d}")
    # PRE-REGISTERED editor gate (plan, before this run): the constructive
    # editor is built in Mojo iff its iterated simulation newly COVERS >= 15
    # deep-floor ids that the ENTIRE per-cell/content new-family union does NOT
    # reach at the 0.9 bar (genuinely new capability — iterative construction,
    # positions-written != positions-read — not a re-expression of a per-cell
    # read CF already grazes). Below => documented STOP for rung #6.
    print(f"  EDITOR union: {len(editor_union)}"
          f"   incremental over per-cell/content union: "
          f"{len(editor_incremental)}"
          f"   (gate EDITOR: incremental >= 15  =>  "
          f"{'GO' if len(editor_incremental) >= 15 else 'STOP'})")
    if editor_incremental:
        print("  editor-incremental ids: "
              + " ".join(sorted(editor_incremental)))

    if probe:
        # PRE-REGISTERED CF-read probe gate (plan approved before this run):
        # sharpen CF's existing content read with DETERMINISTIC fixes
        # (tie-break + finer keys). GO to porting into Mojo write_content iff
        # the sharpened copy-* families newly COVER >= 15 deep-floor ids the
        # committed new-family union does NOT already reach. Below => documented
        # STOP = rung #6's opening evidence (closed-form voting can't be
        # deterministically sharpened -> the meta-learned read is justified).
        tb_union = set().union(*tb_sets.values()) if tb_sets else set()
        fine_union = set().union(*fine_sets.values()) if fine_sets else set()
        probe_union = tb_union | fine_union
        tb_incr = tb_union - union
        fine_incr = fine_union - union
        probe_incr = probe_union - union
        print("\n== CF-READ PROBE (deterministic sharpening of the copy-* "
              "read) ==")
        print("  per-family covered (finer key + det vote):")
        for fam, s in sorted(fine_sets.items(), key=lambda kv: -len(kv[1])):
            print(f"    {fam:14s} {len(s):3d}")
        print(f"  tie-break-only union: {len(tb_union)}"
              f"   incremental over committed: {len(tb_incr)}")
        print(f"  finer-key+det union:  {len(fine_union)}"
              f"   incremental over committed: {len(fine_incr)}")
        print(f"  PROBE union (tb | fine): {len(probe_union)}"
              f"   incremental over committed: {len(probe_incr)}"
              f"   (gate CFPROBE: incremental >= 15  =>  "
              f"{'GO' if len(probe_incr) >= 15 else 'STOP'})")
        if probe_incr:
            print("  probe-incremental ids: " + " ".join(sorted(probe_incr)))
            for tid in sorted(probe_incr):
                src = "tb" if tid in tb_incr else "fine"
                pick = fine_sets if tid in fine_union else tb_sets
                fam = max((f for f in pick if tid in pick[f]),
                          key=lambda f: 1)
                print(f"    {tid}  {fam:14s} [{src}]")

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
    flags = ("--probe", "--meta-probe")
    args = [a for a in sys.argv[1:] if a not in flags]
    dump = args[0] if args else "scratch/arc2_train_v3.txt"
    if "--meta-probe" in sys.argv:
        meta_probe(dump)
    else:
        main(dump, probe="--probe" in sys.argv)
