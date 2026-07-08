# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.math import round
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_content_fetch.mojo`.
from hope import ArcGrid, ArcTaskPair, COLOR_DIM
from memory_composed import (
    ContentFetchComposedMemory,
    LocalWriteComposedMemory,
    CONTENTFETCH_DIM,
    CONTENTFETCH_OFF,
)
from esper_evolution import (
    fit_content,
    FIT_N,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_ITERS,
    FIT_REG,
)
from arc_io import exact_match

# ==========================================================================
# RUNG CF BUILD — the content-fetch layer (the written content-keyed gather).
# The deep-floor audits measured the 146-task floor as CONTENT-ADDRESSED
# construction (output written where the input evidence is NOT); the
# content-read scan's GO (22/146) located the coverage in relational keys
# choosing KEEP/COPY actions over sharp fetch views. This proves
# ContentFetchComposedMemory expresses that class — a written view + action
# table composed on the full LocalWrite prefix — cold and held-out.
#
#   Ckpt A — {ray_down, recolor_largest, halo_nearest, anchor_shift,
#            objlocal_mirror} each >= 0.95 held-out at a FRESH grid with
#            FRESH colours, per-task cold (colours vary per demo, so a colour
#            table cannot memorize — the classes are genuinely content-
#            addressed).
#   Control (ablation) — the SAME fitted state through the LocalWrite prefix
#            ONLY fails ray_down: the content layer is load-bearing.
#   Control (strict superset) — a PURE recolor task writes NO view (slot -1)
#            AND stays >= 0.95: byte-identical to the LocalWrite path.
#   Few-demo — ray_down at n=3 (the corpus median) clears 0.9.
#
# Ground-truth rules mirror tools/synth_tasks.py CONTENT_TRANSFORMS; the halo
# BFS clones GridSubstrate's exactly (the diff_count/_local_sig precedent) so
# nearest-colour ties resolve identically on both sides.
# ==========================================================================


def ri(lo: Int, hi: Int) -> Int:
    return lo + Int(random_float64(0.0, Float64(hi - lo + 1)))


# 4 sparse dots of random colours on bg 0 (the ray class's support).
def dot_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = 0.0
    var placed = 0
    while placed < 4:
        var r = ri(0, rows - 1)
        var c = ri(0, cols - 1)
        if g.data[r * cols + c] == 0.0:
            g.data[r * cols + c] = Float32(ri(1, 9))
            placed += 1
    return g^


# 3 non-overlapping INTERIOR solid squares of distinct colours and distinct
# sizes on bg 0 (1-cell margin off every border and between blobs, so each
# blob is one component with a known bbox and the background is one component
# whose bbox is the whole grid). Returns the grid; the largest blob's colour
# and bbox corner are recomputed by the rules from tracked metadata.
struct BlobMeta(Copyable, Movable):
    var col: Int
    var h: Int
    var r0: Int
    var c0: Int

    def __init__(out self, col: Int, h: Int, r0: Int, c0: Int):
        self.col = col
        self.h = h
        self.r0 = r0
        self.c0 = c0


def blob_grid(
    rows: Int, cols: Int, punch: Bool, mut meta: List[BlobMeta]
) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = 0.0
    meta.clear()
    # distinct colours and sizes by rejection
    var cols3 = List[Int]()
    while len(cols3) < 3:
        var col = ri(1, 9)
        var dup = False
        for i in range(len(cols3)):
            if cols3[i] == col:
                dup = True
        if not dup:
            cols3.append(col)
    var sizes3 = List[Int]()
    while len(sizes3) < 3:
        var s = ri(1, 4)
        var dup = False
        for i in range(len(sizes3)):
            if sizes3[i] == s:
                dup = True
        if not dup:
            sizes3.append(s)
    for b in range(3):
        var h = sizes3[b]
        for _ in range(80):
            var r0 = ri(1, rows - h - 1)
            var c0 = ri(1, cols - h - 1)
            var ok = True
            for r in range(r0 - 1, r0 + h + 1):
                for c in range(c0 - 1, c0 + h + 1):
                    if g.data[r * cols + c] != 0.0:
                        ok = False
            if ok:
                for r in range(r0, r0 + h):
                    for c in range(c0, c0 + h):
                        g.data[r * cols + c] = Float32(cols3[b])
                # non-corner edge punch (bbox intact, h-asymmetric)
                if punch and h >= 3:
                    g.data[(r0 + 1) * cols + c0] = 0.0
                meta.append(BlobMeta(col=cols3[b], h=h, r0=r0, c0=c0))
                break
    return g^


def largest_of(meta: List[BlobMeta]) -> BlobMeta:
    var best = meta[0].copy()
    for i in range(1, len(meta)):
        if meta[i].h > best.h:
            best = meta[i].copy()
    return best^


# ---- ground-truth content-addressed rules (synth_tasks.py parity) ----


def rule_ray_down(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for k in range(g.rows * g.cols):
        out.data[k] = g.data[k]
    for c in range(g.cols):
        var last = Float32(-1.0)
        for r in range(g.rows):
            var v = g.data[r * g.cols + c]
            if v != 0.0:
                last = v
            elif last >= 0.0:
                out.data[r * g.cols + c] = last
    return out^


def rule_recolor_largest(g: ArcGrid, meta: List[BlobMeta]) -> ArcGrid:
    var big = Float32(largest_of(meta).col)
    var out = ArcGrid(g.rows, g.cols)
    for k in range(g.rows * g.cols):
        out.data[k] = big if g.data[k] != 0.0 else 0.0
    return out^


def rule_halo_nearest(g: ArcGrid) -> ArcGrid:
    # Clone of GridSubstrate's nearest-nonbg BFS (row-major seeds, FIFO,
    # relax-on-improve) so colour ties resolve identically; halo at d <= 2.
    var rows = g.rows
    var cols = g.cols
    var n = rows * cols
    var near_col = alloc[Int](n)
    var near_d = alloc[Int](n)
    var queue = alloc[Int](n)
    var head = 0
    var tail = 0
    for k in range(n):
        if g.data[k] != 0.0:
            near_col[k] = Int(round(g.data[k]))
            near_d[k] = 0
            queue[tail] = k
            tail += 1
        else:
            near_col[k] = -1
            near_d[k] = 5
    while head < tail:
        var k = queue[head]
        head += 1
        if near_d[k] + 1 >= 5:
            continue
        var r = k // cols
        var c = k % cols
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                var rr = r + dr
                var cc = c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                var kk = rr * cols + cc
                if near_d[kk] > near_d[k] + 1:
                    near_d[kk] = near_d[k] + 1
                    near_col[kk] = near_col[k]
                    queue[tail] = kk
                    tail += 1
    var out = ArcGrid(rows, cols)
    for k in range(n):
        out.data[k] = g.data[k]
        if g.data[k] == 0.0 and near_d[k] <= 2 and near_col[k] >= 0:
            out.data[k] = Float32(near_col[k])
    near_col.free()
    near_d.free()
    queue.free()
    return out^


def rule_anchor_shift(g: ArcGrid, meta: List[BlobMeta]) -> ArcGrid:
    var big = largest_of(meta)
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            var rr = (r + big.r0) % g.rows
            var cc = (c + big.c0) % g.cols
            out.data[r * g.cols + c] = g.data[rr * g.cols + cc]
    return out^


def rule_objlocal_mirror(g: ArcGrid, meta: List[BlobMeta]) -> ArcGrid:
    # Blob cells mirror within their rectangle bbox; background cells (the
    # single bg component, bbox = whole grid) mirror across the grid.
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            out.data[r * g.cols + c] = g.data[r * g.cols + (g.cols - 1 - c)]
    for i in range(len(meta)):
        var m = meta[i].copy()
        for r in range(m.r0, m.r0 + m.h):
            for c in range(m.c0, m.c0 + m.h):
                if g.data[r * g.cols + c] != 0.0:  # skip the punched cell
                    var mc = m.c0 + (m.c0 + m.h - 1) - c
                    out.data[r * g.cols + c] = g.data[r * g.cols + mc]
    return out^


def make_pair(name: String, rows: Int, cols: Int) -> ArcTaskPair:
    if name == "ray_down":
        var gin = dot_grid(rows, cols)
        var gout = rule_ray_down(gin)
        return ArcTaskPair(gin^, gout^)
    var meta = List[BlobMeta]()
    var gin = blob_grid(rows, cols, name == "objlocal_mirror", meta)
    var gout: ArcGrid
    if name == "recolor_largest":
        gout = rule_recolor_largest(gin, meta)
    elif name == "halo_nearest":
        gout = rule_halo_nearest(gin)
    elif name == "anchor_shift":
        gout = rule_anchor_shift(gin, meta)
    elif name == "objlocal_mirror":
        gout = rule_objlocal_mirror(gin, meta)
    else:  # "recolor": the pure-prefix strict-superset control
        gout = ArcGrid(rows, cols)
        for k in range(rows * cols):
            gout.data[k] = Float32((Int(round(gin.data[k])) + 1) % COLOR_DIM)
    return ArcTaskPair(gin^, gout^)


def make_demos(name: String, n: Int, rows: Int, cols: Int) -> List[ArcTaskPair]:
    var demos = List[ArcTaskPair]()
    for _ in range(n):
        demos.append(make_pair(name, rows, cols))
    return demos^


def eval_held_out(
    name: String,
    state: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    ablate: Bool,
) raises -> Float32:
    var t = make_pair(name, rows, cols)
    var pred = alloc[Float32](rows * cols)
    if ablate:
        # LocalWrite prefix only (no content fetch) — the ablation control.
        LocalWriteComposedMemory.apply(state, t.input_grid, pred)
    else:
        ContentFetchComposedMemory.apply(state, t.input_grid, pred)
    var m = exact_match(pred, t.output_grid.data, rows * cols)
    pred.free()
    return m


def written_view(state: UnsafePointer[Float32, MutAnyOrigin]) -> Int:
    return Int(round(state[CONTENTFETCH_OFF]))


def fit_task(
    name: String, task_seed: Int, n: Int, rows: Int, cols: Int
) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    seed(task_seed)
    var demos = make_demos(name, n, rows, cols)
    var state = alloc[Float32](CONTENTFETCH_DIM)
    ContentFetchComposedMemory.seed(state)
    fit_content(
        state,
        demos,
        rows * cols,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )
    return state


def main() raises:
    comptime R = 12
    comptime C = 12
    print("== Rung CF build: the written content-fetch layer ==")

    # Ckpt A — the five content-addressed classes, cold, held-out.
    var names = List[String]()
    names.append("ray_down")
    names.append("recolor_largest")
    names.append("halo_nearest")
    names.append("anchor_shift")
    names.append("objlocal_mirror")
    for i in range(len(names)):
        var st = fit_task(names[i], i + 1, 8, R, C)
        var ho = eval_held_out(names[i], st, R, C, False)
        print("  ", names[i], " held-out:", ho, " view:", written_view(st))
        if ho < 0.95:
            st.free()
            raise Error("FAIL: " + names[i] + " held-out < 0.95")
        # Ablation control on ray_down: the prefix alone must fail.
        if names[i] == "ray_down":
            var abl = eval_held_out(names[i], st, R, C, True)
            print(
                "   control (ablation, LocalWrite prefix only) held-out:",
                abl,
            )
            if abl >= 0.95:
                st.free()
                raise Error(
                    "FAIL: ablation did not degrade — content layer not"
                    " load-bearing"
                )
        st.free()

    # Strict superset — a pure recolor writes NO view and stays high.
    var sr = fit_task("recolor", 7, 8, R, C)
    var rho = eval_held_out("recolor", sr, R, C, False)
    var wv = written_view(sr)
    print("  recolor (strict superset) held-out:", rho, " view:", wv)
    if rho < 0.95 or wv >= 0:
        sr.free()
        raise Error(
            "FAIL: recolor superset broken (held-out or a written view)"
        )
    sr.free()

    # Few-demo — ray_down at the corpus median n=3.
    var sf = fit_task("ray_down", 11, 3, R, C)
    var fho = eval_held_out("ray_down", sf, R, C, False)
    print("  ray_down n=3 (few-demo) held-out:", fho)
    if fho < 0.9:
        sf.free()
        raise Error("FAIL: ray_down n=3 held-out < 0.9")
    sf.free()

    print("PASS: content-fetch composition proven cold + controls")
