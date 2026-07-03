from std.memory import alloc, UnsafePointer
from std.math import round, exp
from std.collections import List, InlineArray
from hope import ArcGrid, ArcTaskPair, COLOR_DIM
from arc_io import GridDomain
from memory import Memory, ShapeMemory
from memory_es import AttnGatherMemory, ATTN_DIM

# ==========================================
# Composed geometry × colour memory (ARC-AGI-2 block 5): retire the operator
# ==========================================
# The emergent replacement for the structured OperatorMemory on its whole subset
# {flip_h, flip_v, transpose, recolor} — AND their compositions (flip∘recolor),
# which the single memories cannot express. A single jointly-fit combined memory
# is a recorded NEGATIVE RESULT (the soft/sharp coupling — JOURNAL 2026-07-02);
# the working design is the energy-composition reroute (RESEARCH-NOTES.md
# 2026-07-02 #1): two modules, each fit on a signal INVARIANT to the other's
# factor, composed additively, each keeping its own sharpness.
#
# - COLOUR module: a geometry-invariant self-WRITE. For position-permutation
#   geometry the per-demo colour COUNTS are position-free (cnt_out[V(c)] =
#   cnt_in[c] whatever the permutation), so the colour table V is written
#   closed-form from count signatures — per colour c the across-demo count
#   vector is matched against every output colour's by a global-min greedy
#   INJECTIVE assignment (identity for ties/unseen — the few-demo hardening;
#   see write_color). Geometry-only task ⇒ counts unchanged ⇒ V =
#   identity; recolor ⇒ the permutation (incl. the 9→0 wrap, exactly — probe-
#   verified). A write rule in the self-mod family: the map's CONTENT is
#   inferred per task, one forward pass, never ES-searched.
# - GEOMETRY module: the proven AttnGatherMemory ES — run on demos whose inputs
#   are pre-mapped through V (COLOUR-THEN-GATHER, the same decoupling hope.mojo
#   uses). V is cellwise and the gather is position-only, so they commute; with
#   V applied first the geometry search runs on the exact B3 fitness landscape
#   (integer values, no colour-table cliff). Fitting geometry THROUGH a soft
#   V-interp read instead fails (probe: composed 0.10 — the 9→0 cliff shreds
#   the ES signal); the pre-map is load-bearing, not a convenience.
#
# Layout: [0:7] the AttnGather params (M, t, beta) | [7:17] the written V.
# `fill_scale` zeroes the V group, so even a generic ES fit over this memory
# would never move V (the perturbation AND the update are scale-multiplied);
# the real fit driver (`fit_geomcolor` in esper_evolution.mojo) fits only the
# attention slots via fit_operator[AttnGatherMemory] on the pre-mapped demos.
comptime GEOMCOLOR_V_OFF = ATTN_DIM  # the 10-entry written colour table V
comptime GEOMCOLOR_DIM = ATTN_DIM + COLOR_DIM  # 7 geometry | 10 colour


struct GeomColorComposedMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return GEOMCOLOR_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Identity gather + identity colour table: the unfit memory is the
        # identity transform (same convention as every other grid memory).
        AttnGatherMemory.seed(weights)
        for s in range(COLOR_DIM):
            weights[GEOMCOLOR_V_OFF + s] = Float32(s)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        # Attention group keeps its own preconditioner; the V group gets scale
        # 0 — V is WRITTEN from the demos, never searched (see struct comment).
        AttnGatherMemory.fill_scale(scale, ATTN_DIM)
        for s in range(COLOR_DIM):
            scale[GEOMCOLOR_V_OFF + s] = 0.0

    @staticmethod
    def _v_lookup(
        weights: UnsafePointer[Float32, MutAnyOrigin], val: Float32
    ) -> Float32:
        # Hard colour read: nearest-integer index into the written table. The
        # written V entries are themselves ~integer; round on the way out so a
        # residual soft-assignment blend never leaks a fractional colour.
        var idx = Int(round(val))
        if idx < 0:
            idx = 0
        if idx > COLOR_DIM - 1:
            idx = COLOR_DIM - 1
        return round(weights[GEOMCOLOR_V_OFF + idx])

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # Evaluation path: soft gather, then the HARD colour read in place.
        # After a fit the gather is sharp (the ES sharpens beta, ~e^-8 leakage),
        # so gather-then-colour equals the colour-then-gather used during the
        # fit; at the soft SEED beta this read mis-rounds — an unfit composed
        # memory is only approximately the identity, which is fine (every use
        # runs the fit first).
        AttnGatherMemory.apply(weights, inp, dst)
        for k in range(inp.rows * inp.cols):
            dst[k] = Self._v_lookup(weights, dst[k])

    @staticmethod
    def write_color(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
    ):
        """Write the colour table V from the demos' count signatures.

        For each demo, per-colour input and output counts are accumulated; the
        mismatch matrix sums, over demos, the squared count difference between
        input colour c and output colour c'. Assignment is a global-min greedy
        INJECTIVE matching with identity defaults for unseen colours and
        identity preference on exact ties (the few-demo hardening — see the
        inline comment). Geometry-invariant by construction (counts ignore
        position), so it runs BEFORE any geometry search. Stack accumulators
        only — no allocation.
        """
        var num = len(demos)
        if num == 0:
            return
        var mismatch = InlineArray[Float32, COLOR_DIM * COLOR_DIM](fill=0.0)
        var seen = InlineArray[Int, COLOR_DIM](fill=0)
        for d in range(num):
            var cnt_in = InlineArray[Float32, COLOR_DIM](fill=0.0)
            var cnt_out = InlineArray[Float32, COLOR_DIM](fill=0.0)
            var n = demos[d].input_grid.rows * demos[d].input_grid.cols
            for k in range(n):
                var ci = Int(round(demos[d].input_grid.data[k]))
                var co = Int(round(demos[d].output_grid.data[k]))
                if ci >= 0 and ci < COLOR_DIM:
                    cnt_in[ci] += 1.0
                    seen[ci] = 1
                if co >= 0 and co < COLOR_DIM:
                    cnt_out[co] += 1.0
            for c in range(COLOR_DIM):
                for c2 in range(COLOR_DIM):
                    var diff = cnt_in[c] - cnt_out[c2]
                    mismatch[c * COLOR_DIM + c2] += diff * diff
        # HARDENED assignment (the few-demo block; measured: at n=3 the old
        # independent-per-colour softmax left 3-5/10 tasks with a wrong or
        # colliding V). Three task-independent rules:
        #   1. UNSEEN colours (absent from every demo input) are unknowable —
        #      they default to IDENTITY (the same maximum-prior convention as
        #      the ES's identity anchor).
        #   2. Seen colours are assigned by GLOBAL-MIN GREEDY INJECTIVE
        #      matching: repeatedly take the (colour, target) pair with the
        #      smallest mismatch over unassigned colours × available targets —
        #      the map family is injective, so two colours never share a
        #      target (collisions were the dominant few-demo failure).
        #   3. TIES prefer identity (colour keeps its colour), then the lowest
        #      indices — deterministic, no tuned threshold (count vectors are
        #      integer, so genuine ties are exact).
        var assigned = InlineArray[Int, COLOR_DIM](fill=0)
        var taken = InlineArray[Int, COLOR_DIM](fill=0)
        var n_seen = 0
        for c in range(COLOR_DIM):
            if seen[c] == 0:
                weights[GEOMCOLOR_V_OFF + c] = Float32(c)
                assigned[c] = 1
            else:
                n_seen += 1
        for _ in range(n_seen):
            var best_m = Float32(1.0e30)
            var best_c = -1
            var best_t = -1
            for c in range(COLOR_DIM):
                if assigned[c] == 1:
                    continue
                for t in range(COLOR_DIM):
                    if taken[t] == 1:
                        continue
                    var m = mismatch[c * COLOR_DIM + t]
                    var better = m < best_m
                    if m == best_m:
                        # Tie: prefer an identity pair, then lower indices.
                        var new_id = 1 if c == t else 0
                        var cur_id = 1 if best_c == best_t else 0
                        if new_id > cur_id:
                            better = True
                    if better:
                        best_m = m
                        best_c = c
                        best_t = t
            weights[GEOMCOLOR_V_OFF + best_c] = Float32(best_t)
            assigned[best_c] = 1
            taken[best_t] = 1


# ==========================================
# Composed geometry × count-rule memory (the content×geometry block)
# ==========================================
# The block-5 recipe lifted one level: from cellwise colour maps to
# neighbourhood-count CONTENT rules. Task class: out = geom(M(count_P(in)))
# with geometry ∈ the permutation class AttnGather fits, predicate colour P and
# an injective count→colour map M inferred per task. No existing single memory
# expresses this class (AttnGather has no content rule; GridCountMap has no
# geometry; GeomColorComposed composes only a cellwise map).
#
# The structural fact the design rests on: the count rule is local and
# translation-covariant on the TOROIDAL Moore-8 lattice, and flips/transpose
# are symmetries of that lattice (each cell's neighbour-SET maps exactly), so
# the content rule and the geometry COMMUTE — content-then-gather is exact,
# just like block 5's colour-then-gather.
#
# - CONTENT module — a geometry-invariant self-WRITE from HISTOGRAMS. The
#   block-4 correspondence salience (per-cell (count, out) pairs) is scrambled
#   by an unknown geometry; histograms are position-free. For candidate colour
#   p the per-demo count-bin histogram n_j^d and the output-colour histogram
#   m_c^d satisfy m_c^d = n_{M⁻¹(c)}^d for EVERY demo regardless of the
#   permutation — so each bin j is matched to the colour whose across-demo
#   signature it reproduces, and P is selected by the RESIDUAL of the best
#   assignment (probe: exact 12/12 over flip/transpose/identity tasks).
#   Closed-form, one pass, never ES-searched.
# - GEOMETRY module — the proven pinned AttnGather fit on demos pre-mapped
#   through the inferred content rule (fit_geomcount in esper_evolution.mojo).
#
# Layout: [0:7] attn | [7:12] P-selection (one-hot written) | [12:17] the
# count→colour value table V. fill_scale zeroes the 10 content slots.
comptime GEOMCOUNT_A = 5  # colour alphabet (the count-rule family's scope)
comptime GEOMCOUNT_B = 5  # count bins; Moore-8 counts clamped to B-1
comptime GEOMCOUNT_P_OFF = ATTN_DIM
comptime GEOMCOUNT_V_OFF = ATTN_DIM + GEOMCOUNT_A
comptime GEOMCOUNT_DIM = ATTN_DIM + GEOMCOUNT_A + GEOMCOUNT_B


struct GeomCountComposedMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return GEOMCOUNT_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Identity gather; uniform (uninformative) P-selection; V = identity
        # over bins — the unfit memory maps every cell to its own count value,
        # which is NOT the identity transform: unlike the colour memories a
        # count rule has no identity element, so an unwritten content module is
        # honestly uninformative rather than a do-nothing prior.
        AttnGatherMemory.seed(weights)
        for i in range(GEOMCOUNT_A):
            weights[GEOMCOUNT_P_OFF + i] = 1.0 / Float32(GEOMCOUNT_A)
        for j in range(GEOMCOUNT_B):
            weights[GEOMCOUNT_V_OFF + j] = Float32(j)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        # Attention group keeps its preconditioner; the content groups are
        # WRITTEN from the demos, never searched (scale 0 freezes them on any
        # ES path — perturbation and update are both scale-multiplied).
        AttnGatherMemory.fill_scale(scale, ATTN_DIM)
        for i in range(GEOMCOUNT_A + GEOMCOUNT_B):
            scale[ATTN_DIM + i] = 0.0

    # Toroidal Moore-8 count of colour p at (r,c), clamped to B-1 (the same
    # clamped-bin convention as GridCountMapSelfModMemory).
    @staticmethod
    def _count_at(g: ArcGrid, r: Int, c: Int, p: Int) -> Int:
        var rows = g.rows
        var cols = g.cols
        var k = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                var rr = (r + dr + rows) % rows
                var cc = (c + dc + cols) % cols
                if Int(round(g.data[rr * cols + cc])) == p:
                    k += 1
        if k > GEOMCOUNT_B - 1:
            k = GEOMCOUNT_B - 1
        return k

    @staticmethod
    def _selected_p(weights: UnsafePointer[Float32, MutAnyOrigin]) -> Int:
        var best = Float32(-1.0e30)
        var p = 0
        for i in range(GEOMCOUNT_A):
            if weights[GEOMCOUNT_P_OFF + i] > best:
                best = weights[GEOMCOUNT_P_OFF + i]
                p = i
        return p

    # Cellwise content read: V[count_P(g)] at (r,c), hard (integer table).
    @staticmethod
    def content_at(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        g: ArcGrid,
        r: Int,
        c: Int,
        p: Int,
    ) -> Float32:
        return round(weights[GEOMCOUNT_V_OFF + Self._count_at(g, r, c, p)])

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # Content-THEN-gather (the proven order; they commute for this
        # geometry class, and the content read must see the ORIGINAL
        # neighbourhoods). Needs a temp grid for the content image — this is
        # the EVAL path only: the fit driver (fit_geomcount) pre-maps the
        # demos once per task instead of calling apply, so no allocation ever
        # lands in an ES hot loop.
        var p = Self._selected_p(weights)
        var tmp = ArcGrid(inp.rows, inp.cols)
        for r in range(inp.rows):
            for c in range(inp.cols):
                tmp.data[r * inp.cols + c] = Self.content_at(
                    weights, inp, r, c, p
                )
        AttnGatherMemory.apply(weights, tmp, dst)

    @staticmethod
    def write_content(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
    ):
        """Write (P, V) from the demos' histogram signatures.

        For each candidate colour p: per-demo count-bin histograms n_j^d are
        matched, bin by bin, to the output-colour histograms m_c^d (the
        across-demo squared-difference signature); the residual of the best
        assignment scores p. The winning p is written one-hot into the
        P-selection slots and its assignment into V. Geometry-invariant by
        construction (histograms ignore position). Allocation is once per
        write (never in an ES loop).
        """
        var nd = len(demos)
        if nd == 0:
            return
        # Output-colour histograms m_c^d (shared across candidates).
        var mh = alloc[Float32](nd * GEOMCOUNT_A)
        for d in range(nd):
            for c in range(GEOMCOUNT_A):
                mh[d * GEOMCOUNT_A + c] = 0.0
            var n = demos[d].output_grid.rows * demos[d].output_grid.cols
            for k in range(n):
                var col = Int(round(demos[d].output_grid.data[k]))
                if col >= 0 and col < GEOMCOUNT_A:
                    mh[d * GEOMCOUNT_A + col] += 1.0
        var nh = alloc[Float32](nd * GEOMCOUNT_B)
        var m_try = alloc[Int](GEOMCOUNT_B)
        var best_res = Float32(1.0e30)
        for p in range(GEOMCOUNT_A):
            for d in range(nd):
                for j in range(GEOMCOUNT_B):
                    nh[d * GEOMCOUNT_B + j] = 0.0
                for r in range(demos[d].input_grid.rows):
                    for c in range(demos[d].input_grid.cols):
                        nh[
                            d * GEOMCOUNT_B
                            + Self._count_at(demos[d].input_grid, r, c, p)
                        ] += 1.0
            var res = Float32(0.0)
            for j in range(GEOMCOUNT_B):
                var bj = Float32(1.0e30)
                var bc = 0
                for c in range(GEOMCOUNT_A):
                    var s = Float32(0.0)
                    for d in range(nd):
                        var diff = (
                            nh[d * GEOMCOUNT_B + j] - mh[d * GEOMCOUNT_A + c]
                        )
                        s += diff * diff
                    if s < bj:
                        bj = s
                        bc = c
                res += bj
                m_try[j] = bc
            if res < best_res:
                best_res = res
                for i in range(GEOMCOUNT_A):
                    weights[GEOMCOUNT_P_OFF + i] = 1.0 if i == p else Float32(
                        0.0
                    )
                for j in range(GEOMCOUNT_B):
                    weights[GEOMCOUNT_V_OFF + j] = Float32(m_try[j])
        mh.free()
        nh.free()
        m_try.free()


# ==========================================
# Composed SHAPE × geometry memory (Vision A / Next #1 — the shape-change seam)
# ==========================================
# The first ShapeMemory: outputs whose dims DIFFER from the input, with the
# output shape INFERRED IN-CONTEXT (never a hand-coded size heuristic). It lifts
# the composition pattern once more — a shape factor written closed-form from the
# demos, composed with the proven AttnGather content gather:
#
# - SHAPE RULE — a per-axis affine out = round(k*in + b), WRITTEN closed-form by
#   least-squares over the demo dim-pairs (`write`). Position-free shape
#   arithmetic (like write_color/write_content), so it needs no geometry
#   knowledge and runs before any search. Covers crop (k=1, b<0), subsample
#   (k=1/s), constant output (k=0). When the demos share ONE input size the
#   slope/intercept are UNDERDETERMINED (only their combination at that size is
#   pinned) — the honest analogue of the few-demo signature ties; the least-
#   squares fallback stays exact at the observed size. Identifying k and b (and
#   thus generalizing to an UNSEEN input size) requires >= 2 distinct input
#   sizes among the demos — which is exactly what the held-out proof supplies.
# - GEOMETRY — the proven AttnGatherMemory.apply_shaped run on the OUTPUT grid:
#   M = I reads a centred crop, M = sI a subsample, M = ±perm a flip/transpose
#   within the resize. ES-fit over the 7 attention slots (the shape rule is
#   frozen: fill_scale zeros its slots, the GeomColor freeze trick).
#
# Layout: [0:7] AttnGather content | [7:11] shape rule (kr, br, kc, bc). Colour
# composition (a write_color pre-map, which commutes cellwise) is a documented
# next-family extension; this first block proves the shape seam + geometry.
comptime SHAPEGEOM_SHAPE_OFF = ATTN_DIM  # kr, br, kc, bc
comptime SHAPEGEOM_DIM = ATTN_DIM + 4


struct ShapeGeomComposedMemory(ShapeMemory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return SHAPEGEOM_DIM

    @staticmethod
    def seed(state: UnsafePointer[Float32, MutAnyOrigin]):
        # Identity gather + identity shape rule (out == in): the unfit,
        # unwritten memory is the same-shape identity.
        AttnGatherMemory.seed(state)
        state[SHAPEGEOM_SHAPE_OFF + 0] = 1.0  # kr
        state[SHAPEGEOM_SHAPE_OFF + 1] = 0.0  # br
        state[SHAPEGEOM_SHAPE_OFF + 2] = 1.0  # kc
        state[SHAPEGEOM_SHAPE_OFF + 3] = 0.0  # bc

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        # Attention keeps its preconditioner; the shape-rule slots are WRITTEN,
        # never searched (scale 0 freezes them on any ES path).
        AttnGatherMemory.fill_scale(scale, ATTN_DIM)
        for i in range(4):
            scale[SHAPEGEOM_SHAPE_OFF + i] = 0.0

    # Least-squares slope/intercept of `out` on `in` over the demos for one
    # axis, written into state[k_off]/state[b_off]. `axis == 0` fits rows,
    # `axis == 1` fits cols. Falls back to the mean ratio (b = 0) when the input
    # dimension does not vary — underdetermined but exact at the observed size
    # (see the struct comment).
    @staticmethod
    def _write_axis(
        state: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
        axis: Int,
        k_off: Int,
        b_off: Int,
    ):
        var nd = len(demos)
        var s_in = Float32(0.0)
        var s_out = Float32(0.0)
        var s_in2 = Float32(0.0)
        var s_io = Float32(0.0)
        for d in range(nd):
            var din = demos[d].input_grid.rows
            var dout = demos[d].output_grid.rows
            if axis != 0:
                din = demos[d].input_grid.cols
                dout = demos[d].output_grid.cols
            var fi = Float32(din)
            var fo = Float32(dout)
            s_in += fi
            s_out += fo
            s_in2 += fi * fi
            s_io += fi * fo
        var fnd = Float32(nd)
        var denom = fnd * s_in2 - s_in * s_in
        # denom == 0 <=> every demo shares one input size (variance 0).
        if denom > Float32(1.0e-6) or denom < Float32(-1.0e-6):
            var k = (fnd * s_io - s_in * s_out) / denom
            state[k_off] = k
            state[b_off] = (s_out - k * s_in) / fnd
        else:
            # Fallback: the mean out/in ratio with zero intercept.
            var k = Float32(1.0)
            if s_in > Float32(0.0):
                k = s_out / s_in
            state[k_off] = k
            state[b_off] = Float32(0.0)

    @staticmethod
    def write(
        state: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
    ):
        if len(demos) == 0:
            return
        Self._write_axis(
            state, demos, 0, SHAPEGEOM_SHAPE_OFF + 0, SHAPEGEOM_SHAPE_OFF + 1
        )
        Self._write_axis(
            state, demos, 1, SHAPEGEOM_SHAPE_OFF + 2, SHAPEGEOM_SHAPE_OFF + 3
        )

    @staticmethod
    def out_rows(
        state: UnsafePointer[Float32, MutAnyOrigin], inp: ArcGrid
    ) -> Int:
        var v = round(
            state[SHAPEGEOM_SHAPE_OFF + 0] * Float32(inp.rows)
            + state[SHAPEGEOM_SHAPE_OFF + 1]
        )
        var r = Int(v)
        if r < 1:
            r = 1
        return r

    @staticmethod
    def out_cols(
        state: UnsafePointer[Float32, MutAnyOrigin], inp: ArcGrid
    ) -> Int:
        var v = round(
            state[SHAPEGEOM_SHAPE_OFF + 2] * Float32(inp.cols)
            + state[SHAPEGEOM_SHAPE_OFF + 3]
        )
        var c = Int(v)
        if c < 1:
            c = 1
        return c

    @staticmethod
    def apply(
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        out_rows: Int,
        out_cols: Int,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # The shape rule (in state[7:11]) is inert to the gather — the attention
        # slots [0:7] are `state` itself, so the AttnGather read is unchanged.
        AttnGatherMemory.apply_shaped(state, inp, out_rows, out_cols, dst)
