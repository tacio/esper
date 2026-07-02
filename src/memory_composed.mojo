from std.memory import UnsafePointer
from std.math import round, exp
from std.collections import List, InlineArray
from hope import ArcGrid, ArcTaskPair, COLOR_DIM
from arc_io import GridDomain
from memory import Memory
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
#   vector is matched against every output colour's, and a sharp softmax
#   (GEOMCOLOR_TAU) assigns V[c]. Geometry-only task ⇒ counts unchanged ⇒ V =
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
# Count-signature softmax sharpness: one raw-count mismatch in one of 8 demos
# gives weight exp(-32/8) ~ 0.018 — a worst-case V blend of ~0.16, round-safe.
comptime GEOMCOLOR_TAU = Float32(32.0)


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
        input colour c and output colour c'. A sharp softmax over c' rows then
        assigns V[c]. Geometry-invariant by construction (counts ignore
        position), so it runs BEFORE any geometry search. Stack accumulators
        only — no allocation.
        """
        var num = len(demos)
        if num == 0:
            return
        var mismatch = InlineArray[Float32, COLOR_DIM * COLOR_DIM](fill=0.0)
        for d in range(num):
            var cnt_in = InlineArray[Float32, COLOR_DIM](fill=0.0)
            var cnt_out = InlineArray[Float32, COLOR_DIM](fill=0.0)
            var n = demos[d].input_grid.rows * demos[d].input_grid.cols
            for k in range(n):
                var ci = Int(round(demos[d].input_grid.data[k]))
                var co = Int(round(demos[d].output_grid.data[k]))
                if ci >= 0 and ci < COLOR_DIM:
                    cnt_in[ci] += 1.0
                if co >= 0 and co < COLOR_DIM:
                    cnt_out[co] += 1.0
            for c in range(COLOR_DIM):
                for c2 in range(COLOR_DIM):
                    var diff = cnt_in[c] - cnt_out[c2]
                    mismatch[c * COLOR_DIM + c2] += diff * diff
        var inv_d = 1.0 / Float32(num)
        for c in range(COLOR_DIM):
            # Subtract the row minimum so the sharp softmax never underflows to
            # 0/0 (the best match always contributes exp(0) = 1).
            var best = Float32(1.0e30)
            for c2 in range(COLOR_DIM):
                var m = mismatch[c * COLOR_DIM + c2] * inv_d
                if m < best:
                    best = m
            var z = Float32(0.0)
            var val = Float32(0.0)
            for c2 in range(COLOR_DIM):
                var m = mismatch[c * COLOR_DIM + c2] * inv_d
                var w = exp(-GEOMCOLOR_TAU * (m - best))
                z += w
                val += w * Float32(c2)
            weights[GEOMCOLOR_V_OFF + c] = val / z
