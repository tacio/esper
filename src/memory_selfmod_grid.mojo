# The 2-D grid self-modifying memories (the ARC-AGI-2 blocks 1-4): local-rule
# memories keyed on grid neighbourhoods, each meta-learned cold through the
# generic meta_fit_selfmod. GridContext = additive centre/neighbour rules via
# outer-product keys; GridNbhd = the disjunctive/count class via a Moore-8
# histogram key and a sigmoid-threshold read; GridCountMap = arbitrary
# count->colour maps via a meta-learned scoring salience + soft bin value table.
from std.memory import UnsafePointer
from std.math import fma, round, exp, sin, sqrt
from std.collections import List
from hope import ArcGrid, ArcTaskPair, ExamplePair, COLOR_DIM
from arc_io import GridDomain
from memory import SelfModMemory
from memory_selfmod import _sigmoid

# ==========================================
# 2-D context keys (Phase B / first ARC-AGI-2 block): grid-neighbourhood self-write
# ==========================================
# Lift the B4 fuller block from a 1-D sequence neighbour to a 2-D GRID neighbourhood,
# so the memory can learn local pattern->colour rules in-context. Per cell (r,c) on a
# TOROIDAL grid (wrap avoids edge-coverage gaps), the context is (center, up, left) =
# (in[r,c], in[r-1,c], in[r,c-1]). Target class (bounded, matches the linear read):
# ADDITIVE center<->neighbour rules  out[r,c] = h1(center,up) + h2(center,left).
# Disjunctive/count/mod-wrap rules need a nonlinear read — out of scope here.
#
# CHECKPOINT A (this struct) validates the grid 2-D self-write MECHANISM with FIXED
# one-hot context keys (Dk = 2*A^2, two active entries per cell), eta fixed, alpha=0.
# The read S·k = S[ctx_up] + S[ctx_left] is an ADDITIVE decomposition the gated delta
# rule solves by SGD, so `adapt` runs a few EPOCHS over the demo cells (the memory's
# own optimizer running longer — NOT an ES search). Checkpoint B replaces the one-hot
# key with learned embeddings + self-generated gates.
comptime GRIDCTX_A = 5
comptime GRIDCTX_ONEHOT_STATE = 2 * GRIDCTX_A * GRIDCTX_A
comptime GRIDCTX_EPOCHS = 20
comptime GRIDCTX_ETA_A = Float32(0.2)


def _gridctx_sym(v: Float32) -> Int:
    var c = Int(round(v))
    if c < 0:
        return 0
    elif c > GRIDCTX_A - 1:
        return GRIDCTX_A - 1
    return c


struct GridContextSelfWrite:
    @staticmethod
    def state_dim() -> Int:
        return GRIDCTX_ONEHOT_STATE

    @staticmethod
    def adapt(
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(GRIDCTX_ONEHOT_STATE):
            state[i] = 0.0
        comptime block = GRIDCTX_A * GRIDCTX_A
        for _epoch in range(GRIDCTX_EPOCHS):
            for d in range(len(demos)):
                ref pair = demos[d]
                var rows = pair.input_grid.rows
                var cols = pair.input_grid.cols
                for r in range(rows):
                    for c in range(cols):
                        var center = _gridctx_sym(
                            pair.input_grid.data[r * cols + c]
                        )
                        var up = _gridctx_sym(
                            pair.input_grid.data[
                                ((r - 1 + rows) % rows) * cols + c
                            ]
                        )
                        var left = _gridctx_sym(
                            pair.input_grid.data[
                                r * cols + ((c - 1 + cols) % cols)
                            ]
                        )
                        var ctx1 = center * GRIDCTX_A + up
                        var ctx2 = block + center * GRIDCTX_A + left
                        var v = pair.output_grid.data[r * cols + c]
                        # Gated delta rule, one-hot key (two active entries), alpha=0.
                        var e = v - (state[ctx1] + state[ctx2])
                        state[ctx1] += GRIDCTX_ETA_A * e
                        state[ctx2] += GRIDCTX_ETA_A * e

    @staticmethod
    def apply(
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        comptime block = GRIDCTX_A * GRIDCTX_A
        var rows = inp.rows
        var cols = inp.cols
        for r in range(rows):
            for c in range(cols):
                var center = _gridctx_sym(inp.data[r * cols + c])
                var up = _gridctx_sym(
                    inp.data[((r - 1 + rows) % rows) * cols + c]
                )
                var left = _gridctx_sym(
                    inp.data[r * cols + ((c - 1 + cols) % cols)]
                )
                var ctx1 = center * GRIDCTX_A + up
                var ctx2 = block + center * GRIDCTX_A + left
                dst[r * cols + c] = state[ctx1] + state[ctx2]


# CHECKPOINT B — the emergent grid-context block. The key is the concat of two
# outer products of learned per-colour embeddings — E[center]⊗E[up] and
# E[center]⊗E[left] — unit-normalised (the B4 stability lesson); η/α are
# self-generated per cell. The gated delta write runs a few epochs over the demo
# cells (the additive read S·k = h1(center,up)+h2(center,left) is a 2-way
# decomposition SGD solves). The SLOW params (embeddings + gate projections) are
# meta-learned ONCE across a family of local rules; a fresh rule then adapts in a
# single (multi-epoch) forward pass. Reuses meta_fit_selfmod[GridContextSelfModMemory].
comptime GRIDCTX_DE = 5
comptime GRIDCTX_DK = 2 * GRIDCTX_DE * GRIDCTX_DE
comptime GRIDCTX_META_EPOCHS = 6
comptime GRIDCTX_E_OFF = 0
comptime GRIDCTX_WETA_OFF = GRIDCTX_A * GRIDCTX_DE
comptime GRIDCTX_BETA_OFF = GRIDCTX_WETA_OFF + GRIDCTX_DK
comptime GRIDCTX_WALPHA_OFF = GRIDCTX_BETA_OFF + 1
comptime GRIDCTX_BALPHA_OFF = GRIDCTX_WALPHA_OFF + GRIDCTX_DK
comptime GRIDCTX_SLOW_DIM = GRIDCTX_BALPHA_OFF + 1


struct GridContextSelfModMemory(SelfModMemory):
    comptime Dom = GridDomain

    @staticmethod
    def slow_dim() -> Int:
        return GRIDCTX_SLOW_DIM

    @staticmethod
    def state_dim() -> Int:
        return GRIDCTX_DK

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        for c in range(GRIDCTX_A):
            for d in range(GRIDCTX_DE):
                slow[GRIDCTX_E_OFF + c * GRIDCTX_DE + d] = (
                    sin(Float32(c + 1) * Float32(d + 1) * 0.7) * 0.5
                )
        for j in range(GRIDCTX_DK):
            slow[GRIDCTX_WETA_OFF + j] = 0.1 * sin(Float32(j + 1) * 1.3)
            slow[GRIDCTX_WALPHA_OFF + j] = 0.1 * sin(Float32(j + 1) * 2.1)
        slow[GRIDCTX_BETA_OFF] = 0.0
        slow[GRIDCTX_BALPHA_OFF] = -2.0

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    # Concat of two outer-product context keys (center⊗up | center⊗left), unit-norm.
    @staticmethod
    def _key(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        center: Int,
        up: Int,
        left: Int,
        mut k: InlineArray[Float32, GRIDCTX_DK],
    ):
        comptime block = GRIDCTX_DE * GRIDCTX_DE
        var sumsq = Float32(0.0)
        for a in range(GRIDCTX_DE):
            var ec = slow[GRIDCTX_E_OFF + center * GRIDCTX_DE + a]
            for b in range(GRIDCTX_DE):
                var ku = ec * slow[GRIDCTX_E_OFF + up * GRIDCTX_DE + b]
                var kl = ec * slow[GRIDCTX_E_OFF + left * GRIDCTX_DE + b]
                k[a * GRIDCTX_DE + b] = ku
                k[block + a * GRIDCTX_DE + b] = kl
                sumsq = fma(ku, ku, fma(kl, kl, sumsq))
        var inv = 1.0 / (sqrt(sumsq) + 1e-6)
        for j in range(GRIDCTX_DK):
            k[j] = k[j] * inv

    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for j in range(GRIDCTX_DK):
            state[j] = 0.0
        for _epoch in range(GRIDCTX_META_EPOCHS):
            for d in range(len(demos)):
                ref pair = demos[d]
                var rows = pair.input_grid.rows
                var cols = pair.input_grid.cols
                for r in range(rows):
                    for c in range(cols):
                        var center = _gridctx_sym(
                            pair.input_grid.data[r * cols + c]
                        )
                        var up = _gridctx_sym(
                            pair.input_grid.data[
                                ((r - 1 + rows) % rows) * cols + c
                            ]
                        )
                        var left = _gridctx_sym(
                            pair.input_grid.data[
                                r * cols + ((c - 1 + cols) % cols)
                            ]
                        )
                        var k = InlineArray[Float32, GRIDCTX_DK](fill=0.0)
                        GridContextSelfModMemory._key(slow, center, up, left, k)
                        var pred = Float32(0.0)
                        var geta = slow[GRIDCTX_BETA_OFF]
                        var galpha = slow[GRIDCTX_BALPHA_OFF]
                        for j in range(GRIDCTX_DK):
                            pred = fma(state[j], k[j], pred)
                            geta = fma(slow[GRIDCTX_WETA_OFF + j], k[j], geta)
                            galpha = fma(
                                slow[GRIDCTX_WALPHA_OFF + j], k[j], galpha
                            )
                        var eta = _sigmoid(geta)
                        var alpha = _sigmoid(galpha)
                        var e = pair.output_grid.data[r * cols + c] - pred
                        for j in range(GRIDCTX_DK):
                            state[j] = (1.0 - alpha) * state[j] + eta * e * k[j]

    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var rows = inp.rows
        var cols = inp.cols
        for r in range(rows):
            for c in range(cols):
                var center = _gridctx_sym(inp.data[r * cols + c])
                var up = _gridctx_sym(
                    inp.data[((r - 1 + rows) % rows) * cols + c]
                )
                var left = _gridctx_sym(
                    inp.data[r * cols + ((c - 1 + cols) % cols)]
                )
                var k = InlineArray[Float32, GRIDCTX_DK](fill=0.0)
                GridContextSelfModMemory._key(slow, center, up, left, k)
                var pred = Float32(0.0)
                for j in range(GRIDCTX_DK):
                    pred = fma(state[j], k[j], pred)
                dst[r * cols + c] = pred


# ==========================================
# Richer neighbourhoods + NONLINEAR read (ARC-AGI-2 blocks 2-3)
# ==========================================
# GridContext (above) reads pred = S·k — LINEAR in the key — so it expresses only
# ADDITIVE positional rules. It provably cannot do the DISJUNCTIVE / COUNT class:
# "output = C1 if (# neighbours matching a colour) >= t else C2" (an OR / threshold
# on a count). Crossing that barrier needs BOTH a richer neighbourhood AND a
# nonlinear read.
#
# Key idea: a Moore-8 (toroidal 3x3-minus-center) neighbourhood, aggregated into a
# MEAN neighbour-embedding histogram k = (1/8) Σ_{n∈nbrs} E[n] (plus a bias slot).
# The SUM over neighbours is the crux — it encodes per-colour neighbour COUNTS
# (position-invariant), exactly what a count/disjunction rule needs, and precisely
# what GridContext's per-pair concat key does not give. The key is CENTER-FREE:
# the disjunctive/count class depends only on the neighbourhood, so tying the
# histogram to the centre would only fragment the per-colour weight across centres
# and inject irrelevant centre-dependence (empirically it fails to learn). With a
# centre-free histogram every cell trains the same small weight vector — clean
# logistic-regression-style learning. We MEAN (÷8) rather than unit-normalise:
# normalising would divide out the very count magnitude the threshold reads, while
# the mean still bounds ‖k‖ (delta-write stability, the B4 lesson).
#
# Read (the nonlinearity): pred = v0·(1−σ) + v1·σ with σ = σ(g·(S·k) + c). σ makes
# a THRESHOLD on the weighted count = the disjunction/majority a linear S·k can't.
# BLOCK 3 (broadening): the memory infers the WHOLE 2-level rule from demos —
# predicate colour, threshold t, AND both output colours C1/C2 — not just the fixed
# {0,4}/t=2 of the first count block.
#
# The trick that makes this robust is DECOUPLING colour from threshold. The two
# output colours are read straight off the demos (v0/v1 = min/max output, WRITTEN
# into state), and the salience S is trained as a BINARY CLASSIFIER of which colour
# a cell outputs (label y = 1 if output == vmax else 0) — NOT by regressing the raw
# target. So S inherits block 2's proven logistic count-write verbatim
# (S ← (1−α)S + η·(y−σ)·σ(1−σ)·g·k, η/α self-generated); the classifier's LEARNED
# SIGN handles inverted rules (fire → the SMALLER colour), which a value-coupled
# output head fights over. Variable t: the key's bias slot + c self-calibrate the
# threshold offset. The fast state [S | v0 | v1] is WRITTEN over the demos (never
# ES-searched); the ES fits only the small slow vector (E, read scalars g/c, the
# gate projections) across a family — the B3→B4 discipline. Reuses
# meta_fit_selfmod[GridNbhdSelfModMemory] verbatim.
comptime GRIDNBHD_A = 5
comptime GRIDNBHD_DE = 5  # = A so the Ckpt-A fixed one-hot embedding is expressible
# Dk = De histogram features + 1 constant BIAS slot. The bias lets the self-write
# learn the threshold OFFSET (so the step lands between the firing counts
# regardless of the learned magnitude, and the threshold t can itself be learned).
comptime GRIDNBHD_DK = GRIDNBHD_DE + 1
comptime GRIDNBHD_NBRS = 8
comptime GRIDNBHD_META_EPOCHS = 12
# Base learning rate for the salience classifier write (cross-entropy logistic
# delta). The self-generated eta in (0,1) modulates it; this constant restores
# the update magnitude block 2 got for free from its (HI-LO)=4 read gain.
comptime GRIDNBHD_LR = Float32(8.0)

# The two output colours are NOT meta-fixed — they are WRITTEN per task into the
# state as a 2-unit output head [v0, v1] (v0 = low-regime colour, v1 = high). So
# the state is [ S (Dk salience) | v0 | v1 ]. This is what lets the memory infer
# arbitrary output pairs (and, with the key's bias slot, an arbitrary threshold t)
# in-context, not just the fixed {0,4} / t=2 of the first count block.
comptime GRIDNBHD_V0_OFF = GRIDNBHD_DK
comptime GRIDNBHD_V1_OFF = GRIDNBHD_DK + 1
comptime GRIDNBHD_STATE_DIM = GRIDNBHD_DK + 2

comptime GRIDNBHD_E_OFF = 0
comptime GRIDNBHD_G_OFF = GRIDNBHD_A * GRIDNBHD_DE
comptime GRIDNBHD_C_OFF = GRIDNBHD_G_OFF + 1
comptime GRIDNBHD_WETA_OFF = GRIDNBHD_C_OFF + 1
comptime GRIDNBHD_BETA_OFF = GRIDNBHD_WETA_OFF + GRIDNBHD_DK
comptime GRIDNBHD_WALPHA_OFF = GRIDNBHD_BETA_OFF + 1
comptime GRIDNBHD_BALPHA_OFF = GRIDNBHD_WALPHA_OFF + GRIDNBHD_DK
comptime GRIDNBHD_SLOW_DIM = GRIDNBHD_BALPHA_OFF + 1


struct GridNbhdSelfModMemory(SelfModMemory):
    comptime Dom = GridDomain

    @staticmethod
    def slow_dim() -> Int:
        return GRIDNBHD_SLOW_DIM

    @staticmethod
    def state_dim() -> Int:
        return GRIDNBHD_STATE_DIM

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        # ZERO embeddings — the "no learned representation" prior. With E=0 the
        # histogram key is 0, so the read is constant (baseline) and the self-write
        # cannot classify: the meta-fit must DISCOVER separable colour embeddings
        # from scratch, so the emergence claim is non-vacuous (Ckpt B asserts the
        # generic seed fails to clear the bar). Unlike a `sin` seed — which is
        # already separable enough for the strong cross-entropy write to generalise
        # on its own — zero is genuinely uninformative.
        for c in range(GRIDNBHD_A):
            for d in range(GRIDNBHD_DE):
                slow[GRIDNBHD_E_OFF + c * GRIDNBHD_DE + d] = 0.0
        slow[GRIDNBHD_G_OFF] = 2.0  # read sharpness
        slow[GRIDNBHD_C_OFF] = 0.0  # read threshold bias
        for j in range(GRIDNBHD_DK):
            slow[GRIDNBHD_WETA_OFF + j] = 0.1 * sin(Float32(j + 1) * 1.3)
            slow[GRIDNBHD_WALPHA_OFF + j] = 0.1 * sin(Float32(j + 1) * 2.1)
        slow[GRIDNBHD_BETA_OFF] = 0.0  # eta ~ 0.5
        slow[
            GRIDNBHD_BALPHA_OFF
        ] = -6.0  # alpha ~ 0.002 (near-pure accumulation)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    # The 8 toroidal Moore neighbours of (r,c), colour-symbolised, in a fixed order.
    @staticmethod
    def _gather8(
        data: UnsafePointer[Float32, MutAnyOrigin],
        rows: Int,
        cols: Int,
        r: Int,
        c: Int,
        mut nbrs: InlineArray[Int, GRIDNBHD_NBRS],
    ):
        var idx = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                var rr = (r + dr + rows) % rows
                var cc = (c + dc + cols) % cols
                nbrs[idx] = _gridctx_sym(data[rr * cols + cc])
                idx += 1

    # Mean-aggregated, centre-free neighbour-embedding histogram: (1/8) Σ_n E[n],
    # plus a constant bias slot. Counts stay proportional (the nonlinear read
    # thresholds them); ‖k‖ bounded, so no unit-norm needed.
    @staticmethod
    def _key(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        nbrs: InlineArray[Int, GRIDNBHD_NBRS],
        mut k: InlineArray[Float32, GRIDNBHD_DK],
    ):
        comptime De = GRIDNBHD_DE
        var inv = 1.0 / Float32(GRIDNBHD_NBRS)
        for b in range(De):
            var acc = Float32(0.0)
            for i in range(GRIDNBHD_NBRS):
                acc += slow[GRIDNBHD_E_OFF + nbrs[i] * De + b]
            k[b] = acc * inv
        k[De] = 1.0  # constant bias slot (threshold offset)

    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for j in range(GRIDNBHD_DK):
            state[j] = 0.0

        # Read the two output colours straight off the demos (min/max), and store
        # them as the written output levels v0/v1. DECOUPLED from the salience: the
        # salience is trained as a BINARY CLASSIFIER of which colour a cell outputs
        # (label y = 1 if output == vmax else 0), not by regressing the raw target.
        # This is the crux of the block-3 generalisation — it inherits block 2's
        # proven logistic count-write verbatim, while the classifier's LEARNED SIGN
        # handles inverted rules (fire → the smaller colour) that a value-coupled
        # head fights over. Arbitrary colours: v0/v1. Arbitrary threshold t: the
        # key's bias slot + c self-calibrate the offset.
        var vmin = demos[0].output_grid.data[0]
        var vmax = vmin
        for d in range(len(demos)):
            ref pg = demos[d].output_grid
            for i in range(pg.rows * pg.cols):
                var v = pg.data[i]
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
        state[GRIDNBHD_V0_OFF] = vmin
        state[GRIDNBHD_V1_OFF] = vmax
        var mid = 0.5 * (vmin + vmax)
        var g = slow[GRIDNBHD_G_OFF]
        var cc = slow[GRIDNBHD_C_OFF]

        for _epoch in range(GRIDNBHD_META_EPOCHS):
            for d in range(len(demos)):
                ref pair = demos[d]
                var rows = pair.input_grid.rows
                var cols = pair.input_grid.cols
                for r in range(rows):
                    for c in range(cols):
                        var nbrs = InlineArray[Int, GRIDNBHD_NBRS](fill=0)
                        GridNbhdSelfModMemory._gather8(
                            pair.input_grid.data, rows, cols, r, c, nbrs
                        )
                        var k = InlineArray[Float32, GRIDNBHD_DK](fill=0.0)
                        GridNbhdSelfModMemory._key(slow, nbrs, k)
                        var z = Float32(0.0)
                        var geta = slow[GRIDNBHD_BETA_OFF]
                        var galpha = slow[GRIDNBHD_BALPHA_OFF]
                        for j in range(GRIDNBHD_DK):
                            z = fma(state[j], k[j], z)
                            geta = fma(slow[GRIDNBHD_WETA_OFF + j], k[j], geta)
                            galpha = fma(
                                slow[GRIDNBHD_WALPHA_OFF + j], k[j], galpha
                            )
                        var sig = _sigmoid(g * z + cc)
                        # Binary label: does this cell output the HIGH colour?
                        var y = Float32(1.0) if pair.output_grid.data[
                            r * cols + c
                        ] > mid else Float32(0.0)
                        # Cross-entropy logistic delta (no vanishing sig(1-sig) so
                        # it doesn't stall under a sharp read), gated + gain-scaled.
                        var e = y - sig
                        var eta = _sigmoid(geta)
                        var alpha = _sigmoid(galpha)
                        var upd = GRIDNBHD_LR * eta * e
                        for j in range(GRIDNBHD_DK):
                            state[j] = (1.0 - alpha) * state[j] + upd * k[j]

    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var rows = inp.rows
        var cols = inp.cols
        var g = slow[GRIDNBHD_G_OFF]
        var cc = slow[GRIDNBHD_C_OFF]
        var v0 = state[GRIDNBHD_V0_OFF]
        var v1 = state[GRIDNBHD_V1_OFF]
        for r in range(rows):
            for c in range(cols):
                var nbrs = InlineArray[Int, GRIDNBHD_NBRS](fill=0)
                GridNbhdSelfModMemory._gather8(inp.data, rows, cols, r, c, nbrs)
                var k = InlineArray[Float32, GRIDNBHD_DK](fill=0.0)
                GridNbhdSelfModMemory._key(slow, nbrs, k)
                var z = Float32(0.0)
                for j in range(GRIDNBHD_DK):
                    z = fma(state[j], k[j], z)
                var sig = _sigmoid(g * z + cc)
                dst[r * cols + c] = v0 * (1.0 - sig) + v1 * sig


# ==========================================
# Multi-bin count -> colour MAP (ARC-AGI-2 block 4)
# ==========================================
# Blocks 2-3 read a colour's Moore-8 neighbour COUNT through a single sigmoid ->
# a 2-LEVEL (threshold) output. This memory generalises the read to an ARBITRARY
# map out = M(count_P): count -> colour, non-contiguous / non-monotone, >= 3
# output colours. Both the predicate colour P and the map M are inferred per task.
#
# WHY NOT a gradient self-write (the block-3 pattern): at S=0 every cell shares the
# count-score, so the only signal driving a gradient-written salience is the LINEAR
# COVARIANCE of count_c with the output -- which VANISHES for non-monotone maps (the
# whole point here). Verified empirically: the coupled write fails/erratic on
# non-monotone maps. Identifying WHICH colour to count is a discrete SELECTION, not
# a smooth gradient target.
#
# Mechanism: a META-LEARNED SCORING salience + a soft count-bin value table.
#   1. Per colour c, compute demo STATISTICS of how count_c relates to the output:
#      the variance REDUCTION  Var(out) - E_j[Var(out | count_c=j)]  (captures ANY
#      functional dependence, monotone or not), the linear correlation, and the mean
#      count. A meta-learned linear score picks the predicate: a = softmax(tau*(w.feat)).
#      The meta-fit learns to weight variance-reduction OVER correlation -- correlation
#      is exactly what fails on non-monotone maps, so the memory LEARNS the right
#      statistic. (w=0 seed -> uniform a -> constant read -> fails: non-vacuous.)
#   2. Count-score z = sum_c a[c]*count_c (a ~ onehot(P) -> z = count_P, integer-scaled).
#   3. Soft count-bins phi_j = softmax_j(-temp*(z-mu_j)^2); value table V[j] WRITTEN by
#      a delta keyed by the soft bin; read pred = sum_j phi_j*V[j] (mu/temp meta-learned).
# The fast state [a | V] is WRITTEN from the demos in one pass (never ES-searched);
# the ES fits only the small slow vector (scoring w/bias/tau, temp, mu, value rate).
comptime GRIDCMAP_A = 5
comptime GRIDCMAP_NBRS = 8
comptime GRIDCMAP_B = 5  # count bins (observable count_P range is ~0..4)
comptime GRIDCMAP_NFEAT = 3  # per-colour features: [reduction, corr, mean_count]
comptime GRIDCMAP_EPOCHS = 8

comptime GRIDCMAP_V_OFF = GRIDCMAP_A  # value table follows the A selection weights
comptime GRIDCMAP_STATE_DIM = GRIDCMAP_A + GRIDCMAP_B

comptime GRIDCMAP_W_OFF = 0  # scoring weights over NFEAT features
comptime GRIDCMAP_BSCORE_OFF = GRIDCMAP_NFEAT
comptime GRIDCMAP_TAU_OFF = GRIDCMAP_BSCORE_OFF + 1  # selection sharpness
comptime GRIDCMAP_TEMP_OFF = GRIDCMAP_TAU_OFF + 1  # bin sharpness
comptime GRIDCMAP_MU_OFF = GRIDCMAP_TEMP_OFF + 1
comptime GRIDCMAP_EV_OFF = GRIDCMAP_MU_OFF + GRIDCMAP_B  # value-table write rate
comptime GRIDCMAP_SLOW_DIM = GRIDCMAP_EV_OFF + 1


struct GridCountMapSelfModMemory(SelfModMemory):
    comptime Dom = GridDomain

    @staticmethod
    def slow_dim() -> Int:
        return GRIDCMAP_SLOW_DIM

    @staticmethod
    def state_dim() -> Int:
        return GRIDCMAP_STATE_DIM

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        # w = 0: uninformative scoring -> uniform colour selection -> constant read
        # -> the generic seed fails; the meta-fit must DISCOVER which statistic
        # (variance-reduction) identifies the predicate colour (non-vacuous).
        for i in range(GRIDCMAP_NFEAT):
            slow[GRIDCMAP_W_OFF + i] = 0.0
        slow[GRIDCMAP_BSCORE_OFF] = 0.0
        slow[GRIDCMAP_TAU_OFF] = 1.0
        slow[
            GRIDCMAP_TEMP_OFF
        ] = 4.0  # bins need to be sharp (probe: temp >= 3)
        for j in range(GRIDCMAP_B):
            slow[GRIDCMAP_MU_OFF + j] = Float32(
                j
            )  # bins at integer counts 0..4
        slow[GRIDCMAP_EV_OFF] = 0.0  # eta_v ~ 0.5

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    # Moore-8 colour histogram at (r,c), toroidal: hist[colour] = neighbour count.
    @staticmethod
    def _hist(
        data: UnsafePointer[Float32, MutAnyOrigin],
        rows: Int,
        cols: Int,
        r: Int,
        c: Int,
        mut hist: InlineArray[Float32, GRIDCMAP_A],
    ):
        for i in range(GRIDCMAP_A):
            hist[i] = 0.0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                var rr = (r + dr + rows) % rows
                var cc = (c + dc + cols) % cols
                hist[_gridctx_sym(data[rr * cols + cc])] += 1.0

    # Bin score z -> soft bins phi over meta-learned centres mu (stable softmax).
    @staticmethod
    def _bins(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        z: Float32,
        mut phi: InlineArray[Float32, GRIDCMAP_B],
    ):
        var temp = slow[GRIDCMAP_TEMP_OFF]
        var mx = Float32(-1.0e30)
        for j in range(GRIDCMAP_B):
            var d = z - slow[GRIDCMAP_MU_OFF + j]
            var lg = -temp * d * d
            phi[j] = lg
            if lg > mx:
                mx = lg
        var s = Float32(0.0)
        for j in range(GRIDCMAP_B):
            var ev = exp(phi[j] - mx)
            phi[j] = ev
            s += ev
        var invs = 1.0 / (s + 1e-9)
        for j in range(GRIDCMAP_B):
            phi[j] = phi[j] * invs

    # Meta-learned scoring -> soft colour-selection a (written into state[0..A-1]).
    # For each colour c the per-demo-cell pairs (count_c, output) yield three
    # features; a = softmax(tau * (w.feat + b)).
    @staticmethod
    def _select(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # Overall output moments (shared across colours).
        var n = 0
        var osum = Float32(0.0)
        var osq = Float32(0.0)
        for d in range(len(demos)):
            ref pg = demos[d].output_grid
            for i in range(pg.rows * pg.cols):
                var o = pg.data[i]
                osum += o
                osq = fma(o, o, osq)
                n += 1
        var invn = 1.0 / Float32(n)
        var omean = osum * invn
        var ovar = osq * invn - omean * omean + 1e-6

        var score = InlineArray[Float32, GRIDCMAP_A](fill=0.0)
        var smax = Float32(-1.0e30)
        for cc in range(GRIDCMAP_A):
            # Accumulate per-count-bin output stats for colour cc, plus cross term.
            var bn = InlineArray[Float32, GRIDCMAP_B](fill=0.0)
            var bs = InlineArray[Float32, GRIDCMAP_B](fill=0.0)
            var bq = InlineArray[Float32, GRIDCMAP_B](fill=0.0)
            var cnt_sum = Float32(0.0)
            var cnt_out = Float32(0.0)  # sum count*out (for correlation)
            var cnt_sq = Float32(0.0)
            for d in range(len(demos)):
                ref pair = demos[d]
                var rows = pair.input_grid.rows
                var cols = pair.input_grid.cols
                for r in range(rows):
                    for c in range(cols):
                        var hist = InlineArray[Float32, GRIDCMAP_A](fill=0.0)
                        GridCountMapSelfModMemory._hist(
                            pair.input_grid.data, rows, cols, r, c, hist
                        )
                        var k = Int(hist[cc])
                        if k >= GRIDCMAP_B:
                            k = GRIDCMAP_B - 1
                        var o = pair.output_grid.data[r * cols + c]
                        bn[k] += 1.0
                        bs[k] += o
                        bq[k] = fma(o, o, bq[k])
                        var kf = hist[cc]
                        cnt_sum += kf
                        cnt_out = fma(kf, o, cnt_out)
                        cnt_sq = fma(kf, kf, cnt_sq)
            # Feature 1: variance reduction (within-bin) -> functional dependence.
            var within = Float32(0.0)
            for j in range(GRIDCMAP_B):
                if bn[j] > 0.5:
                    var m = bs[j] / bn[j]
                    within += bq[j] - bn[j] * m * m
            within = within * invn
            var reduction = (ovar - within) / ovar  # in ~[0,1]
            # Feature 2: |linear correlation| of count_cc with output.
            var cmean = cnt_sum * invn
            var cvar = cnt_sq * invn - cmean * cmean + 1e-6
            var cov = cnt_out * invn - cmean * omean
            var corr = cov / sqrt(cvar * ovar)
            if corr < 0.0:
                corr = -corr
            # Feature 3: mean count (scale ~ [0,1]).
            var meanc = cmean / Float32(GRIDCMAP_NBRS)
            var sc = slow[GRIDCMAP_BSCORE_OFF]
            sc = fma(slow[GRIDCMAP_W_OFF + 0], reduction, sc)
            sc = fma(slow[GRIDCMAP_W_OFF + 1], corr, sc)
            sc = fma(slow[GRIDCMAP_W_OFF + 2], meanc, sc)
            score[cc] = sc
            if sc > smax:
                smax = sc
        # a = softmax(tau * score)
        var tau = slow[GRIDCMAP_TAU_OFF]
        var ssum = Float32(0.0)
        for cc in range(GRIDCMAP_A):
            var e = exp(tau * (score[cc] - smax))
            state[cc] = e
            ssum += e
        var invs = 1.0 / (ssum + 1e-9)
        for cc in range(GRIDCMAP_A):
            state[cc] = state[cc] * invs

    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # 1. Meta-learned scoring picks the predicate colour (soft), -> state[0..A-1].
        GridCountMapSelfModMemory._select(slow, demos, state)
        # 2. Init the value table spread over the demo output range (distinct bins).
        var vmin = demos[0].output_grid.data[0]
        var vmax = vmin
        for d in range(len(demos)):
            ref pg = demos[d].output_grid
            for i in range(pg.rows * pg.cols):
                var v = pg.data[i]
                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v
        for j in range(GRIDCMAP_B):
            var f = Float32(j) / Float32(GRIDCMAP_B - 1)
            state[GRIDCMAP_V_OFF + j] = vmin + (vmax - vmin) * f
        # 3. Delta-write the value table keyed by the soft count-bin.
        var eta_v = _sigmoid(slow[GRIDCMAP_EV_OFF])
        for _epoch in range(GRIDCMAP_EPOCHS):
            for d in range(len(demos)):
                ref pair = demos[d]
                var rows = pair.input_grid.rows
                var cols = pair.input_grid.cols
                for r in range(rows):
                    for c in range(cols):
                        var hist = InlineArray[Float32, GRIDCMAP_A](fill=0.0)
                        GridCountMapSelfModMemory._hist(
                            pair.input_grid.data, rows, cols, r, c, hist
                        )
                        var z = Float32(0.0)
                        for cc in range(GRIDCMAP_A):
                            z = fma(state[cc], hist[cc], z)
                        var phi = InlineArray[Float32, GRIDCMAP_B](fill=0.0)
                        GridCountMapSelfModMemory._bins(slow, z, phi)
                        var pred = Float32(0.0)
                        for j in range(GRIDCMAP_B):
                            pred = fma(phi[j], state[GRIDCMAP_V_OFF + j], pred)
                        var e = pair.output_grid.data[r * cols + c] - pred
                        for j in range(GRIDCMAP_B):
                            state[GRIDCMAP_V_OFF + j] = fma(
                                eta_v * e, phi[j], state[GRIDCMAP_V_OFF + j]
                            )

    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var rows = inp.rows
        var cols = inp.cols
        for r in range(rows):
            for c in range(cols):
                var hist = InlineArray[Float32, GRIDCMAP_A](fill=0.0)
                GridCountMapSelfModMemory._hist(
                    inp.data, rows, cols, r, c, hist
                )
                var z = Float32(0.0)
                for cc in range(GRIDCMAP_A):
                    z = fma(state[cc], hist[cc], z)
                var phi = InlineArray[Float32, GRIDCMAP_B](fill=0.0)
                GridCountMapSelfModMemory._bins(slow, z, phi)
                var pred = Float32(0.0)
                for j in range(GRIDCMAP_B):
                    pred = fma(phi[j], state[GRIDCMAP_V_OFF + j], pred)
                dst[r * cols + c] = pred
