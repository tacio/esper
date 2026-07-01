from std.memory import UnsafePointer
from std.math import tanh, fma, floor, round, exp, sin, sqrt
from std.collections import List, InlineArray
from hope import (
    ArcGrid,
    ArcTaskPair,
    ExamplePair,
    Sequence,
    OP_DIM,
    COLOR_OFF,
    COLOR_DIM,
    seed_identity_operator,
    apply_operator,
)
from arc_io import Domain, GridDomain, SeqDomain

# Per-group ES step scale for the structured operator (a diagonal preconditioner).
# The colour-LUT parameters are normalized to ~unit scale but with tight 1/9
# spacing, so they need a smaller step than the affine. This used to be a
# hardcoded branch in ESWorkspace; it now belongs to the operator memory that
# owns the colour group (a different memory has a different/uniform scale).
comptime COLOR_SCALE = Float32(0.6)


# ==========================================
# Memory abstraction (Phase B)
# ==========================================
# A Memory is what the ES fits in-context: a parameter vector of fixed length
# (`param_dim`, grid-size-independent), a `seed` for the prior/init, an `apply`
# that runs the memory on a Domain Example (writing a flat prediction buffer),
# and a per-parameter ES preconditioner `fill_scale`. The associated `Dom` ties
# the memory to its Domain (Mojo traits can't take parameters, so the domain is
# an associated type reached as `M.Dom`). The ES/two-timescale core in
# esper_evolution.mojo is generic `[M: Memory]` over this trait — static dispatch,
# zero overhead. Growing one general Memory until it subsumes the structured
# operator is the whole point of Phase B; there is deliberately NO runtime
# memory-selector (that would be a DSL over memories).
trait Memory:
    comptime Dom: Domain

    @staticmethod
    def param_dim() -> Int:
        ...

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        ...

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        ...

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: Self.Dom.Example,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...


# The structured affine + colour-LUT operator, wrapped as the first Memory.
# Pure delegation to the existing hope.mojo entry points — this is the
# zero-behavior-change instance that keeps the whole suite green through the new
# generic path. (The operator stays the Phase-A baseline; emergent memories grow
# alongside it and are measured on the subsets they can express.)
struct OperatorMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return OP_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        seed_identity_operator(weights)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # The colour group gets the smaller step (diagonal preconditioner).
        for c in range(COLOR_DIM):
            scale[COLOR_OFF + c] = COLOR_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        apply_operator(weights, inp.data, dst, inp.rows, inp.cols)


# ==========================================
# First emergent memory: a per-cell MLP (Phase B / B1)
# ==========================================
# The first training-wheel removal: a generic per-cell function approximator with
# NO hand-coded colour structure. Over a K=1 receptive field (the centre cell) it
# is a 1->H->1 MLP: x = in[r,c]/9; z = W2 . tanh(W1*x + b1) + b2; and the output is
# SQUASHED to the valid colour range, out[r,c] = 9 * (tanh(z)+1)/2. The squash is
# essential: unlike the operator (a bounded gather of input values), a raw linear
# MLP output is unbounded, so a large W2 explodes the MSE and the ES diverges;
# bounding the output to [0,9] keeps the fitness landscape sane under the shared
# annealed schedule (the operator gets the same [0,9]-bounded property for free).
# The colour mapping (e.g. recolor) is *learned* as the output layer over a fixed
# tanh basis tiling [0,1] — emergent, not a LUT. Same-shape, continuous (smooth ES
# landscape; exact_match rounds only at scoring), scalar-per-cell like the operator
# gather. Cannot express global geometry (flip/transpose) — that needs a global
# addressing memory (B3); the MLP targets the local/colour subset.
comptime MLP_HIDDEN = 16
comptime MLP_W1_OFF = 0
comptime MLP_B1_OFF = MLP_HIDDEN
comptime MLP_W2_OFF = 2 * MLP_HIDDEN
comptime MLP_B2_OFF = 3 * MLP_HIDDEN
comptime MLP_DIM = 3 * MLP_HIDDEN + 1
# The output layer (W2, b2) moves freely; the basis (W1, b1) is nudged gently so
# the seeded tiling of [0,1] stays a stable feature map (a diagonal preconditioner,
# the MLP analogue of the operator's COLOR_SCALE).
comptime MLP_BASIS_SCALE = Float32(0.2)
# Output squash to EXACTLY the valid colour range [0, 9]. Bounding the output
# keeps the MSE (and the ES) stable; squashing to [0,9] specifically means a
# SATURATED output lands on a valid extreme colour (0 or 9), which rounds
# correctly — a wider range would saturate to invalid colours (e.g. -2/11) that
# never match (diagnosed via scratch/probe_mlp). Extremes need only out<=0.5 or
# >=8.5 (well inside tanh saturation), so the exact-match tolerance covers them.
comptime MLP_OUT_LO = Float32(0.0)
comptime MLP_OUT_HI = Float32(9.0)


# The per-element MLP forward (shared by MLPMemory over a grid and SeqMLPMemory
# over a sequence — both are purely per-cell, so the math lives once here): map a
# raw token value through x = v/9; z = W2 . tanh(W1*x + b1) + b2; squash to [0,9].
def _mlp_cell(
    weights: UnsafePointer[Float32, MutAnyOrigin], x_raw: Float32
) -> Float32:
    var x = x_raw / 9.0
    var z = weights[MLP_B2_OFF]
    for h in range(MLP_HIDDEN):
        var a = tanh(fma(weights[MLP_W1_OFF + h], x, weights[MLP_B1_OFF + h]))
        z = fma(weights[MLP_W2_OFF + h], a, z)
    # Squash to [MLP_OUT_LO, MLP_OUT_HI] (bounded output -> bounded MSE -> stable
    # ES; the [0,9] range means a saturated output lands on a valid extreme
    # colour, which rounds correctly under exact_match's tolerance).
    return MLP_OUT_LO + (MLP_OUT_HI - MLP_OUT_LO) * 0.5 * (tanh(z) + 1.0)


struct MLPMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return MLP_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Seed a fixed tanh basis that tiles the colour range. Each hidden unit
        # is a STEEP ramp (W1=20): it transitions over ~1/W1 in x, which must be
        # narrower than the 1/9 spacing between adjacent colours, else the basis
        # can't separate them — so the basis is near piecewise-constant, a soft
        # lookup table. Unit h transitions (W1*x + b1 = 0) at x = center_h, with
        # the centers tiling a range slightly WIDER than [0,1] (the valid
        # x = colour/9 span) so the edge colours (0 and 9) sit INSIDE the basis
        # with ramps on both sides (else they fall at the tiling boundary with no
        # features to shape them). The output layer starts at zero (W2 = b2 = 0);
        # the ES fits W2/b2 to interpolate the colour map over this basis (a
        # robust, ~linear-in-W2 fit). The 9->0 recolor wrap fits for free: colour
        # 9 saturates the squash low, which is exactly 0.
        comptime LO = Float32(-0.2)
        comptime HI = Float32(1.2)
        var denom = Float32(MLP_HIDDEN - 1)
        for h in range(MLP_HIDDEN):
            var center = LO + (HI - LO) * (Float32(h) / denom)
            weights[MLP_W1_OFF + h] = 20.0
            weights[MLP_B1_OFF + h] = -20.0 * center
            weights[MLP_W2_OFF + h] = 0.0
        weights[MLP_B2_OFF] = 0.0

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # Keep the basis (W1, b1) gentle; let the output layer (W2, b2) move.
        for h in range(MLP_HIDDEN):
            scale[MLP_W1_OFF + h] = MLP_BASIS_SCALE
            scale[MLP_B1_OFF + h] = MLP_BASIS_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(inp.rows * inp.cols):
            dst[i] = _mlp_cell(weights, inp.data[i])


# ==========================================
# Second domain (Phase B / B2): sequence memories
# ==========================================
# These two memories prove the Domain seam carries cross-domain: both conform to
# the SAME Memory trait with Dom = SeqDomain (a non-grid domain) and are fit by the
# UNCHANGED ES core. SeqOperatorMemory is the structured 1-D analog of
# OperatorMemory (a position affine + value LUT); SeqMLPMemory is the emergent
# per-element analog of MLPMemory (it reuses the exact same MLP layout + _mlp_cell).

# 1-D structured operator layout (grid-size-independent, the analog of OP_DIM):
# a 2-param centered affine on position (scale a, translation t) + a 10-entry value
# LUT over tokens 0..9, normalized /9 to keep it at the affine's ~unit scale.
comptime SEQ_COORD_DIM = 2
comptime SEQ_VALUE_DIM = 10
comptime SEQ_DIM = SEQ_COORD_DIM + SEQ_VALUE_DIM
comptime SEQ_A_OFF = 0
comptime SEQ_T_OFF = 1
comptime SEQ_VALUE_OFF = SEQ_COORD_DIM
# The value group needs a smaller ES step than the affine (tight 1/9 spacing) — a
# diagonal preconditioner, exactly the 1-D analog of the operator's COLOR_SCALE.
comptime SEQ_VALUE_SCALE = Float32(0.6)


# Value-LUT lookup (the 1-D analog of _color_of): round to the nearest token, clamp
# to [0,9], read the (normalized) LUT entry and rescale to the 0..9 range.
def _seq_value_of(
    weights: UnsafePointer[Float32, MutAnyOrigin], x: Float32
) -> Float32:
    var idx = Int(round(x))
    if idx < 0:
        idx = 0
    elif idx > SEQ_VALUE_DIM - 1:
        idx = SEQ_VALUE_DIM - 1
    return Float32(SEQ_VALUE_DIM - 1) * weights[SEQ_VALUE_OFF + idx]


# Read a sequence cell with zero-fill out of bounds (the 1-D analog of
# _cell_or_zero) — the search can sample OOB but the optima never do.
def _seq_cell_or_zero(
    data: UnsafePointer[Float32, MutAnyOrigin], length: Int, i: Int
) -> Float32:
    if i < 0 or i >= length:
        return 0.0
    return data[i]


# The structured sequence operator. Per output position p (over `length`), centered
# cp = p - (L-1)/2; source sp = a*cp + (L-1)/2 + t. VALUE-then-gather: the two
# integer source cells are mapped through the value LUT BEFORE the 1-D linear
# interpolation (the 1-D case of the grid's bilinear colour-then-gather — it
# decouples the value fit from the affine's precision, and the linear blend keeps a
# smooth ES landscape). Output stays continuous; exact_match rounds only at scoring.
# Integer params reproduce reverse (a=-1, t=0), shift-k (a=1, t=k) and increment
# (value LUT (v+1)%10) exactly.
struct SeqOperatorMemory(Memory):
    comptime Dom = SeqDomain

    @staticmethod
    def param_dim() -> Int:
        return SEQ_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        weights[SEQ_A_OFF] = 1.0  # identity position map
        weights[SEQ_T_OFF] = 0.0
        for v in range(SEQ_VALUE_DIM):
            weights[SEQ_VALUE_OFF + v] = Float32(v) / Float32(SEQ_VALUE_DIM - 1)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # The value group gets the smaller step (diagonal preconditioner).
        for v in range(SEQ_VALUE_DIM):
            scale[SEQ_VALUE_OFF + v] = SEQ_VALUE_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: Sequence,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var a = weights[SEQ_A_OFF]
        var t = weights[SEQ_T_OFF]
        var center = Float32(inp.length - 1) * Float32(0.5)
        for p in range(inp.length):
            var cp = Float32(p) - center
            var sp = fma(a, cp, center + t)
            var i0 = Int(floor(sp))
            var f = sp - Float32(i0)
            var v0 = _seq_value_of(
                weights, _seq_cell_or_zero(inp.data, inp.length, i0)
            )
            var v1 = _seq_value_of(
                weights, _seq_cell_or_zero(inp.data, inp.length, i0 + 1)
            )
            dst[p] = fma(f, v1 - v0, v0)


# The emergent per-element sequence memory: the SAME 1->H->1 tanh MLP as MLPMemory
# (it reuses MLP_DIM/seed/fill_scale and the shared _mlp_cell forward verbatim),
# applied over the sequence instead of the grid. Expresses element-wise maps only
# (e.g. increment) — NO hand-coded value structure; cannot do reverse/shift (a
# position permutation; that is B3's emergent global addressing). This shows the
# emergent path transfers to the new domain unchanged.
struct SeqMLPMemory(Memory):
    comptime Dom = SeqDomain

    @staticmethod
    def param_dim() -> Int:
        return MLP_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        MLPMemory.seed(weights)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        MLPMemory.fill_scale(scale, n)

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: Sequence,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(inp.length):
            dst[i] = _mlp_cell(weights, inp.data[i])


# ==========================================
# Emergent global addressing (Phase B / B3): attention gather (geometry)
# ==========================================
# The per-cell MLP (B1) is LOCAL — it cannot express geometry, because flip/
# transpose are GLOBAL coordinate permutations (out[i] = in[perm(i)]). OperatorMemory
# does geometry with a HAND-CODED affine that computes one source address and gathers
# exactly there. B3 re-earns geometry through an emergent GLOBAL READ instead: a
# learned position attention. Each output cell i (centered coord v_i) reads from ALL
# input cells j (centered coord v_j) weighted by a softmax over the coordinate
# similarity -beta*||M*v_i + t - v_j||^2, where M is a learned 2x2 projection, t a
# learned translation, and beta a learned temperature. As beta grows the softmax ->
# one-hot at the input cell nearest M*v_i + t, so an integer M reproduces a
# permutation EXACTLY (flip_h = [[1,0],[0,-1]], flip_v = [[-1,0],[0,1]], transpose =
# [[0,1],[1,0]]); a soft beta keeps a smooth ES gradient in M (the bilinear analog).
# The mechanism — a global similarity read, not a single hand-coded address — is the
# substrate B4's self-modifying memory builds on; the residual linear coord
# projection is what later milestones dissolve. param_dim is fixed (7), grid-size-
# independent. Like apply_operator this is an inherently gather/reduction kernel, so
# it is scalar-per-cell; the SIMD/FMA hot path (the weight-space ES update) is
# untouched and generic. Geometry-only — colour stays MLPMemory's job; one emergent
# memory covering BOTH geometry and colour is a future step (a single ES fit over the
# coupled 56-D memory was not honestly learnable without hand-staging the fit — see
# the journal's B3 entry; that retirement of OperatorMemory routes through B4).
comptime ATTN_M_OFF = 0  # 2x2 coordinate projection [[m00,m01],[m10,m11]]
comptime ATTN_T_OFF = 4  # translation (t_r, t_c)
comptime ATTN_BETA_OFF = 6  # temperature (raw; beta = raw^2 keeps it >= 0 w/o overflow)
comptime ATTN_DIM = 7
# beta = ATTN_BETA_SEED^2 ~ 2: a soft ~1-cell peak so identity reads mostly self and
# the gradient in M is non-zero (beta -> 0 gives a flat, M-independent mean; beta ->
# inf gives a flat plateau — moderate beta is where moving M moves the soft peak).
comptime ATTN_BETA_SEED = Float32(1.4)
comptime ATTN_BETA_SCALE = Float32(1.0)


struct AttnGatherMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return ATTN_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Identity addressing: M = I, t = 0 -> q_i = v_i -> each cell reads (mostly)
        # itself, so an unfit memory is the identity and the ES departs from it.
        weights[ATTN_M_OFF + 0] = 1.0
        weights[ATTN_M_OFF + 1] = 0.0
        weights[ATTN_M_OFF + 2] = 0.0
        weights[ATTN_M_OFF + 3] = 1.0
        weights[ATTN_T_OFF + 0] = 0.0
        weights[ATTN_T_OFF + 1] = 0.0
        weights[ATTN_BETA_OFF] = ATTN_BETA_SEED

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # Temperature gets its own step (diagonal preconditioner) so it can sharpen
        # at a different rate than the geometry params.
        scale[ATTN_BETA_OFF] = ATTN_BETA_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var m00 = weights[ATTN_M_OFF + 0]
        var m01 = weights[ATTN_M_OFF + 1]
        var m10 = weights[ATTN_M_OFF + 2]
        var m11 = weights[ATTN_M_OFF + 3]
        var t_r = weights[ATTN_T_OFF + 0]
        var t_c = weights[ATTN_T_OFF + 1]
        var beta_raw = weights[ATTN_BETA_OFF]
        var beta = beta_raw * beta_raw  # >= 0, no exp overflow risk
        var rows = inp.rows
        var cols = inp.cols
        var cr = Float32(rows - 1) * Float32(0.5)
        var cc = Float32(cols - 1) * Float32(0.5)

        for r in range(rows):
            var vr = Float32(r) - cr
            for c in range(cols):
                var vc = Float32(c) - cc
                # Read target q = M*v + t (where in coord space to gather from).
                var qr = fma(m00, vr, fma(m01, vc, t_r))
                var qc = fma(m10, vr, fma(m11, vc, t_c))

                # Pass 1: max score over all input cells (numerical stability — the
                # subtracted max makes the largest exponent 0, so exp never overflows).
                var max_score = Float32(-1.0e30)
                for rj in range(rows):
                    var dvr = qr - (Float32(rj) - cr)
                    var dvr2 = dvr * dvr
                    for cj in range(cols):
                        var dvc = qc - (Float32(cj) - cc)
                        var score = -beta * (dvr2 + dvc * dvc)
                        if score > max_score:
                            max_score = score

                # Pass 2: softmax-weighted gather (streaming, no per-cell buffer).
                var z = Float32(0.0)
                var s = Float32(0.0)
                for rj in range(rows):
                    var dvr = qr - (Float32(rj) - cr)
                    var dvr2 = dvr * dvr
                    for cj in range(cols):
                        var dvc = qc - (Float32(cj) - cc)
                        var score = -beta * (dvr2 + dvc * dvc)
                        var w = exp(score - max_score)
                        z += w
                        s = fma(w, inp.data[rj * cols + cj], s)
                dst[r * cols + c] = s / z


# ==========================================
# Self-modifying memory (Phase B / B4): an in-context self-WRITE rule
# ==========================================
# The lesson from B3 was that the derivative-free ES can't fit coupled/high-dim
# FAST weight spaces. HOPE's self-modifying memory (NL §8) is the fix: instead of
# the ES blindly searching the fast weights, the memory runs its OWN update rule
# that WRITES its fast state from the demonstrations in a single forward pass (the
# ES/meta-learner then fits only the small SLOW rule params). This is the first,
# minimal instance: an associative colour memory that builds the recolor map from
# the demos by a Hebbian self-write, then reads it on the held-out input — one pass,
# no per-task ES fit.
#
# CHECKPOINT 1 (this struct) validates the MECHANISM with FIXED projections (no
# meta-learning yet): the key is the integer colour itself (one-hot addressing), the
# value is the demo's output colour, the gate is 1, and the read is count-normalised.
# `adapt` accumulates state from the demo (in,out) cells; `apply` reads it. If this
# does not generalise held-out, the whole approach is wrong (fail-fast). Checkpoint 2
# replaces the fixed one-hot projection with meta-learned slow params.
#
# State layout (STATE_DIM = 2*COLOR_DIM): [0:10] = running sum of out-colour per
# in-colour, [10:20] = per-in-colour counts (so the read averages -> the map).
comptime SELFMOD_STATE_DIM = 2 * COLOR_DIM


struct RecolorSelfWrite:
    @staticmethod
    def state_dim() -> Int:
        return SELFMOD_STATE_DIM

    @staticmethod
    def _colour_index(v: Float32) -> Int:
        var c = Int(round(v))
        if c < 0:
            return 0
        elif c > COLOR_DIM - 1:
            return COLOR_DIM - 1
        return c

    # Inner self-write: build the colour map from the demonstrations in ONE pass.
    @staticmethod
    def adapt(
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(SELFMOD_STATE_DIM):
            state[i] = 0.0
        for d in range(len(demos)):
            ref pair = demos[d]
            var n = pair.input_grid.rows * pair.input_grid.cols
            for k in range(n):
                var ci = RecolorSelfWrite._colour_index(pair.input_grid.data[k])
                # Hebbian write with the (one-hot) key ci: accumulate value + count.
                state[ci] += pair.output_grid.data[k]
                state[COLOR_DIM + ci] += 1.0

    # Read the written map for each input cell (count-normalised; identity fallback
    # for a colour the demos never showed).
    @staticmethod
    def apply(
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for k in range(inp.rows * inp.cols):
            var ci = RecolorSelfWrite._colour_index(inp.data[k])
            var cnt = state[COLOR_DIM + ci]
            if cnt > 0.0:
                dst[k] = state[ci] / cnt
            else:
                dst[k] = Float32(ci)


# ==========================================
# Self-modifying memory trait (Phase B / B4)
# ==========================================
# A self-modifying memory fits a small SLOW parameter vector (meta-learned) and, at
# adaptation time, WRITES its own fast `state` from the demonstrations via its own
# update rule (`adapt`) — the ES never searches the fast state (the reframe of the
# B3 ES-bottleneck lesson: move the search off the fast weights). `apply` reads the
# written state for a test input. Generic over its Domain (`Dom`), so the meta-fit
# core in esper_evolution.mojo is a single generic `meta_fit_selfmod[M: SelfModMemory]`.
trait SelfModMemory:
    comptime Dom: Domain

    @staticmethod
    def slow_dim() -> Int:
        ...

    @staticmethod
    def state_dim() -> Int:
        ...

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        ...

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        ...

    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ExamplePair[Self.Dom.Example]],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...

    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: Self.Dom.Example,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...


# ==========================================
# Self-modifying memory with a META-LEARNED associative read (Phase B / B4, ckpt 2)
# ==========================================
# Checkpoint 1 proved the self-WRITE generalises with a FIXED (one-hot) read. This
# makes the read EMERGENT: the memory still writes a per-colour value table from the
# demos in one pass (`adapt`), but the read is a learned softmax ATTENTION over
# meta-learned colour embeddings E (+ a temperature beta) — pred(q) = sum over the
# present colours c of softmax_c(beta * E[q]·E[c]) * value[c]. Exact retrieval needs
# the embeddings SEPARABLE and beta sharp, so the meta-learner must DISCOVER a
# working addressing from a generic seed (not the hand-set one-hot). The SLOW params
# (E, beta) are meta-learned ONCE across a recolor family; a fresh task then adapts
# in a single forward pass. This is the genuine self-modifying claim — the read
# projections are learned, not given — and the ES only ever fits the small slow
# vector (the fast state is WRITTEN by `adapt`, never ES-searched), which is the
# whole point of the B3-lesson reframe.
comptime SELFMOD_DK = 8
comptime SELFMOD_SLOW_DIM = COLOR_DIM * SELFMOD_DK + 1  # E[10 x Dk] + beta
comptime SELFMOD_BETA_OFF = COLOR_DIM * SELFMOD_DK


struct RecolorSelfModMemory(SelfModMemory):
    comptime Dom = GridDomain

    @staticmethod
    def slow_dim() -> Int:
        return SELFMOD_SLOW_DIM

    @staticmethod
    def state_dim() -> Int:
        return SELFMOD_STATE_DIM

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        # Generic, weakly-separated embeddings (NOT the one-hot answer) so the
        # meta-learner has real work — it must sharpen them into a usable addressing.
        for c in range(COLOR_DIM):
            for d in range(SELFMOD_DK):
                slow[c * SELFMOD_DK + d] = (
                    sin(Float32(c + 1) * Float32(d + 1) * 0.7) * 0.5
                )
        slow[SELFMOD_BETA_OFF] = 2.0

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    @staticmethod
    def _colour_index(v: Float32) -> Int:
        var c = Int(round(v))
        if c < 0:
            return 0
        elif c > COLOR_DIM - 1:
            return COLOR_DIM - 1
        return c

    # Self-write: build the per-colour value table from the demos (one pass).
    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ArcTaskPair],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(SELFMOD_STATE_DIM):
            state[i] = 0.0
        for d in range(len(demos)):
            ref pair = demos[d]
            var n = pair.input_grid.rows * pair.input_grid.cols
            for k in range(n):
                var ci = RecolorSelfModMemory._colour_index(
                    pair.input_grid.data[k]
                )
                state[ci] += pair.output_grid.data[k]
                state[COLOR_DIM + ci] += 1.0
        # Normalise sums into per-colour values.
        for c in range(COLOR_DIM):
            if state[COLOR_DIM + c] > 0.0:
                state[c] = state[c] / state[COLOR_DIM + c]

    # Learned associative read: softmax attention over present colours, keyed by the
    # meta-learned embeddings.
    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var beta = slow[SELFMOD_BETA_OFF]
        for k in range(inp.rows * inp.cols):
            var q = RecolorSelfModMemory._colour_index(inp.data[k])
            var q_off = q * SELFMOD_DK

            # Pass 1: max similarity over present colours (exp stability).
            var max_score = Float32(-1.0e30)
            for c in range(COLOR_DIM):
                if state[COLOR_DIM + c] > 0.0:
                    var dot = Float32(0.0)
                    for d in range(SELFMOD_DK):
                        dot = fma(
                            slow[q_off + d], slow[c * SELFMOD_DK + d], dot
                        )
                    var score = beta * dot
                    if score > max_score:
                        max_score = score

            # Pass 2: softmax-weighted read of the per-colour values.
            var z = Float32(0.0)
            var acc = Float32(0.0)
            for c in range(COLOR_DIM):
                if state[COLOR_DIM + c] > 0.0:
                    var dot = Float32(0.0)
                    for d in range(SELFMOD_DK):
                        dot = fma(
                            slow[q_off + d], slow[c * SELFMOD_DK + d], dot
                        )
                    var w = exp(beta * dot - max_score)
                    z += w
                    acc = fma(w, state[c], acc)
            if z > 0.0:
                dst[k] = acc / z
            else:
                dst[k] = Float32(q)


# ==========================================
# Fuller self-modifying block (Phase B / B4): gated delta-rule self-write
# ==========================================
# B4-step-1's write was a fixed per-colour ACCUMULATE. The fuller HOPE block (NL §8)
# makes the write a learned, self-generated update rule: per input the memory
# generates its own key / value / learning-rate eta / forget-gate alpha and runs its
# OWN internal optimization — a GATED DELTA RULE, which is one SGD step on the
# associative loss ||S·k - v||^2 with retention:  S <- (1-alpha)*S + eta*(v - S·k)*k.
# Proven on a SEQUENCE local-context map out[i] = f(in[i], in[i-1]) (circular left
# neighbour): a neighbour-dependent rule the per-cell MLP / colour-LUT CANNOT express
# — the memory must KEY ON THE CONTEXT (the pair) and learn the association in-context.
#
# Symbol alphabet A (small, so the A*A contexts are covered by the demos); the value
# is the target symbol. Circular neighbour avoids edge-token coverage gaps.
comptime SEQCTX_A = 5


def _seq_sym(v: Float32) -> Int:
    var c = Int(round(v))
    if c < 0:
        return 0
    elif c > SEQCTX_A - 1:
        return SEQCTX_A - 1
    return c


# CHECKPOINT A — validate the gated-delta mechanism with FIXED one-hot context keys
# (Dk = A*A), eta = 1, alpha = 0. With a one-hot key the delta step touches only the
# context's slot: S[ctx] += (v - S[ctx]) -> S[ctx] = v (exact per-context table). This
# validates the write+read plumbing; checkpoint B replaces the one-hot key with
# learned compact embeddings + self-generated gates.
comptime DELTA_ONEHOT_STATE = SEQCTX_A * SEQCTX_A


struct DeltaSelfWrite:
    @staticmethod
    def state_dim() -> Int:
        return DELTA_ONEHOT_STATE

    @staticmethod
    def adapt(
        demos: List[ExamplePair[Sequence]],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(DELTA_ONEHOT_STATE):
            state[i] = 0.0
        for d in range(len(demos)):
            ref pair = demos[d]
            var length = pair.input_grid.length
            for i in range(length):
                var c0 = _seq_sym(pair.input_grid.data[i])
                var c1 = _seq_sym(
                    pair.input_grid.data[(i - 1 + length) % length]
                )
                var ctx = c0 * SEQCTX_A + c1
                # Gated delta rule, one-hot key, eta=1, alpha=0 -> S[ctx] = v.
                state[ctx] += pair.output_grid.data[i] - state[ctx]

    @staticmethod
    def apply(
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: Sequence,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var length = inp.length
        for i in range(length):
            var c0 = _seq_sym(inp.data[i])
            var c1 = _seq_sym(inp.data[(i - 1 + length) % length])
            dst[i] = state[c0 * SEQCTX_A + c1]


# CHECKPOINT B — the fuller self-modifying block (emergent, meta-learned). The read
# is a bilinear map of a self-generated key: k = OUTER PRODUCT of learned per-symbol
# embeddings, k[a*De+b] = E[c0][a]·E[c1][b]. The outer product (not a concat, which
# would only give an additive g(c0)+h(c1) read) lets a linear S·k represent an
# ARBITRARY local rule f(c0,c1). The learning-rate eta and forget-gate alpha are
# self-generated per input (sigmoid of a learned projection of k). `adapt` runs the
# gated delta rule S <- (1-alpha)S + eta(v - S·k)k over the demo cells (the internal
# associative objective); `apply` reads S·k. The SLOW params (embeddings E + the two
# gate projections) are meta-learned ONCE across a family of random rules; a fresh
# rule then adapts in a single forward pass. Dk = De^2 (>= A*A) so the bilinear read
# can span all A*A contexts once E is separable — which the meta-learner discovers.
comptime DELTA_DE = 5
comptime DELTA_DK = DELTA_DE * DELTA_DE
comptime DELTA_E_OFF = 0
comptime DELTA_WETA_OFF = SEQCTX_A * DELTA_DE
comptime DELTA_BETA_OFF = DELTA_WETA_OFF + DELTA_DK
comptime DELTA_WALPHA_OFF = DELTA_BETA_OFF + 1
comptime DELTA_BALPHA_OFF = DELTA_WALPHA_OFF + DELTA_DK
comptime DELTA_SLOW_DIM = DELTA_BALPHA_OFF + 1


def _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))


struct DeltaSelfModMemory(SelfModMemory):
    comptime Dom = SeqDomain

    @staticmethod
    def slow_dim() -> Int:
        return DELTA_SLOW_DIM

    @staticmethod
    def state_dim() -> Int:
        return DELTA_DK

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        # Generic, weakly-separated embeddings (NOT one-hot) so the meta-learner must
        # discover a separable addressing; small gate projections; eta starts ~0.5
        # (b_eta = 0), alpha starts small (~0.12, b_alpha = -2) so early writes stick.
        for c in range(SEQCTX_A):
            for d in range(DELTA_DE):
                slow[DELTA_E_OFF + c * DELTA_DE + d] = (
                    sin(Float32(c + 1) * Float32(d + 1) * 0.7) * 0.5
                )
        for j in range(DELTA_DK):
            slow[DELTA_WETA_OFF + j] = 0.1 * sin(Float32(j + 1) * 1.3)
            slow[DELTA_WALPHA_OFF + j] = 0.1 * sin(Float32(j + 1) * 2.1)
        slow[DELTA_BETA_OFF] = 0.0
        slow[DELTA_BALPHA_OFF] = -2.0

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    # Build the outer-product context key into `k` (stack-resident, no heap alloc),
    # NORMALISED to unit norm. Normalisation is essential for a stable delta rule: it
    # bounds the effective step size eta*|k|^2 <= eta < 2 regardless of the embedding
    # scale, so the ES cannot wander the embeddings into a region where the recurrent
    # write diverges (an unnormalised key let the meta-fit blow up and score BELOW the
    # random seed). It is a no-op when the embeddings are already one-hot.
    @staticmethod
    def _key(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        c0: Int,
        c1: Int,
        mut k: InlineArray[Float32, DELTA_DK],
    ):
        var sumsq = Float32(0.0)
        for a in range(DELTA_DE):
            var e0 = slow[DELTA_E_OFF + c0 * DELTA_DE + a]
            for b in range(DELTA_DE):
                var val = e0 * slow[DELTA_E_OFF + c1 * DELTA_DE + b]
                k[a * DELTA_DE + b] = val
                sumsq = fma(val, val, sumsq)
        var inv = 1.0 / (sqrt(sumsq) + 1e-6)
        for j in range(DELTA_DK):
            k[j] = k[j] * inv

    # Self-write: gated delta rule over the demo cells (the internal objective).
    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ExamplePair[Sequence]],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for j in range(DELTA_DK):
            state[j] = 0.0
        for d in range(len(demos)):
            ref pair = demos[d]
            var length = pair.input_grid.length
            for i in range(length):
                var c0 = _seq_sym(pair.input_grid.data[i])
                var c1 = _seq_sym(
                    pair.input_grid.data[(i - 1 + length) % length]
                )
                var k = InlineArray[Float32, DELTA_DK](fill=0.0)
                DeltaSelfModMemory._key(slow, c0, c1, k)
                # pred = S·k; self-generated eta/alpha from a projection of k.
                var pred = Float32(0.0)
                var geta = slow[DELTA_BETA_OFF]
                var galpha = slow[DELTA_BALPHA_OFF]
                for j in range(DELTA_DK):
                    pred = fma(state[j], k[j], pred)
                    geta = fma(slow[DELTA_WETA_OFF + j], k[j], geta)
                    galpha = fma(slow[DELTA_WALPHA_OFF + j], k[j], galpha)
                var eta = _sigmoid(geta)
                var alpha = _sigmoid(galpha)
                var e = pair.output_grid.data[i] - pred
                for j in range(DELTA_DK):
                    state[j] = (1.0 - alpha) * state[j] + eta * e * k[j]

    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: Sequence,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var length = inp.length
        for i in range(length):
            var c0 = _seq_sym(inp.data[i])
            var c1 = _seq_sym(inp.data[(i - 1 + length) % length])
            var k = InlineArray[Float32, DELTA_DK](fill=0.0)
            DeltaSelfModMemory._key(slow, c0, c1, k)
            var pred = Float32(0.0)
            for j in range(DELTA_DK):
                pred = fma(state[j], k[j], pred)
            dst[i] = pred


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
