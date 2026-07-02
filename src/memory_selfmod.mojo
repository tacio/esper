# The self-modifying memory family, core mechanisms (Phase B / B4): memories
# that WRITE their own fast state from the demonstrations via their own update
# rule (the ES meta-learns only the small slow vector). RecolorSelfWrite is the
# fixed-projection checkpoint; RecolorSelfModMemory adds the meta-learned
# associative read; DeltaSelfModMemory is the fuller gated delta-rule block on
# the sequence domain. The 2-D grid family lives in memory_selfmod_grid.mojo.
from std.memory import UnsafePointer
from std.math import fma, round, exp, sin, sqrt
from std.collections import List
from hope import ArcGrid, ArcTaskPair, Sequence, ExamplePair, COLOR_DIM
from arc_io import GridDomain, SeqDomain
from memory import SelfModMemory

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
