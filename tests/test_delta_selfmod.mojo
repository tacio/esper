# suite-tier: full
#   Large-scale self-modifying meta-fit milestone proof (~64s). Deferred from the
#   `fast` tier — the meta_fit_selfmod core is already covered there by
#   test_selfmod_memory. See run_tests.sh / CLAUDE.md "Testing".
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_delta_selfmod.mojo`.
from hope import Sequence, ExamplePair, Task
from memory import (
    DeltaSelfWrite,
    DeltaSelfModMemory,
    SEQCTX_A,
    DELTA_ONEHOT_STATE,
    DELTA_SLOW_DIM,
    DELTA_DK,
)
from esper_evolution import meta_fit_selfmod
from arc_io import exact_match

# ==========================================================================
# B4 (fuller block) — a self-modifying memory with a GATED DELTA-RULE self-write,
# proven on a SEQUENCE local-context map out[i] = f(in[i], in[i-1]) (circular left
# neighbour). This rule is neighbour-dependent — a per-cell MLP / colour-LUT cannot
# express it; the memory must key on the (cell, neighbour) CONTEXT.
#
# Checkpoint A (this file, for now): validate the gated-delta mechanism with FIXED
# one-hot context keys (no ES). The delta write builds the per-context map from the
# demo cells in one pass; read on HELD-OUT sequences. Fail-fast gate.
# ==========================================================================

comptime A = SEQCTX_A
comptime LEN = 12
comptime SeqPair = ExamplePair[Sequence]
comptime SeqTask = Task[Sequence]


def rand_seq(length: Int) -> Sequence:
    var s = Sequence(length)
    for k in range(length):
        s.data[k] = Float32(Int(random_float64(0.0, Float64(A))))
    return s^


# A random local rule f: (c0, c1) -> out over the A*A contexts.
def rand_rule(f: UnsafePointer[Int, MutAnyOrigin]):
    for i in range(A * A):
        f[i] = Int(random_float64(0.0, Float64(A)))


# out[i] = f(in[i], in[i-1]) with a circular left neighbour.
def apply_rule(s: Sequence, f: UnsafePointer[Int, MutAnyOrigin]) -> Sequence:
    var o = Sequence(s.length)
    var length = s.length
    for i in range(length):
        var c0 = Int(s.data[i])
        var c1 = Int(s.data[(i - 1 + length) % length])
        o.data[i] = Float32(f[c0 * A + c1])
    return o^


def make_task(
    f: UnsafePointer[Int, MutAnyOrigin], n_train: Int, n_test: Int
) -> SeqTask:
    var train = List[SeqPair]()
    for _ in range(n_train):
        var gin = rand_seq(LEN)
        train.append(SeqPair(gin^, apply_rule(gin, f)))
    var test = List[SeqPair]()
    for _ in range(n_test):
        var gin = rand_seq(LEN)
        test.append(SeqPair(gin^, apply_rule(gin, f)))
    return SeqTask(train^, test^)


# A FRESH random rule, unseen sequences: one-pass adapt, then score held-out.
def fresh_heldout(slow: UnsafePointer[Float32, MutAnyOrigin]) raises -> Float32:
    var f = alloc[Int](A * A)
    rand_rule(f)
    var task = make_task(f, 8, 8)
    var state = alloc[Float32](DELTA_DK)
    DeltaSelfModMemory.adapt(slow, task.train, state)
    var ms = Float32(0.0)
    for j in range(len(task.test)):
        var pred = alloc[Float32](LEN)
        DeltaSelfModMemory.apply(slow, state, task.test[j].input_grid, pred)
        ms += exact_match(pred, task.test[j].output_grid.data, LEN)
        pred.free()
    state.free()
    f.free()
    return ms / Float32(len(task.test))


def main() raises:
    seed(0)

    var f = alloc[Int](A * A)
    rand_rule(f)

    var demos = List[SeqPair]()
    for _ in range(8):
        var gin = rand_seq(LEN)
        demos.append(SeqPair(gin^, apply_rule(gin, f)))

    var state = alloc[Float32](DELTA_ONEHOT_STATE)
    DeltaSelfWrite.adapt(demos, state)

    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var t = rand_seq(LEN)
        var truth = apply_rule(t, f)
        var pred = alloc[Float32](LEN)
        DeltaSelfWrite.apply(state, t, pred)
        match_sum += exact_match(pred, truth.data, LEN)
        pred.free()
    var held_out = match_sum / Float32(trials)
    print(
        "  ckptA gated-delta local-context held-out (one pass, one-hot key):",
        held_out,
    )

    state.free()
    f.free()

    if held_out < 0.95:
        raise Error(
            "ERROR: the gated-delta self-write did not learn the local-context"
            " map to >= 0.95 held-out in one pass (got "
            + String(held_out)
            + ")."
        )

    print(
        "Delta self-mod checkpoint A passed: the gated delta-rule wrote the"
        " context->value map from the demos in one pass and generalised."
    )

    # ---------- Checkpoint B: meta-learn the projections (the fuller block) ----------
    # Re-seed so ckpt B's RNG is independent of ckpt A (reproducible in isolation).
    seed(0)
    var slow = alloc[Float32](DELTA_SLOW_DIM)
    DeltaSelfModMemory.seed_slow(slow)

    var before = Float32(0.0)
    for _ in range(5):
        before += fresh_heldout(slow)
    before = before / 5.0
    print("  ckptB fresh held-out BEFORE meta-fit (generic seed):", before)

    var meta = List[SeqTask]()
    for _ in range(10):
        var f = alloc[Int](A * A)
        rand_rule(f)
        meta.append(make_task(f, 8, 4))
        f.free()
    meta_fit_selfmod[DeltaSelfModMemory](
        slow, meta, LEN, 128, 0.1, 0.003, 0.5, 0.01, 4500
    )

    var after = Float32(0.0)
    var n_fresh = 10
    for _ in range(n_fresh):
        after += fresh_heldout(slow)
    after = after / Float32(n_fresh)
    print(
        "  ckptB fresh held-out AFTER meta-fit (one-pass, 10 unseen rules):",
        after,
    )
    slow.free()

    if after < 0.95:
        raise Error(
            "ERROR: the meta-learned fuller self-mod block did not solve a"
            " fresh local-context rule to >= 0.95 held-out in one pass (got "
            + String(after)
            + ")."
        )
    if before >= 0.5:
        raise Error(
            "ERROR: the generic seed already generalised (before="
            + String(before)
            + ") — the meta-fit is not doing the work; emergence claim vacuous."
        )

    print(
        "Delta self-mod checkpoint B passed: the fuller block (self-generated"
        " key/eta/alpha + gated delta write) meta-learned its projections cold;"
        " a fresh arbitrary local-context rule adapts in ONE forward pass."
    )
