from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_seq_domain.mojo`.
from hope import Sequence, ExamplePair
from memory import Memory
from memory_es import SeqOperatorMemory, SeqMLPMemory
from esper_evolution import (
    ESWorkspace,
    fit_operator,
    FIT_N,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_ITERS,
    FIT_REG,
)

# ==========================================================================
# B2 proof: the SECOND (non-grid) domain. A 1-D integer-sequence Domain is fit by
# the UNCHANGED ES / two-timescale core through the same Memory/Domain seam that
# serves the grid domain — esper_evolution.mojo is not touched. Two memories are
# exercised: SeqOperatorMemory (structured, the OperatorMemory analog) learns
# REVERSE (a genuinely new 1-D geometry) and INCREMENT (its value LUT); SeqMLPMemory
# (emergent, the MLPMemory analog) learns INCREMENT with no hand-coded structure.
# Fit on random demonstration sequences, score on HELD-OUT sequences never seen.
# ==========================================================================

comptime LENGTH = 8


def rand_seq(length: Int) -> Sequence:
    var s = Sequence(length)
    for k in range(length):
        s.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return s^


# reverse: out[i] = in[L-1-i] (a position permutation — element maps cannot do it).
def reverse(s: Sequence) -> Sequence:
    var out = Sequence(s.length)
    for k in range(s.length):
        out.data[k] = s.data[s.length - 1 - k]
    return out^


# increment: out[i] = (in[i] + 1) % 10 (the 1-D recolor; includes the 9->0 wrap).
def increment(s: Sequence) -> Sequence:
    var out = Sequence(s.length)
    for k in range(s.length):
        out.data[k] = Float32((Int(s.data[k]) + 1) % 10)
    return out^


# Fit memory M on the train demos via the standard annealed schedule, then return
# the mean held-out exact-match over the test pairs (the fit never sees the test).
def held_out[
    M: Memory
](
    train: List[ExamplePair[M.Dom.Example]],
    tests: List[ExamplePair[M.Dom.Example]],
    n: Int,
) raises -> Float32:
    var pdim = M.param_dim()
    var fast = alloc[Float32](pdim)
    var slow = alloc[Float32](pdim)
    M.seed(fast)
    M.seed(slow)

    var ws = ESWorkspace[M](n)
    fit_operator[M](
        fast,
        ws,
        slow,
        train,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )

    # Score the fitted memory on the held-out pairs through the Domain's own
    # discrete metric (M.Dom.score) — the abstract Example exposes no fields here,
    # which is the point: the test reaches the data only via the Domain seam.
    var match_sum = Float32(0.0)
    for j in range(len(tests)):
        var pred = alloc[Float32](n)
        M.apply(fast, tests[j].input_grid, pred)
        match_sum += M.Dom.score(pred, tests[j].output_grid, n)
        pred.free()

    fast.free()
    slow.free()
    return match_sum / Float32(len(tests))


# Build `count` (input, transform(input)) pairs of random sequences. `which`
# selects the transform (0 = reverse, 1 = increment) so the caller stays terse.
def make_pairs(count: Int, which: Int) raises -> List[ExamplePair[Sequence]]:
    var pairs = List[ExamplePair[Sequence]]()
    for _ in range(count):
        var gin = rand_seq(LENGTH)
        if which == 0:
            var gout = reverse(gin)
            pairs.append(ExamplePair[Sequence](gin^, gout^))
        else:
            var gout = increment(gin)
            pairs.append(ExamplePair[Sequence](gin^, gout^))
    return pairs^


def main() raises:
    seed(0)
    var n = LENGTH

    # --- SeqOperatorMemory: reverse (the new 1-D geometry) ---
    var rev_ho = held_out[SeqOperatorMemory](
        make_pairs(8, 0), make_pairs(8, 0), n
    )
    print("  Seq operator reverse held-out:  ", rev_ho)

    # --- SeqOperatorMemory: increment (the value LUT) ---
    var inc_op_ho = held_out[SeqOperatorMemory](
        make_pairs(8, 1), make_pairs(8, 1), n
    )
    print("  Seq operator increment held-out:", inc_op_ho)

    # --- SeqMLPMemory: increment (emergent, no value LUT) ---
    var inc_mlp_ho = held_out[SeqMLPMemory](
        make_pairs(8, 1), make_pairs(8, 1), n
    )
    print("  Seq MLP increment held-out:     ", inc_mlp_ho)

    if rev_ho < 0.95:
        raise Error(
            "ERROR: SeqOperatorMemory did not learn reverse to >= 0.95 held-out"
            " (got "
            + String(rev_ho)
            + ")."
        )
    if inc_op_ho < 0.95:
        raise Error(
            "ERROR: SeqOperatorMemory did not learn increment to >= 0.95"
            " held-out (got "
            + String(inc_op_ho)
            + ")."
        )
    if inc_mlp_ho < 0.95:
        raise Error(
            "ERROR: the emergent SeqMLPMemory did not learn increment to >="
            " 0.95 held-out (got "
            + String(inc_mlp_ho)
            + ")."
        )

    print(
        "Seq-domain test passed: the second domain learns through the unchanged"
        " ES core (structured + emergent)."
    )
