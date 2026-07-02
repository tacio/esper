from std.memory import alloc, UnsafePointer
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_shape.mojo`.
from hope import ArcGrid, ArcTaskPair, OP_DIM, seed_identity_operator
from memory_es import OperatorMemory
from esper_evolution import fitness

# ==========================================================================
# Shape-handling guard test. The structured operator is same-shape (output dims
# == input dims), which covers the Phase-A expressible subset. A demo whose
# output area differs from its input area is inexpressible and must be handled
# GRACEFULLY: heavily penalized, never crashing on a mismatched-length compare.
# ==========================================================================


def main() raises:
    var weights = alloc[Float32](OP_DIM)
    var slow = alloc[Float32](OP_DIM)
    seed_identity_operator(weights)
    seed_identity_operator(slow)
    var op_output = alloc[Float32](64)

    # Same-shape identity demo: the identity operator reproduces it, so the
    # fitness (negative MSE) is ~0.
    var same = List[ArcTaskPair]()
    var a_in = ArcGrid(3, 3)
    var a_out = ArcGrid(3, 3)
    for k in range(9):
        a_in.data[k] = Float32(k % 10)
        a_out.data[k] = Float32(k % 10)
    same.append(ArcTaskPair(a_in^, a_out^))
    var f_same = fitness[OperatorMemory](
        weights, slow, same, op_output, Float32(0.0)
    )
    if f_same < -0.001:
        raise Error(
            "same-shape identity demo should score ~0, got " + String(f_same)
        )

    # Shape-changing demo: 3x3 input, 3x6 output. Inexpressible -> heavy penalty,
    # and no out-of-bounds read.
    var diff = List[ArcTaskPair]()
    var b_in = ArcGrid(3, 3)
    var b_out = ArcGrid(3, 6)
    for k in range(9):
        b_in.data[k] = Float32(k % 10)
    for k in range(18):
        b_out.data[k] = Float32(k % 10)
    diff.append(ArcTaskPair(b_in^, b_out^))
    var f_diff = fitness[OperatorMemory](
        weights, slow, diff, op_output, Float32(0.0)
    )
    if f_diff > -1.0e6:
        raise Error(
            "shape-changing demo should be heavily penalized, got "
            + String(f_diff)
        )

    weights.free()
    slow.free()
    op_output.free()
    print("Shape-handling test passed.")
