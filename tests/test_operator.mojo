from std.memory import alloc, UnsafePointer

# Run from the project root with `mojo run -I src tests/test_operator.mojo`.
from hope import (
    OP_DIM,
    COORD_OFF,
    COLOR_OFF,
    COLOR_DIM,
    apply_operator,
    seed_identity_operator,
)

# ==========================================================================
# Structured-operator unit test.
#
# The learned grid->grid operator must, for hand-set weights, reproduce the
# canonical transforms exactly: identity, flip_h, transpose (square), and a
# recolor LUT (including the 9->0 wrap). These are the Phase-A expressible
# subset; every one is an exact special case of the affine + colour-LUT
# operator. Values are integer colours stored as float and the transforms are
# permutations / exact relabels, so exact float equality is a valid check.
# ==========================================================================


def check(
    label: String,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    exp_ptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) raises:
    for i in range(n):
        if out_ptr[i] != exp_ptr[i]:
            raise Error(
                "ERROR: "
                + label
                + " mismatch at cell "
                + String(i)
                + " (got "
                + String(out_ptr[i])
                + ", expected "
                + String(exp_ptr[i])
                + ")"
            )


def main() raises:
    var rows = 3
    var cols = 3
    var n = rows * cols

    # Input colours 1..9 (includes a 9 so the recolor wrap is exercised).
    var inp = alloc[Float32](n)
    for i in range(n):
        inp[i] = Float32((i + 1) % 10)

    var out = alloc[Float32](n)
    var exp = alloc[Float32](n)
    var weights = alloc[Float32](OP_DIM)

    # 1. Identity: out == in.
    seed_identity_operator(weights)
    apply_operator(weights, inp, out, rows, cols)
    check("identity", out, inp, n)

    # 2. flip_h: a3 = -1 reverses each row.
    seed_identity_operator(weights)
    weights[COORD_OFF + 3] = -1.0
    apply_operator(weights, inp, out, rows, cols)
    for r in range(rows):
        for c in range(cols):
            exp[r * cols + c] = inp[r * cols + (cols - 1 - c)]
    check("flip_h", out, exp, n)

    # 3. transpose: A = [[0,1],[1,0]] (square grid keeps shape).
    seed_identity_operator(weights)
    weights[COORD_OFF + 0] = 0.0
    weights[COORD_OFF + 1] = 1.0
    weights[COORD_OFF + 2] = 1.0
    weights[COORD_OFF + 3] = 0.0
    apply_operator(weights, inp, out, rows, cols)
    for r in range(rows):
        for c in range(cols):
            exp[r * cols + c] = inp[c * cols + r]
    check("transpose", out, exp, n)

    # 4. recolor: cmap[c] = (c + 1) % 10 (stored normalized, /9), identity geometry.
    seed_identity_operator(weights)
    for c in range(COLOR_DIM):
        weights[COLOR_OFF + c] = Float32((c + 1) % 10) / Float32(COLOR_DIM - 1)
    apply_operator(weights, inp, out, rows, cols)
    for i in range(n):
        exp[i] = Float32((Int(inp[i]) + 1) % 10)
    check("recolor", out, exp, n)

    inp.free()
    out.free()
    exp.free()
    weights.free()
    print("Operator test passed.")
