# End-to-end driver: build an arena-hosted HopeNode whose fast weights are an
# OP_DIM operator vector, learn a transform (flip_h) in-context from demonstration
# pairs via the Evolution Strategy, then apply the learned operator to a held-out
# test input and report how well it generalizes.
# Run with: mojo run -I src src/main.mojo
from hope import (
    HopeArena,
    build_node,
    ArcGrid,
    ArcTaskPair,
    OP_DIM,
    seed_identity_operator,
)
from esper_evolution import forward_with_learning
from arc_io import exact_match
from std.collections import List
from std.random import seed, random_float64


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


def flip_h_grid(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            out.set(r, c, g.get(r, g.cols - 1 - c))
    return out^


def main() raises:
    seed(0)
    var arena = HopeArena(1 << 16)

    # Root expert: OP_DIM slow (prior) + OP_DIM fast (in-context memory) weights.
    # Both start at the identity operator: the prior anchors there, and the fast
    # weights depart from it as the ES fits the demonstrated transform.
    var node = build_node(arena, OP_DIM, OP_DIM)
    seed_identity_operator(node[].slow)
    seed_identity_operator(node[].fast)

    # Demonstrations: random grids paired with their horizontal flip.
    var rows = 4
    var cols = 4
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rows, cols)
        var gout = flip_h_grid(gin)
        demos.append(ArcTaskPair(gin^, gout^))

    # Held-out test input (never seen during the fit) and its true flip.
    var test = rand_grid(rows, cols)
    var truth = flip_h_grid(test)

    var result = forward_with_learning(node, demos, test)

    print("Esper forward pass complete (learned flip_h from demonstrations).")
    print(
        "  test  row 0 :",
        test.get(0, 0),
        test.get(0, 1),
        test.get(0, 2),
        test.get(0, 3),
    )
    print(
        "  result row 0:",
        result.get(0, 0),
        result.get(0, 1),
        result.get(0, 2),
        result.get(0, 3),
    )
    print(
        "  truth  row 0:",
        truth.get(0, 0),
        truth.get(0, 1),
        truth.get(0, 2),
        truth.get(0, 3),
    )

    var held_out = exact_match(result.data, truth.data, test.size())
    print("  held-out exact match:", held_out)
