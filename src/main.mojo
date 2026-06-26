# End-to-end driver: build an arena-hosted HopeNode, run the nested-learning
# forward pass over a demonstration, and report the fitness of the result.
# Run with: mojo run -I src src/main.mojo
from hope import (
    HopeArena,
    build_node,
    ArcGrid,
    ArcTaskPair,
    forward_with_learning,
    PRIM_SHIFT,
)
from arc_io import calculate_fitness
from std.collections import List


def main() raises:
    var arena = HopeArena(1 << 16)

    # Root expert: 8 slow (routing) weights, 8 fast (memory) weights.
    var node = build_node(arena, 8, 8)
    # Seed the fast weights so that, after the learning phase nudges them, the
    # primitive still shifts by (dx, dy) = (1, 0).
    node[].fast[0] = 2.0
    node[].fast[1] = 0.0

    # One demonstration pair (shift the input right by one column).
    var demo_in = ArcGrid(3, 3)
    var demo_out = ArcGrid(3, 3)
    for i in range(9):
        demo_in.data[i] = Float32(i)
    var demos = List[ArcTaskPair]()
    demos.append(ArcTaskPair(demo_in^, demo_out^))

    # Test input.
    var test = ArcGrid(3, 3)
    for i in range(9):
        test.data[i] = Float32(i)

    var result = forward_with_learning(node, demos, test, PRIM_SHIFT)

    print("Esper forward pass complete.")
    print(
        "  test input row 0 :", test.get(0, 0), test.get(0, 1), test.get(0, 2)
    )
    print(
        "  result    row 0 :",
        result.get(0, 0),
        result.get(0, 1),
        result.get(0, 2),
    )

    # Fitness of the result against the (unchanged) test input as a target.
    var fitness = calculate_fitness(result.data, test.data, test.size())
    print("  fitness vs input :", fitness)
