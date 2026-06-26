# Smoke test for the HopeArena allocator and the POD HopeNode.
# Imports the real definitions from src (run with `-I src`) rather than
# duplicating them, so the test cannot drift from the implementation.
from hope import HopeArena, HopeNode, build_node


def main() raises:
    # Raw bump allocation round-trips through arena memory.
    var arena = HopeArena(4096)
    var float_ptr = arena.alloc_node[Float32]()
    float_ptr[0] = 42.0
    if float_ptr[0] != 42.0:
        raise Error("ERROR: Memory read/write failure.")

    # A POD HopeNode lives entirely inside the arena (header + weight slices).
    var node = build_node(arena, 4, 8)
    if node[].slow_dim != 4 or node[].fast_dim != 8:
        raise Error("ERROR: node dimensions not initialised.")
    node[].fast[3] = 7.0
    if node[].fast[3] != 7.0:
        raise Error("ERROR: node fast-weight slice not writable.")
    if node[].num_children != 0:
        raise Error("ERROR: new node should have no children.")

    # Children are recorded as arena indices.
    node[].add_child(2)
    if node[].num_children != 1 or node[].children[0] != 2:
        raise Error("ERROR: add_child did not record the child index.")

    print("HopeArena + HopeNode tests passed.")
