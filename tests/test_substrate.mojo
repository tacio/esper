from std.memory import alloc
from grid_substrate import GridSubstrate, SUB_DIST_CAP


def check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error("FAIL: " + msg)


def main() raises:
    # 6x6: bg 0, a 2x2 block of colour 3 at (1,1)-(2,2), a lone 5 at (4,4).
    var rows = 6
    var cols = 6
    var q = alloc[Float32](rows * cols)
    for k in range(rows * cols):
        q[k] = 0.0
    q[1 * cols + 1] = 3.0
    q[1 * cols + 2] = 3.0
    q[2 * cols + 1] = 3.0
    q[2 * cols + 2] = 3.0
    q[4 * cols + 4] = 5.0

    var sub = GridSubstrate(q, rows, cols)
    check(sub.bg == 0, "bg plurality")
    check(sub.n_comp == 3, "component count (bg + block + dot)")
    check(sub.large_col == 3, "largest register")
    check(sub.small_col == 5, "smallest register")
    check(sub.uniq_col == -1, "unique register absent (two singletons)")
    check(sub.major_col == 3, "majority nonbg register")
    check(sub.anc_large_r == 1 and sub.anc_large_c == 1, "largest anchor")
    check(sub.anc_uniq_r == -1, "unique anchor absent")

    # nearest-nonbg: (0,0) is diagonal-adjacent to the block corner (1,1).
    check(sub.near_d[0] == 1 and sub.near_col[0] == 3, "nearest at (0,0)")
    # (5,0): chebyshev 3 to the block cell (2,1).
    check(sub.near_d[5 * cols + 0] == 3, "nearest distance at (5,0)")
    check(sub.near_col[5 * cols + 0] == 3, "nearest colour at (5,0)")
    # block cells are their own nearest.
    check(sub.near_d[1 * cols + 1] == 0, "nearest at a nonbg cell")

    # rays (view fetch): up at (3,1) sees the block; up at (0,1) sees nothing;
    # down at (0,4) sees the lone 5.
    var rf = sub.fetch(0, 3, 1)
    check(rf[0] == 1 and rf[1] == 3, "ray up hit")
    rf = sub.fetch(0, 0, 1)
    check(rf[0] == 0 and rf[1] == -1, "ray up miss")
    rf = sub.fetch(1, 0, 4)
    check(rf[0] == 1 and rf[1] == 5, "ray down hit")

    # nearest view rel = capped distance.
    rf = sub.fetch(4, 0, 0)
    check(rf[0] == 1 and rf[1] == 3, "nearest view at (0,0)")
    rf = sub.fetch(4, 5, 5)
    check(rf[0] == 1 and rf[1] == 5, "nearest view at (5,5)")

    # registers: largest at a block cell matches the centre (rel 2).
    rf = sub.fetch(5, 1, 1)
    check(rf[0] == 2 and rf[1] == 3, "largest register at block cell")
    rf = sub.fetch(7, 0, 0)
    check(rf[0] == 0 and rf[1] == -1, "unique register absent")

    # anchor large+ at (0,0) reads q(1,1) = 3.
    rf = sub.fetch(9, 0, 0)
    check(rf[0] == 1 and rf[1] == 3, "anchor large+ read")
    rf = sub.fetch(11, 0, 0)
    check(rf[0] == 0 and rf[1] == -1, "anchor uniq absent")

    # object-local h-mirror inside the block: (1,1) -> (1,2) = 3, rel 2.
    rf = sub.fetch(13, 1, 1)
    check(rf[0] == 2 and rf[1] == 3, "obj-local h mirror inside block")

    # second grid: two colour-3 singletons + one 5 => unique register = 5.
    var q2 = alloc[Float32](5 * 5)
    for k in range(25):
        q2[k] = 0.0
    q2[0] = 3.0
    q2[2 * 5 + 0] = 3.0
    q2[4 * 5 + 4] = 5.0
    var sub2 = GridSubstrate(q2, 5, 5)
    check(sub2.uniq_col == 5, "unique register present")
    check(sub2.anc_uniq_r == 4 and sub2.anc_uniq_c == 4, "unique anchor")
    check(sub2.large_col == 3, "largest tie keeps first")

    q.free()
    q2.free()
    print("PASS: grid substrate (components, registers, BFS, rays, fetch)")
