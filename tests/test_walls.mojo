from std.memory import alloc, UnsafePointer
from sandbox import (
    SB_ROWS,
    SB_COLS,
    SB_CELLS,
    SB_WALL,
    SB_WALLS_SHELVES,
    SB_WALLS_COLUMNS,
    SB_WALLS_ROOM,
    SB_WALLS_KINDS,
    BC_DIM,
    SandboxTask,
    add_wall_rect,
    gen_walls_layout,
    sandbox_step,
    sandbox_bc,
    sandbox_cell_key,
)
from empowerment import empowerment, EMP_SET_CAP

# ==========================================================================
# T-POC-1 step 1 (walls = world 2): mechanics proofs + the world-2 validation
# control. Everything here is exact and RNG-free — the numbers are pinned
# constants, not statistical claims.
#   1. Mechanics: a mid-air shelf holds a falling block; the avatar is
#      blocked by walls (and only walls); BC/cell keys count movable content
#      only (topology never leaks into the behaviour space — the cross-world
#      retrieval seam); the room layout keeps a door and the start cell.
#   2. The trap exhibit (categorical): a sealed 1-cell pocket collapses
#      exact 4-step empowerment to ~3.91 bits vs ~8.27 in the open field —
#      walls create optionality structure the open world provably lacks
#      (B-POC-2.5 measured its paint-flattened landscape at ~7.9 bits
#      everywhere; the only relief was board edges).
#   3. The field control (graded): over all standable positions, the columns
#      world widens the empowerment spread and digs below the open world's
#      minimum — topology differentiates the landscape, not just one cell.
# Gates are pinned with headroom below the exact measured values
# (open: mean 8.171 min 6.870 max 8.267; columns: min 6.443 spread 1.949;
# sealed pocket: 3.907).
# ==========================================================================

comptime EMP_H = 4


# Empowerment field stats over every standable (non-wall) avatar position.
def field(
    task: SandboxTask,
    lvl: UnsafePointer[Float32, MutAnyOrigin],
    seen: UnsafePointer[Int64, MutAnyOrigin],
    ticks: UnsafePointer[Int, MutAnyOrigin],
    mut mn: Float32,
    mut mx: Float32,
    mut mean: Float32,
):
    mn = Float32(1e9)
    mx = Float32(-1e9)
    var total = Float32(0.0)
    var n = 0
    for r in range(SB_ROWS):
        for c in range(SB_COLS):
            if task.grid[r * SB_COLS + c] < 0.0:
                continue
            var e = empowerment(
                task.grid, r, c, 1, task, EMP_H, lvl, seen, ticks
            )
            if e < mn:
                mn = e
            if e > mx:
                mx = e
            total += e
            n += 1
    mean = total / Float32(n)


def main() raises:
    # --- 1. Mechanics -----------------------------------------------------
    var t = SandboxTask()
    gen_walls_layout(t, SB_WALLS_SHELVES, 0)
    if t.grid[5 * SB_COLS + 8] != SB_WALL:
        raise Error("ERROR: expected wall at (5,8) in shelves variant 0.")
    t.grid[0 * SB_COLS + 8] = 3.0
    var r = 8
    var c = 8
    var brush = 1
    for _ in range(8):
        sandbox_step(t.grid, r, c, brush, 0, 1, 5)  # cycle brush: no move
    if t.grid[4 * SB_COLS + 8] != 3.0 or t.grid[6 * SB_COLS + 8] != 0.0:
        raise Error("ERROR: block did not settle ON the mid-air shelf.")

    r = 6
    c = 8
    sandbox_step(t.grid, r, c, brush, 0, 1, 0)  # up into (5,8) = wall
    if r != 6:
        raise Error("ERROR: avatar walked into a wall.")
    sandbox_step(t.grid, r, c, brush, 0, 1, 3)  # right into (6,9) = empty
    if c != 9:
        raise Error("ERROR: avatar blocked where there is no wall.")

    # BC + cell key must count movable content only: a walls-only grid reads
    # as empty (this keeps goal BCs comparable across worlds — the T-POC-1
    # cross-world retrieval seam).
    var t2 = SandboxTask()
    gen_walls_layout(t2, SB_WALLS_SHELVES, 3)
    var bc = alloc[Float32](BC_DIM)
    sandbox_bc(t2.grid, 0, 0, bc)
    var occ = Float32(0.0)
    for i in range(BC_DIM - 2):
        occ += bc[i]
    var empty_key = sandbox_cell_key(SandboxTask().grid, 0, 0)
    if occ != 0.0 or sandbox_cell_key(t2.grid, 0, 0) != empty_key:
        raise Error("ERROR: BC / cell key counted wall cells.")
    bc.free()

    # Every layout kind x a variant sweep keeps the start cell standable.
    for kind in range(SB_WALLS_KINDS):
        for variant in range(8):
            var tv = SandboxTask()
            gen_walls_layout(tv, kind, variant)
            if tv.grid[tv.start_r * SB_COLS + tv.start_c] != 0.0:
                raise Error("ERROR: start cell overwritten by a wall.")
    var t3 = SandboxTask()
    gen_walls_layout(t3, SB_WALLS_ROOM, 2)
    var door_cells = 0
    for rr in range(4, 13):
        if t3.grid[rr * SB_COLS + 11] == 0.0:
            door_cells += 1
    if door_cells < 1 or door_cells > 2:
        raise Error("ERROR: room door not 1-2 cells.")
    print("  walls mechanics: shelf/avatar/BC/layout gates OK")

    # --- 2. The trap exhibit (categorical) --------------------------------
    var lvl = alloc[Float32]((EMP_H + 1) * SB_CELLS)
    var seen = alloc[Int64](EMP_SET_CAP)
    var ticks = alloc[Int](1)
    ticks[0] = 0

    var tp = SandboxTask()
    add_wall_rect(tp, 1, 2, 1, 2)
    add_wall_rect(tp, 3, 2, 3, 2)
    add_wall_rect(tp, 2, 1, 2, 1)
    add_wall_rect(tp, 2, 3, 2, 3)
    var e_sealed = empowerment(tp.grid, 2, 2, 1, tp, EMP_H, lvl, seen, ticks)

    var open_t = SandboxTask()
    var open_mn = Float32(0.0)
    var open_mx = Float32(0.0)
    var open_mean = Float32(0.0)
    field(open_t, lvl, seen, ticks, open_mn, open_mx, open_mean)
    print(
        "  empowerment: sealed pocket",
        e_sealed,
        " open field min/mean/max",
        open_mn,
        open_mean,
        open_mx,
    )
    if e_sealed > 4.5 or e_sealed > open_mn - 2.0:
        raise Error(
            "ERROR: sealed pocket not categorically less empowered "
            "than the open world."
        )

    # --- 3. The field control (graded) ------------------------------------
    var cols_t = SandboxTask()
    gen_walls_layout(cols_t, SB_WALLS_COLUMNS, 0)
    var w_mn = Float32(0.0)
    var w_mx = Float32(0.0)
    var w_mean = Float32(0.0)
    field(cols_t, lvl, seen, ticks, w_mn, w_mx, w_mean)
    var open_spread = open_mx - open_mn
    var w_spread = w_mx - w_mn
    print(
        "  columns world min/mean/max",
        w_mn,
        w_mean,
        w_mx,
        " spread",
        w_spread,
        "(open",
        open_spread,
        ") enum ticks",
        ticks[0],
    )
    if w_mn > open_mn - 0.3:
        raise Error("ERROR: walls did not dig below the open-world minimum.")
    if w_spread < open_spread + 0.3:
        raise Error("ERROR: walls did not widen the empowerment spread.")

    lvl.free()
    seen.free()
    ticks.free()
    print("PASS: test_walls (mechanics + world-2 empowerment differentiation)")
