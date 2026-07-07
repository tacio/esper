# GPU/CPU fitness parity (rung G1). The GPU contract is: trajectories may
# diverge (float reduction order), fitness values may not — for the same
# weights and demos the batched device fitness must equal the CPU reference
# within reduction-order float noise. This is the level where CPU/GPU
# agreement is meaningful, so it is what the parity test pins: several weight
# vectors (identity, exact flip, a soft off-integer read, a far translation
# that exercises the window clamp) over several grid sizes (window-covered
# small grids AND real-ARC 30x30 where the window truncates), plus the
# same-shape mismatch penalty. Skips (comptime) on hosts without a GPU — the
# device path isn't even compiled there.
from std.sys import has_accelerator
from std.memory import UnsafePointer, alloc
from hope import ArcGrid, ArcTaskPair
from esper_evolution import fitness
from memory_es import AttnGatherMemory, ATTN_DIM, ATTN_BETA_SEED
from gpu_es import gpu_fitness_plain

comptime REG = Float32(0.0001)


def make_grid(rows: Int, cols: Int, salt: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for i in range(rows * cols):
        g.data[i] = Float32((i * 7 + salt * 3 + i * i % 5) % 10)
    return g^


def flip_h_of(g: ArcGrid) -> ArcGrid:
    var o = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            o.data[r * o.cols + c] = g.data[r * g.cols + (g.cols - 1 - c)]
    return o^


def check_pair(
    name: String,
    weights: UnsafePointer[Float32, MutAnyOrigin],
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    cap: Int,
) raises:
    var op_out = alloc[Float32](cap)
    var f_cpu = fitness[AttnGatherMemory](weights, slow, demos, op_out, REG)
    op_out.free()
    var f_gpu = gpu_fitness_plain(weights, slow, demos, cap, REG)
    var diff = f_cpu - f_gpu
    if diff < 0:
        diff = -diff
    var mag = f_cpu if f_cpu > 0 else -f_cpu
    if mag < 1.0:
        mag = 1.0
    print("  ", name, " cpu:", f_cpu, " gpu:", f_gpu, " |diff|:", diff)
    if diff > Float32(1.0e-4) * mag:
        raise Error("GPU/CPU fitness mismatch for " + name)


def run() raises:
    # Demo set spanning window-covered small grids and window-truncated 30x30.
    var demos = List[ArcTaskPair]()
    var cap = 30 * 30
    var sizes_r: List[Int] = [5, 9, 13, 30]
    var sizes_c: List[Int] = [7, 9, 30, 30]
    for d in range(len(sizes_r)):
        var g_in = make_grid(sizes_r[d], sizes_c[d], d)
        var g_out = flip_h_of(g_in)
        demos.append(ArcTaskPair(g_in^, g_out^))

    var slow = alloc[Float32](ATTN_DIM)
    AttnGatherMemory.seed(slow)
    var w = alloc[Float32](ATTN_DIM)

    # 1. Identity seed (soft temperature, anchor = 0).
    AttnGatherMemory.seed(w)
    check_pair("identity", w, slow, demos, cap)

    # 2. Exact flip_h (integer params, hard-ish read).
    AttnGatherMemory.seed(w)
    w[3] = -1.0
    check_pair("flip_h", w, slow, demos, cap)

    # 3. Soft off-integer read (wide softmax, every window cell contributes).
    w[0] = 0.83
    w[1] = 0.21
    w[2] = -0.14
    w[3] = 0.9
    w[4] = 1.3
    w[5] = -0.7
    w[6] = 0.55
    check_pair("soft", w, slow, demos, cap)

    # 4. Far translation — the window centre clamps at the grid edge.
    AttnGatherMemory.seed(w)
    w[4] = 25.0
    w[5] = -25.0
    check_pair("clamped", w, slow, demos, cap)

    # 5. Same-shape mismatch penalty (output area != input area).
    var mm = List[ArcTaskPair]()
    var m_in = make_grid(6, 6, 11)
    var m_out = make_grid(3, 3, 12)
    mm.append(ArcTaskPair(m_in^, m_out^))
    var g_in = make_grid(8, 8, 13)
    var g_out = flip_h_of(g_in)
    mm.append(ArcTaskPair(g_in^, g_out^))
    AttnGatherMemory.seed(w)
    check_pair("mismatch-penalty", w, slow, mm, cap)

    slow.free()
    w.free()
    print("GPU parity test passed: device fitness == CPU reference.")


def main() raises:
    comptime if not has_accelerator():
        print("SKIP: no accelerator available; GPU parity test not run")
        return
    else:
        run()
