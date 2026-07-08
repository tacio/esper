# GPU environment smoke test: proves the toolchain can compile, launch, and
# read back a real GPU kernel. On machines without an accelerator (e.g. CI)
# `has_accelerator()` is comptime-False, so the device path is not even
# compiled and the test passes as a skip.
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import UnsafePointer
from std.sys import has_accelerator

comptime N = 1024
comptime BLOCK = 256


def vec_add(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
):
    var i = global_idx.x
    if i < N:
        c[i] = a[i] + b[i]


def run_gpu_smoke() raises:
    var ctx = DeviceContext()
    print("GPU device:", ctx.name())

    var a = ctx.enqueue_create_buffer[DType.float32](N)
    var b = ctx.enqueue_create_buffer[DType.float32](N)
    var c = ctx.enqueue_create_buffer[DType.float32](N)

    with a.map_to_host() as ah, b.map_to_host() as bh:
        for i in range(N):
            ah[i] = Float32(i)
            bh[i] = Float32(2 * i)

    ctx.enqueue_function[vec_add](
        a, b, c, grid_dim=(N + BLOCK - 1) // BLOCK, block_dim=BLOCK
    )
    ctx.synchronize()

    with c.map_to_host() as ch:
        for i in range(N):
            if ch[i] != Float32(3 * i):
                raise Error(
                    "GPU vec_add mismatch at "
                    + String(i)
                    + ": got "
                    + String(ch[i])
                    + ", expected "
                    + String(3 * i)
                )

    print("PASS: GPU kernel launch + readback correct over", N, "elements")


def main() raises:
    comptime if not has_accelerator():
        print("SKIP: no accelerator available; GPU smoke test not run")
        return
    else:
        run_gpu_smoke()
