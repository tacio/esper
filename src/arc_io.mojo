from std.sys import simd_width_of, size_of
from std.memory import memcpy, UnsafePointer
from std.math import round
from std.collections import List
from hope import ArcGrid, ArcTaskPair, ArcTask

comptime nelts = simd_width_of[DType.float32]()

# 16-byte grid-block header: two little-endian Int64 (rows, cols), then the
# flattened Float32 payload. Produced by src/arc_compiler.py.
comptime HEADER_BYTES = 16


# Read one grid-block starting at `offset` in a raw byte buffer, advancing
# `offset` past it. Shared by the single-grid `.bin` loader and the `.task`
# bundle loader; validates the header against the buffer length so a truncated
# or malformed file raises instead of triggering an out-of-bounds copy.
def _read_grid_block(
    ptr: UnsafePointer[UInt8, MutAnyOrigin],
    data_len: Int,
    mut offset: Int,
) raises -> ArcGrid:
    if offset + HEADER_BYTES > data_len:
        raise Error("ARC bin: grid header runs past end of file")
    var header = (ptr + offset).bitcast[Int64]()
    var rows = Int(header[0])
    var cols = Int(header[1])
    if rows < 0 or cols < 0:
        raise Error("ARC bin: grid header has negative dimensions")

    var payload_bytes = rows * cols * size_of[Float32]()
    var payload_off = offset + HEADER_BYTES
    if payload_off + payload_bytes > data_len:
        raise Error(
            "ARC bin: grid payload is shorter than the header advertises"
        )

    var grid = ArcGrid(rows, cols)
    memcpy(
        dest=grid.data.bitcast[UInt8](),
        src=ptr + payload_off,
        count=payload_bytes,
    )
    offset = payload_off + payload_bytes
    return grid^


def load_arc_grid(file_path: String) raises -> ArcGrid:
    """Load a single ARC grid from a raw .bin file compiled by arc_compiler.py.
    """
    var f = open(file_path, "r")
    var data = f.read_bytes()
    f.close()
    var offset = 0
    return _read_grid_block(data.unsafe_ptr(), len(data), offset)


def load_arc_task(file_path: String) raises -> ArcTask:
    """Load a whole task from a binary `.task` bundle written by
    `arc_compiler._save_task`: two little-endian int64 counts (n_train, n_test)
    then 2*n_train + 2*n_test grid-blocks (train_in, train_out, ..., test_in,
    test_out). Single read, no globbing or text parsing — the shell globs the
    `.task` files into the driver's argv.
    """
    var f = open(file_path, "r")
    var data = f.read_bytes()
    f.close()
    var data_len = len(data)
    if data_len < HEADER_BYTES:
        raise Error("ARC task bundle too small to contain its header")

    var ptr = data.unsafe_ptr()
    var counts = ptr.bitcast[Int64]()
    var n_train = Int(counts[0])
    var n_test = Int(counts[1])
    if n_train < 0 or n_test < 0:
        raise Error("ARC task bundle has negative counts")

    var offset = HEADER_BYTES
    var train = List[ArcTaskPair]()
    for _ in range(n_train):
        var gin = _read_grid_block(ptr, data_len, offset)
        var gout = _read_grid_block(ptr, data_len, offset)
        train.append(ArcTaskPair(gin^, gout^))

    var test = List[ArcTaskPair]()
    for _ in range(n_test):
        var gin = _read_grid_block(ptr, data_len, offset)
        var gout = _read_grid_block(ptr, data_len, offset)
        test.append(ArcTaskPair(gin^, gout^))

    return ArcTask(train^, test^)


def calculate_fitness(
    pred_ptr: UnsafePointer[Float32, MutAnyOrigin],
    target_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
) -> Float32:
    """Negative Mean Squared Error between prediction and target.

    Higher (closer to 0) is fitter — this is the ES fitness signal. SIMD main
    loop + scalar remainder for bare-metal throughput.
    """
    var mse_sum = Float32(0.0)

    for i in range(0, size - nelts + 1, nelts):
        var diff = pred_ptr.load[width=nelts](i) - target_ptr.load[width=nelts](
            i
        )
        mse_sum += (diff * diff).reduce_add()

    var remainder = size % nelts
    if remainder > 0:
        for i in range(size - remainder, size):
            var diff = pred_ptr[i] - target_ptr[i]
            mse_sum += diff * diff

    return -(mse_sum / Float32(size))


def exact_match(
    pred_ptr: UnsafePointer[Float32, MutAnyOrigin],
    target_ptr: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
) -> Float32:
    """Discrete objective reward: fraction of cells that match exactly.

    ARC grids hold integer colors (0-9) while the engine evolves continuous
    weights, so each prediction is rounded to the nearest integer before
    comparison. This mirrors ARC's all-or-nothing per-cell scoring and
    complements the continuous negative-MSE signal from `calculate_fitness`.
    Same SIMD main-loop + scalar-remainder shape as the rest of the hot path.
    """
    var matches = Float32(0.0)

    for i in range(0, size - nelts + 1, nelts):
        var p_vec = round(pred_ptr.load[width=nelts](i))
        var t_vec = round(target_ptr.load[width=nelts](i))
        matches += p_vec.eq(t_vec).cast[DType.float32]().reduce_add()

    var remainder = size % nelts
    if remainder > 0:
        for i in range(size - remainder, size):
            if round(pred_ptr[i]) == round(target_ptr[i]):
                matches += 1.0

    return matches / Float32(size)
