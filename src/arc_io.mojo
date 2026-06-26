from std.sys import simd_width_of, size_of
from std.memory import memcpy, UnsafePointer
from hope import ArcGrid

comptime nelts = simd_width_of[DType.float32]()

# 16-byte header: two little-endian Int64 (rows, cols), then the flattened
# Float32 payload. Produced by src/arc_compiler.py.
comptime HEADER_BYTES = 16


def load_arc_grid(file_path: String) raises -> ArcGrid:
    """Load an ARC grid from a raw .bin file compiled by arc_compiler.py.

    The header is validated against the payload length so a truncated or
    malformed file raises instead of triggering an out-of-bounds copy.
    """
    var f = open(file_path, "r")
    var data = f.read_bytes()
    f.close()

    if len(data) < HEADER_BYTES:
        raise Error("ARC bin file too small to contain a header")

    var header = data.unsafe_ptr().bitcast[Int64]()
    var rows = Int(header[0])
    var cols = Int(header[1])
    if rows < 0 or cols < 0:
        raise Error("ARC bin header has negative dimensions")

    var expected_bytes = rows * cols * size_of[Float32]()
    if len(data) < HEADER_BYTES + expected_bytes:
        raise Error("ARC bin payload is shorter than the header advertises")

    var grid = ArcGrid(rows, cols)
    var payload = data.unsafe_ptr() + HEADER_BYTES
    memcpy(dest=grid.data.bitcast[UInt8](), src=payload, count=expected_bytes)
    return grid^


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
