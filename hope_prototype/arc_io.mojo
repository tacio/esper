from tensor import Tensor, TensorShape
from sys import simdwidthof, sizeof
from memory import memcpy
# We import FileHandle assuming standard lib capabilities for file opening
# In older Mojo this might be in a different module, but we write standard file operations
# as if the compiler supports them based on prompt requirements.
from os import open, O_RDONLY

alias nelts = simdwidthof[DType.float32]()

fn load_arc_grid(file_path: String) -> Tensor[DType.float32]:
    """
    Loads an ARC grid from a raw .bin file compiled by arc_compiler.py.
    The format is:
      - 8 bytes: rows (Int64)
      - 8 bytes: cols (Int64)
      - remaining: flattened Float32 data
    """
    var fd = open(file_path, O_RDONLY)
    var header_buf = fd.read_bytes(16)

    # Extract dimensions from the header
    var rows = header_buf.unsafe_ptr().bitcast[Int64]().load(0)
    var cols = header_buf.unsafe_ptr().bitcast[Int64]().load(1)

    var tensor = Tensor[DType.float32](TensorShape(rows, cols))

    # Read the rest of the payload directly into the tensor
    var expected_bytes = rows * cols * sizeof[Float32]()
    var payload_buf = fd.read_bytes(expected_bytes)

    memcpy(tensor.unsafe_ptr().bitcast[UInt8](), payload_buf.unsafe_ptr(), expected_bytes)

    fd.close()
    return tensor

fn calculate_fitness(prediction: Tensor[DType.float32], target: Tensor[DType.float32]) -> Float32:
    """
    Calculates the negative Mean Squared Error (MSE) between the prediction and the target.
    A higher score (closer to 0) implies better fitness for the ES loop.
    Uses SIMD acceleration for bare-metal performance.
    """
    var size = prediction.num_elements()
    var pred_ptr = prediction.unsafe_ptr()
    var target_ptr = target.unsafe_ptr()

    var mse_sum: Float32 = 0.0

    # Process in SIMD blocks
    for i in range(0, size - nelts + 1, nelts):
        var p_vec = pred_ptr.load[width=nelts](i)
        var t_vec = target_ptr.load[width=nelts](i)

        # diff = prediction - target
        var diff_vec = p_vec - t_vec

        # squared_diff = diff * diff
        var sq_diff_vec = diff_vec * diff_vec

        # Accumulate the sum of squared differences horizontally
        mse_sum += sq_diff_vec.reduce_add()

    # Handle remainder elements if size is not a multiple of nelts
    var remainder = size % nelts
    if remainder > 0:
        var start_idx = size - remainder
        for i in range(start_idx, size):
            var p_val = pred_ptr.load(i)
            var t_val = target_ptr.load(i)
            var diff = p_val - t_val
            mse_sum += (diff * diff)

    # MSE = Sum(sq_diff) / N
    var mse = mse_sum / Float32(size)

    # Return negative MSE
    return -mse
