from tensor import Tensor, TensorShape
from sys import simdwidthof, sizeof
from memory import memcpy, UnsafePointer

alias nelts = simdwidthof[DType.float32]()

fn load_arc_grid(file_path: String) raises -> Tensor[DType.float32]:
    """
    Loads an ARC grid from a raw .bin file compiled by arc_compiler.py.
    The format is:
      - 8 bytes: rows (Int)
      - 8 bytes: cols (Int)
      - remaining: flattened Float32 data
    """
    # Use standard built-in file open
    var f = open(file_path, "r")
    var data = f.read_bytes()
    f.close()

    var data_ptr = data.unsafe_ptr()

    # Extract dimensions from the header
    var rows = data_ptr.bitcast[Int]().load(0)
    var cols = data_ptr.bitcast[Int]().load(1)

    var tensor = Tensor[DType.float32](TensorShape(rows, cols))

    # Read the rest of the payload directly into the tensor
    var expected_bytes = rows * cols * sizeof[Float32]()

    # Offset by 16 bytes for the two Int header blocks
    var payload_ptr = data_ptr.offset(16)

    memcpy(tensor.unsafe_ptr().bitcast[UInt8](), payload_ptr, expected_bytes)

    return tensor

fn calculate_fitness(pred_ptr: UnsafePointer[Float32], target_ptr: UnsafePointer[Float32], size: Int) -> Float32:
    """
    Calculates the negative Mean Squared Error (MSE) between the prediction and the target.
    A higher score (closer to 0) implies better fitness for the ES loop.
    Uses SIMD acceleration for bare-metal performance.
    """
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
