from esper_evolution import ESWorkspace, evolve_fast_weights
from memory import UnsafePointer
from tensor import Tensor, TensorShape

fn main():
    var size = 100
    var N = 10
    var alpha: Float32 = 0.01
    var sigma: Float32 = 0.1
    var fast_weights = UnsafePointer[Float32].alloc(size)
    for i in range(size):
        fast_weights.store(i, 0.0)

    var workspace = ESWorkspace(size)

    # Create a mock target tensor to resolve the compilation error
    var target = Tensor[DType.float32](TensorShape(10, 10))
    for i in range(100):
        target[i] = 0.0

    evolve_fast_weights(fast_weights, workspace, target, N, alpha, sigma)

    fast_weights.free()
    print("Success")
