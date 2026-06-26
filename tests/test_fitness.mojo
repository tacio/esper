# Verifies calculate_fitness (negative MSE) against hand-computed values,
# including the size < SIMD-width path (scalar remainder only).
from arc_io import calculate_fitness
from std.memory import alloc


def approx(a: Float32, b: Float32) -> Bool:
    var d = a - b
    if d < 0:
        d = -d
    return d < 1e-5


def main() raises:
    var n = 4
    var pred = alloc[Float32](n)
    var target = alloc[Float32](n)

    # Identical buffers -> MSE 0 -> fitness 0.
    for i in range(n):
        pred[i] = Float32(i)
        target[i] = Float32(i)
    if not approx(calculate_fitness(pred, target, n), 0.0):
        raise Error("ERROR: identical buffers should score 0 fitness.")

    # Constant error of 2 everywhere -> MSE 4 -> fitness -4.
    for i in range(n):
        pred[i] = target[i] + 2.0
    if not approx(calculate_fitness(pred, target, n), -4.0):
        raise Error("ERROR: expected fitness -4.0 for constant error of 2.")

    pred.free()
    target.free()
    print("calculate_fitness tests passed.")
