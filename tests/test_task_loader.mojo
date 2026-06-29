from arc_io import load_arc_task

# Run from the project root after run_tests.sh has generated build/sample.task:
#   mojo run -I src tests/test_task_loader.mojo
#
# ==========================================================================
# Task-bundle loader round-trip test. build/sample.task holds 2 train pairs and
# 1 test pair of 2x2 grids (each output is the horizontal flip of its input).
# The loader must recover the counts, dimensions, and cell values exactly.
# Fields are accessed by chaining (task.train[0].input_grid.get(...)) so the
# move-only ArcGrid is borrowed, never copied.
# ==========================================================================


def main() raises:
    var task = load_arc_task("build/sample.task")

    if len(task.train) != 2:
        raise Error("expected 2 train pairs, got " + String(len(task.train)))
    if len(task.test) != 1:
        raise Error("expected 1 test pair, got " + String(len(task.test)))

    # train[0]: input [[1,2],[3,4]] -> output (flip_h) [[2,1],[4,3]]
    if task.train[0].input_grid.rows != 2 or task.train[0].input_grid.cols != 2:
        raise Error("train[0] input dimensions wrong")
    if (
        task.train[0].input_grid.get(0, 0) != 1.0
        or task.train[0].input_grid.get(1, 1) != 4.0
    ):
        raise Error("train[0] input values wrong")
    if (
        task.train[0].output_grid.get(0, 0) != 2.0
        or task.train[0].output_grid.get(0, 1) != 1.0
    ):
        raise Error("train[0] output values wrong")

    # train[1]: input [[5,6],[7,8]]
    if task.train[1].input_grid.get(1, 0) != 7.0:
        raise Error("train[1] input values wrong")

    # test[0]: input [[9,0],[1,2]] -> output [[0,9],[2,1]]
    if (
        task.test[0].input_grid.get(0, 0) != 9.0
        or task.test[0].input_grid.get(1, 0) != 1.0
    ):
        raise Error("test[0] input values wrong")
    if task.test[0].output_grid.get(0, 0) != 0.0:
        raise Error("test[0] output values wrong")

    print("Task loader test passed.")
