from std.memory import UnsafePointer
from std.collections import List
from hope import ExamplePair
from arc_io import Domain


# ==========================================
# Memory abstraction (Phase B)
# ==========================================
# A Memory is what the ES fits in-context: a parameter vector of fixed length
# (`param_dim`, grid-size-independent), a `seed` for the prior/init, an `apply`
# that runs the memory on a Domain Example (writing a flat prediction buffer),
# and a per-parameter ES preconditioner `fill_scale`. The associated `Dom` ties
# the memory to its Domain (Mojo traits can't take parameters, so the domain is
# an associated type reached as `M.Dom`). The ES/two-timescale core in
# esper_evolution.mojo is generic `[M: Memory]` over this trait — static dispatch,
# zero overhead. Growing one general Memory until it subsumes the structured
# operator is the whole point of Phase B; there is deliberately NO runtime
# memory-selector (that would be a DSL over memories).
trait Memory:
    comptime Dom: Domain

    @staticmethod
    def param_dim() -> Int:
        ...

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        ...

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        ...

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: Self.Dom.Example,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...


# ==========================================
# Self-modifying memory trait (Phase B / B4)
# ==========================================
# A self-modifying memory fits a small SLOW parameter vector (meta-learned) and, at
# adaptation time, WRITES its own fast `state` from the demonstrations via its own
# update rule (`adapt`) — the ES never searches the fast state (the reframe of the
# B3 ES-bottleneck lesson: move the search off the fast weights). `apply` reads the
# written state for a test input. Generic over its Domain (`Dom`), so the meta-fit
# core in esper_evolution.mojo is a single generic `meta_fit_selfmod[M: SelfModMemory]`.
trait SelfModMemory:
    comptime Dom: Domain

    @staticmethod
    def slow_dim() -> Int:
        ...

    @staticmethod
    def state_dim() -> Int:
        ...

    @staticmethod
    def seed_slow(slow: UnsafePointer[Float32, MutAnyOrigin]):
        ...

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        ...

    @staticmethod
    def adapt(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ExamplePair[Self.Dom.Example]],
        state: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...

    @staticmethod
    def apply(
        slow: UnsafePointer[Float32, MutAnyOrigin],
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: Self.Dom.Example,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...


# ==========================================
# Shape-changing memory trait (Vision A / Next #1 — the output-size seam)
# ==========================================
# Every Memory/SelfModMemory above is SAME-SHAPE: `apply` writes exactly
# `inp` cells and the caller reads back `capacity(inp)` cells. A ShapeMemory
# breaks that coupling — it produces an output whose shape DIFFERS from the
# input, with the output shape itself INFERRED IN-CONTEXT from the demos (a rule
# parameter like any other, never a hand-coded size heuristic). It factors, like
# every composed memory (block 5), into two parts fit on signals invariant to
# each other:
#   * a SHAPE RULE written closed-form from the demos (`write`) — position-free
#     shape arithmetic (in-dims -> out-dims), read back by `out_rows`/`out_cols`;
#   * a CONTENT rule (the ES-fit params in `state`) that fills the predicted
#     output grid via a gather from the input (`apply` takes the output dims).
# A distinct trait (parallel to SelfModMemory) keeps the same-shape core and all
# existing memories untouched: the shape-aware fitness/fit driver in
# esper_evolution.mojo is generic `[M: ShapeMemory]`, and the seam is additive.
# `param_dim`/`seed`/`fill_scale` mirror Memory (fill_scale ZEROS the written
# shape-rule slots so the ES never moves them — the GeomColor freeze trick).
trait ShapeMemory:
    comptime Dom: Domain

    @staticmethod
    def param_dim() -> Int:
        ...

    @staticmethod
    def seed(state: UnsafePointer[Float32, MutAnyOrigin]):
        ...

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        ...

    # Write the closed-form factors (the shape rule) from the demos — one
    # forward pass, never ES-searched.
    @staticmethod
    def write(
        state: UnsafePointer[Float32, MutAnyOrigin],
        demos: List[ExamplePair[Self.Dom.Example]],
    ):
        ...

    # Predicted output dims for `inp` under the written shape rule.
    @staticmethod
    def out_rows(
        state: UnsafePointer[Float32, MutAnyOrigin], inp: Self.Dom.Example
    ) -> Int:
        ...

    @staticmethod
    def out_cols(
        state: UnsafePointer[Float32, MutAnyOrigin], inp: Self.Dom.Example
    ) -> Int:
        ...

    # Produce an `out_rows * out_cols` output into `dst` (the output-shape-aware
    # apply — the caller supplies the predicted output dims).
    @staticmethod
    def apply(
        state: UnsafePointer[Float32, MutAnyOrigin],
        inp: Self.Dom.Example,
        out_rows: Int,
        out_cols: Int,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...
