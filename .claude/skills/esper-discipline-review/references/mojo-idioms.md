# Mojo 1.0.0b2 idioms — Esper cheat-sheet

The toolchain is **hard-pinned to Mojo 1.0.0b2** (PyPI, not the Modular wheel index).
The 1.0 line broke older-nightly code. Write these idioms from the start — the
`mojo_lint.py` PreToolUse hook blocks the deterministic violations, but the hook only
sees added lines, so internalize the whole set.

## Blocked by the hook (hard rules)

| Wrong (old / non-1.0) | Right (1.0.0b2) |
|---|---|
| `fn foo(...)` | `def foo(...)` — add `raises` if it can raise |
| `alias N = 8` | `comptime N = 8` |
| hand-written `__moveinit__` | derive `(Movable)` / `(Copyable, Movable)` |
| `Tensor` (stdlib type) | removed — raw `UnsafePointer` / SIMD slices |
| `from memory import ...` | `from std.memory import ...` (all stdlib is `std.`-qualified) |
| `from python import Python` in `src/` | forbidden on the runtime path — Python lives only in `tools/` |
| `import torch/numpy/jax/...` in `src/` | forbidden — zero external ML libs; learning is ES |

`std.`-qualified imports used throughout: `std.memory` (`alloc`, `memset_zero`,
`memcpy(dest=, src=, count=)`, `UnsafePointer`), `std.sys` (`size_of`,
`simd_width_of`, `argv`, `has_accelerator`), `std.math` (`fma`, `round`),
`std.random` (`randn_float64`, `seed`), `std.collections` (`List`, `InlineArray`).

## Warned by the hook / easy to get wrong (verify by hand)

- **SIMD equality returns a scalar `Bool`.** `a == b` on two SIMD vectors is *whole-vector*
  equality. For an elementwise mask use `a.eq(b)`, then
  `.cast[DType.float32]().reduce_add()` to count hits. (See `exact_match` in `arc_io.mojo`.)
- **No SIMD `.round()` method.** Use the free `round()` from `std.math` (works on SIMD and scalars).
- **Pointer fields need an explicit origin:** `UnsafePointer[Float32, MutAnyOrigin]`.
- **`UnsafePointer` is non-null by design** — do **not** guard `__del__` with
  `if self.data:`; free unconditionally.
- **Lifecycle:** `def __init__(out self, ...)`, `def __del__(deinit self)`. Pair `alloc`
  in `__init__` with `free` in `__del__` (unconditional). Moves consume the source, so a
  moved-from value's `__del__` never runs.
- **Raw memory / SIMD:** `var p = alloc[T](count)` / `p.free()`; arithmetic
  `(p + n).bitcast[T]()`; `p.load[width=nelts](i)` / `p.store[width=nelts](i, v)`.
- **The three-part SIMD loop shape (every time):**
  vectorized main loop `range(0, size - nelts + 1, nelts)`, then a **scalar remainder**
  loop `range(size - remainder, size)`. `comptime nelts = simd_width_of[DType.float32]()`.
- **No dynamic allocation in hot loops** — reuse the pre-allocated `ESWorkspace`
  (`src/esper_evolution.mojo`); never `alloc`/`free` per ES iteration.
- **POD over the arena** — structs placed in `HopeArena` (`HopeNode`) stay POD (raw
  pointers + ints + `InlineArray`, no owning members); reference other nodes by arena index.
- **GPU:** ships in the same PyPI wheel. `from std.gpu.host import DeviceContext`,
  `from std.gpu import global_idx`; gate device code with
  `comptime if not has_accelerator()`. Canonical smoke test: `tests/test_gpu_env.mojo`.

## Reference implementations to match
- Hot-loop SIMD/FMA: `src/esper_evolution.mojo` (`evolve_fast_weights` ~L191, the
  `fma()` step blocks, `evolve_slow`, `fit_shape`).
- Fitness / elementwise compare: `src/arc_io.mojo` (`calculate_fitness` ~L95, `exact_match`).
- Canonical (dormant) operator pattern: `src/hope.mojo` (`update_fast_weights` ~L366).
