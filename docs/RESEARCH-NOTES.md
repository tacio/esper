# Research Notes — external methods mapped onto Esper

Literature findings gathered at decision points, each mapped onto Esper's concepts (fast/slow
weights, the ES, self-modifying memories, the Domain/Memory seam). Companion docs: **`ROADMAP.md`**
(direction — links here for the evidence behind direction changes), **`JOURNAL.md`** (the narrative
of *when and why* each research pass happened), **`NL-summary.md`** (the HOPE/Nested-Learning theory).

Rules for this file: newest section on top; every claim carries its source; each finding ends with
the *Esper mapping* — what it means for us concretely, not just what the paper says.

---

## 2026-07-02 — Composing emergent memories (the block-5 soft/sharp wall)

**Trigger.** Block 5 (compose geometry + colour into one memory, retire `OperatorMemory`) hit an
irreconcilable coupling after ~7 mechanism iterations: the geometry ES needs a **soft** attention
gather (a gradient in the projection `M`), while recolor needs a **sharp** gather (clean colour
bins); one learned temperature + one colour table cannot serve both, and every fix traded transpose
against recolor (details in JOURNAL 2026-07-02). Development paused to survey how the field composes
separately-learned modules and handles soft↔hard discreteness.

### 1. Energy-based compositional inference ⭐ (the direct dissolve)

- [Generalizable Reasoning through Compositional Energy Minimization](https://arxiv.org/pdf/2510.20607) (2025)
- [Learning Iterative Reasoning through Energy Diffusion (IRED)](https://energy-based-model.github.io/ired/) (ICML 2024)

Recast each module not as a forward map `apply(w, in) → out` but as an **energy**
`E_module(out | in, demos)` — low energy = "this output is consistent with what I know". Modules are
trained **separately**; at inference they compose by **summation** `E = E_geom + E_colour`, and the
answer is found by minimizing the combined energy over the *output* with an **annealed** schedule
(gradient descent / MCMC, landscape sharpened progressively). IRED's key technique is learning a
*sequence of annealed energy landscapes* — the soft→hard schedule lives on the **inference loop**,
not inside any module.

**Esper mapping.** This dissolves the block-5 wall rather than solving it head-on:
- No shared `beta`: each energy keeps its own internal sharpness; annealing moves to the solver —
  the principled version of what block 5 tried to force into one fixed temperature.
- Composition is training-free and **additive** (`+`), which respects the no-runtime-selector
  principle: a summed energy is a conjunction of constraints, not a DSL over memories.
- Maps onto HOPE cleanly: modules = slow priors; test-time energy minimization = the fast
  in-context adaptation. Solvable with our existing annealed loop (derivative-free-friendly:
  ES/MCMC over the output), pure Mojo, zero external libs.
- Concrete seam: an `EnergyMemory` trait exposing `energy(out, in, demos)`; geometry- and
  colour-energies seeded from the already-proven `AttnGatherMemory` and recolor memories.

### 2. Identifiability of modular structure ⭐ (the emergent-pipeline guardrail)

- [Discovering modular solutions that generalize compositionally](https://arxiv.org/abs/2312.15001) (Schug et al., ICLR 2024)

Teacher-student theory for when primitive modules are **provably recoverable** from demonstrations
alone (up to linear transformation) after seeing only **O(M)** module combinations, not the
exponential number. Composition = a **linear hypernetwork**: task weights `W(z) = Σ_m z_m·Θ_m`
with `z` a per-task latent code and `Θ` shared module templates; meta-learning is bilevel (inner:
fit `z` per task; outer: fit `Θ` across tasks). The three conditions:

1. **Compositional support** — every primitive appears in *some* training task.
2. **Connected support** — task families overlap in the modules they use (no isolated clusters).
3. **No over-parameterization** — student capacity ≈ true module count; **excess capacity makes the
   learner memorize per-task solutions instead of factoring shared primitives**.

**Esper mapping.** This is the theory behind "later make the pipeline itself emergently learned":
- The bilevel scheme **is** our Reptile slow/fast split — `z` = fast code, `Θ` = slow templates.
  An emergent pipeline = meta-learn hypernetwork templates over a task distribution engineered for
  conditions 1–2; novel test-time combinations then generalize by linear algebra, no retraining.
- Condition 3 is a live caution *today*: our memories are likely over-capacity for single-transform
  task families, which pushes them toward per-task memorization instead of factoring geometry/colour
  as reusable primitives. Right-sizing capacity is a lever we have not yet pulled.
- `synth_tasks.py` is where conditions 1–2 are engineered (task families must *share* primitives —
  e.g. flip∘recolor, transpose∘recolor — not only appear in isolation).

### 3. Learned decoupling (our decouple-lesson, made adaptive)

- [Advancing CMA-ES with Learning-Based Cooperative Coevolution](https://arxiv.org/html/2504.17578v1) (2025)

Confirms the recurring Esper lesson (a derivative-free ES cannot jointly fit coupled spaces —
decouple) and upgrades it: **learn the decomposition** instead of hand-fixing it. A small learned
policy watches cheap statistics of the running optimizer (within-group correlation, step-size,
improvement history — zero extra function evaluations) and decides per-step whether variable groups
are optimized jointly or separately, switching strategies when improvement stalls.

**Esper mapping.** Our decoupling (geometry ES + closed-form colour; scoring-salience vs gradient
self-write) is hand-designed per block. A meta-learned grouping policy over the same kind of cheap
demo/fitness statistics is the emergent version — same shape as block 4's scoring salience, one
level up. Not urgent; becomes relevant when the number of interacting parameter groups grows.

### 4. Latent program spaces (the reserve architectural pivot)

- [Searching Latent Program Spaces](https://arxiv.org/pdf/2411.08706) (2024)

Learn an encoder/decoder so each task is a **continuous latent `z` that decodes to an executable
operator**; test-time adaptation = gradient/gradient-free search over `z` only. The decoder
guarantees every point decodes to a *valid* operator, so discrete structure (the argmax) leaves the
search loop entirely; searching the smooth latent space replaces searching raw parameter space.

**Esper mapping.** An alternative to ES-over-raw-operator-params for the grid path: meta-learn a
decoder (slow), search `z` in-context (fast). A bigger change than #1 — hold in reserve if the
energy reframe stalls. Note the family resemblance to #2: both put a small per-task code in front
of big shared weights; the code is what the fast timescale fits.

### Context: where the field is (ARC Prize 2025)

- [ARC Prize 2025: Technical Report](https://arxiv.org/html/2601.10904v1) ·
  [ARC-AGI 2025 research review](https://lewish.io/posts/arc-agi-2025-research-review)

Test-time training dominates; top entries run **refinement loops in weight space, not symbolic
space** (winner NVARC: 24% with a 4B model + synthetic data + TTT); "fast weights = hidden state
updated by an online learner" (fast-weight programmers / linear-transformer view). Esper's core bet
— per-task in-context fitting of a parametric memory, no symbolic DSL — is aligned with the
frontier, not fighting it.

### Distilled into the roadmap

1. **Block 5 reroute:** retire `OperatorMemory` via **energy composition** (#1), not a single
   jointly-fit memory — geometry- and colour-energies trained separately, summed at inference,
   annealing on the solver. The single-memory joint fit is a documented negative result.
2. **Emergent pipeline (new horizon item):** hypernetwork templates + per-task code under
   identifiability conditions (#2); engineer `synth_tasks.py` families for compositional +
   connected support; watch memory capacity (condition 3).
3. **Adaptive decoupling (#3)** and **latent program search (#4)** parked as reserves.
