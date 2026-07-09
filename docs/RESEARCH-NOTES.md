# Research Notes — external methods mapped onto Esper

Literature findings gathered at decision points, each mapped onto Esper's concepts (fast/slow
weights, the ES, self-modifying memories, the Domain/Memory seam). Companion docs: **`ROADMAP.md`**
(direction — links here for the evidence behind direction changes), **`JOURNAL.md`** (the narrative
of *when and why* each research pass happened), **`NL-summary.md`** (the HOPE/Nested-Learning theory).

Rules for this file: newest section on top; every claim carries its source; each finding ends with
the *Esper mapping* — what it means for us concretely, not just what the paper says.

---

## 2026-07-08 — Content-addressed construction (the deep-floor negative)

**Trigger.** CMS-0 returned STOP (3/146 deep-floor tasks chain-of-proven-factors shaped) and the
same-day factor-coverage scan sharpened it: nine candidate object-level per-cell key families cover
a union of **4/146** — the floor lies outside the *whole class* of per-cell functions over
position-aligned context. Its dominant measured property is **content-addressed construction**:
output written at positions other than where the input evidence sits (move / copy / draw / extend).
This pass surveys mechanisms of content-addressed retrieval and construction compatible with the
spine (per-task in-context fit, no DSL, no backprop on the runtime path).

### 1. Recursive answer refinement — TRM ⭐ (the constructive loop)

- [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
  (Jolicoeur-Martineau, 2025; ARC Prize 2025 Paper Award 1st) ·
  [follow-up analysis](https://arxiv.org/abs/2512.11847)

A **7M-param, 2-layer** network scores 45% on ARC-AGI-1 (8% on ARC-AGI-2) — above most frontier
LLMs — by making inference *constructive*: it maintains a full materialized **answer grid `y`**
plus a latent scratchpad `z`, and repeatedly applies one tiny network as
`z ← net(x, y, z)` (×6) then `y ← net(y, z)` (×1), for up to 16 improvement steps (≈42 effective
recursion depth). Self-attention runs over the whole embedded grid, so any output cell can read any
input *or current-answer* cell by content. Ablations are emphatic: a *single* small network beats
HRM's dual 4-layer stack (87.4% vs 82.4% on Sudoku-Extreme); 4 layers is *worse* than 2 (79.5%);
full backprop through the recursion is the largest single gain over the fixed-point approximation
(56.5%→87.4%). Trained with deep supervision + heavy augmentation; no cross-task meta-learning.

**Esper mapping.** The escape from the per-cell class is architectural, not a richer key: replace
`out = f(in)` computed once per cell with **`out` = the fixed point of repeated small edits
conditioned on (input, current answer)** — each pass can *write* somewhere other than where it
*read*, which is exactly the measured missing capability. This is rung #6's self-mod write rule
made iterative: the fast adaptation loop already re-applies the memory's own write rule over demos;
TRM says run the same shape at *inference* over a materialized answer grid, with the ES
meta-learning only the small update rule (the slow vector). The "less is more" ablation is the
Schug over-capacity guardrail confirmed from the opposite direction — recursion depth substitutes
for parameters, which suits a derivative-free ES (small `OP_DIM`, more forward passes) far better
than one big forward map. Caveat: TRM trains by backprop through 7M params; Esper's version must
keep the meta-learned rule small enough for ES — the bet is that TRM's *capacity* is mostly doing
what our written tables + substrate already do, and only the *edit rule* needs learning.

### 2. Content-keyed attention (AttnGather: position-query → content-key) ⭐ (the nearest rung)

- [Key-value memory in the brain](https://arxiv.org/html/2501.02950v1) (Gershman, Fiete, Irie 2025)
  — the general theory: attention = content-addressed kv retrieval; the correlation-matrix /
  fast-weight-programmer view ties it to our fast-weights framing.
- [Pointer-Augmented Neural Memory](https://arxiv.org/pdf/2404.11870) (2024) — separating *where*
  (pointer/address) from *what* (content) is what lets a small model copy never-seen content:
  the process generalizes because it manipulates addresses, not values.

The proven `AttnGather` is already attention — but its queries are **affine functions of position**
(7 params), so it can only express position-permutations. The generalization: output cell `i` forms
a query from its position *and local content*; keys/values are computed from all input cells;
the gather is the same softmax read the GPU kernel already runs. "Copy the colour of the cell whose
content matches X" — the move/copy/draw class — becomes expressible with a handful of extra
parameters (a small bilinear form over cell features), still one ES search, still one kernel shape.

**Esper mapping.** The smallest step from the existing spine that reaches content-addressing:
`AttnGatherMemory`'s score function gains content terms (colour match, object-membership match,
substrate features) alongside the affine position map; `fill_scale` freezing, the pre-map recipe,
and the GPU fitness path all carry over. **Pre-build gate:** prototype as `factor_scan` read
families first — content-addressed reads (value fetched from a position *selected by content*:
object-anchor-relative reads, nearest-cell-with-colour-p, unique/odd-object lookup) measured
against the same 146 ids. Only a family that clears the coverage bar gets built in Mojo.

### 3. Object slots + the relational bottleneck (substrate for the write rule)

- [Slot Abstractors](https://arxiv.org/abs/2403.03458) (Mondal et al., ICML 2024) ·
  [OCRA](https://arxiv.org/pdf/2306.02500) (2023)

Slot attention factors a scene into object slots; a **relational bottleneck** (downstream reasoning
sees only slot–slot *relations*, not raw features) is load-bearing — replacing relational
cross-attention with standard cross-attention costs up to **52%**. Scales to many objects and
multi-rule problems where earlier object-centric models capped out at one rule.

**Esper mapping.** `factor_scan.precompute` (components, bboxes, sizes, ranks, distances) is
already a hand-coded slot extractor — and the scan *proved* slot features keyed at the aligned cell
are not enough (4/146). The lesson lands one level up: slots must feed **where to write** (the
self-mod rule selects a slot and a target region), not decorate a per-cell key. The relational
bottleneck is the emergence-friendly form: a write rule reading only slot relations (same-colour-as,
larger-than, inside) inherits generalization over novel colours/sizes for free — the object-level
analogue of writing each factor on an invariant signal. Emergent slot *discovery* is training
wheels we can defer: connected components stay substrate, as the scan already treats them.

### 4. Per-task-only fitting has demonstrated headroom — CompressARC (existence proof)

- [ARC-AGI Without Pretraining](https://arxiv.org/abs/2512.06104) (Liao & Gu, 2025; Paper Award 3rd)

**76K params, no pretraining, no dataset, no search**: a randomly-initialized equivariant network
is fit *on the single target task* by gradient descent on an MDL objective (a VAE-style loss whose
description-length term prices the weights), reaching 20% on ARC-AGI-1. Compression replaces
program search; the equivariances (colour permutation, dihedral) are the only prior.

**Esper mapping.** The closest published relative of the cold-fit bar (nothing warm-started,
per-task only) — evidence that Esper's no-pretraining path has real headroom (20% with zero prior
knowledge). MDL is the principled version of the Schug capacity guardrail and of our
strictly-reduces-residual acceptance gates: a written table should pay for its description length
in residual reduction. It uses backprop, but at 76K params an ES-fit variant is not absurd —
parked as a reserve; the near-term value is the *objective*, not the architecture.

### Context: the field converged on refinement loops (ARC Prize 2025)

- [ARC Prize 2025: Technical Report](https://arxiv.org/html/2601.10904v1) ·
  [research review](https://lewish.io/posts/arc-agi-2025-research-review)

The unifying theme across winners is **generate → verify → refine** per task: NVARC (1st, 24%)
TTT + synthetic data; the ARChitects (2nd, 16.5%) a 2D-aware **masked-diffusion** model —
the output grid *constructed* by iterative denoising with recursive self-refinement — replacing
their 2024 autoregressive system; MindsAI (3rd) TTT + augmentation ensembles. Nobody in the top
tier computes the output in one per-cell pass; everyone iterates against a verification signal.
Esper already has the verification signal (demo residual = ES fitness); what it lacks is the
constructive iteration (#1).

### Distilled into the roadmap

1. **Nearest rung — content-keyed gather (#2):** extend `factor_scan` with content-addressed read
   families (object-anchor-relative, nearest-colour, unique-object lookup) as the pre-build gate
   over the 146; on coverage evidence, extend `AttnGatherMemory`'s score with content terms
   (same ES, same GPU kernel shape).
2. **Rung #6 shaping — constructive self-mod (#1 + #3):** the meta-trained write rule becomes an
   *iterated editor* over a materialized answer grid (TRM's y/z loop), reading object-slot
   relations (relational bottleneck) to decide where to write; ES meta-learns only the small rule.
3. **MDL acceptance (#4):** adopt description-length-vs-residual as the uniform form of the
   capacity guardrail on written stages; CompressARC parked as the ES-fit reserve architecture.

### Follow-through (2026-07-09): both forward branches scanned → STOP; the band is a SELECTION problem

Rung CF landed the WRITTEN content read and moved the deep floor (mean held-out 0.188 → 0.625),
leaving a ~72-id partial-fix band at ~0.9. The two mechanisms distilled above (#1/#3 constructive
editor, #2 soft content-keyed gather) were each prototyped as a calibrated `factor_scan` family
class and pre-gated on the 146 before any Mojo:

- **Soft content-keyed gather (#2).** `softscore-*` = centre-*relative* relational reads (nearest
  cell whose component is strictly larger/smaller than the centre's, emit its value) — the faithful
  scan proxy for the additive-score argmax `src[argmax_j(-β|q_i-x_j|² + Σw_k·feat_k)]` (I rejected a
  "reflect through a content landmark" family: additive-score argmax cannot express reflection about
  a data point). Calib 20/20 pos, 0/20 neg. Result: **+3 ids** incremental over the hard content
  union (bar 15) → **STOP**. The ES-moveable soft score does not open band territory beyond the
  sharp CF table.
- **Constructive editor (#1+#3).** `scan_editor` = a faithful TRM-loop simulation: materialized
  answer grid, one colour-abstract local relational rule read over the *evolving* grid, ≤16 passes
  to a fixed point, **constant votes dropped** (coverage earned only through relational propagation
  — writes-as-evidence, positions-written ≠ positions-read). Calib 20/20 extend + 20/20 flood,
  0/20 recolor. Result: **+0 ids** incremental over the entire per-cell/content union (bar 15) →
  **STOP**. The deep floor contains no colour-abstract local-propagation class of any size.

**Reframe.** The band is not under-powered *selection scores* (gather) nor *elsewhere-construction*
(editor): it is CF's **selection consistency**. The near-miss block is dominated by `copy-*`
families at LOO 0.70–0.90 with net_fix 0.25–0.5 — the correct *source* grazed, the keyed table not
consistent enough to cross the 0.90 bar. The measure-first evidence points the next rung at
*sharpening CF's existing content read* (key granularity, tie-breaking, band LOO consistency), not
a new memory family. MDL acceptance (#4) is unspent and still applies there.

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
