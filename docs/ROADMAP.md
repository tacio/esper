# Esper Roadmap

The durable, canonical view of Esper's **direction**: what it is, the discipline it runs by,
what's done, what's next, and the horizon beyond. This is the source of truth for *where the
project is going*. Companion docs: **`docs/JOURNAL.md`** is the timestamped narrative (the *why*
behind every step — this roadmap links to it for depth), **`docs/NL-summary.md`** distills the
theory (the "Nested Learning"/HOPE paper), **`docs/RESEARCH-NOTES.md`** maps external literature
onto Esper at decision points (the evidence behind direction changes), and **`CLAUDE.md`** holds
the conventions and hard constraints.

> Per-session implementation plans (Claude Code plan files) are **ephemeral** and scoped to one
> session — they are *not* the project roadmap. This file is.

---

## What Esper is & the spine

A bare-metal, pure-Mojo neuro-symbolic reasoning engine (target: ARC-AGI 2). The **spine**, which
all work respects:

- **Fast weights = a *learned* memory fit in-context** by a derivative-free Evolution Strategy on
  each task's demonstration pairs; **slow weights = the meta-learned prior**. Two timescales (HOPE).
- **Never hand the engine a symbolic DSL.** Geometric/colour/relational transforms must *emerge* as
  fitted parameters; symbolic transforms live only in the offline generator (`synth_tasks.py`) as
  ground truth to rediscover. Structured priors are removable "training wheels".
- **The metric is held-out generalization** — fit on a task's train pairs, score the *unseen* test
  pair. Uncheatable by memorization.
- The north star is a **fully emergent, self-modifying, multi-frequency memory** (HOPE: two-timescale
  + self-modifying block + Continuum Memory System). See `docs/NL-summary.md`.

## Working principles (hard-won discipline)

These are the rules the project runs by — learned, sometimes painfully, and worth keeping across
sessions:

- **Held-out generalization is the only success metric.** Report the real number, including 0%.
- **The emergence bar is a single COLD fit.** If a memory only reaches the bar via per-task
  hand-staging, mid-fit parameter boosts, or bespoke tuning, that is **"stone soup"** — the reasoning
  lives in the human, not the model — and it is recorded as a **documented negative result**, not a
  milestone. (This is why the combined geometry+colour memory was abandoned; see B3.)
- **Move the search off the fast weights.** A derivative-free ES cannot fit coupled/high-dim *fast*
  weight spaces. The fix (B3→B4 reframe): the ES fits only the small/slow params; **fast adaptation is
  the memory's own write rule**, run over the demonstrations in a forward pass, never ES-searched.
- **Additive, trait-based growth.** Memories conform to `Memory`/`SelfModMemory` over a `Domain`; each
  is measured on the subset it expresses. There is **no runtime memory-selector** (that would be a DSL
  over memories). New capability is additive — the generic ES/two-timescale core stays untouched.
- **Sharpness belongs to the solver, not the module.** When composed modules need *different* softness
  (block 5: geometry wants a soft gather, colour wants sharp bins), no single shared temperature works;
  compose additively (e.g. summed energies) and put the soft→hard **annealing on the inference loop**.
- **At a real wall, pause and survey.** When a mechanism fails for a *general* reason (not a tuning
  reason), stop iterating and check the literature; record the findings in `docs/RESEARCH-NOTES.md`
  mapped onto Esper's concepts, then reroute the roadmap.
- **Journal every step** (`docs/JOURNAL.md`) — the *why*, the discoveries, the blockers and fixes.

## Status — done

Concise, milestone-level; each links to `docs/JOURNAL.md` for the full narrative.

### Phase A (M1–M9) — learn ARC transforms and generalize, held-out

- **M1–M7 — the learned operator.** Fast weights parameterize a grid→grid operator (6-param centered
  affine + 10-entry colour LUT), fit in-context by the annealed ES. The whole expressible subset
  {flip_h, flip_v, transpose, recolor} reaches **held-out 1.0, train/test gap 0.0**. Key design calls:
  smooth bilinear gather (the ES needs a gradient everywhere), colour-then-gather decoupling, a
  per-group ES preconditioner. (JOURNAL 2026-06-29.)
- **M8 — real ARC-AGI 2, honest number.** Ingest the real corpus; held-out eval with a same-shape
  guard: **5 / 1000 solved on the training split, 0 / 120 on the public-eval split** — the honest
  ceiling of a 16-param operator (most of ARC is outside geometry+colour). Reproducible, parallel eval.
  (JOURNAL 2026-06-29 / 06-30.)
- **M9 — the slow prior is meta-learned.** Reptile outer loop turns `slow` from a fixed identity anchor
  into a meta-learned prior; a fresh in-family task fits at a *narrow* eval budget where a cold prior
  can't (meta-prior held-out 1.0 vs 0.0). The second timescale is real. (JOURNAL 2026-06-29.)

### Phase B (B1–B4) — the abstraction seam and progressive emergence

- **B1 — Memory/Domain seam + first emergent memory.** The ES/two-timescale core is generic
  `[M: Memory]` (associated `Dom: Domain`); zero-overhead static dispatch. `OperatorMemory` wraps the
  operator (suite stays green); `MLPMemory` — a per-cell tanh MLP — learns recolor to **held-out 1.0
  with no hand-coded LUT**, the first training-wheel removal. (JOURNAL 2026-06-30 14:37.)
- **B2 — second (non-grid) domain.** A 1-D `Sequence` domain plugs into the seam with **zero core
  changes** (the proof the abstraction isn't grid-coupled); metrics reuse the shape-agnostic
  `calculate_fitness`/`exact_match` for free. `SeqOperatorMemory` (reverse/increment) and `SeqMLPMemory`
  (increment) both **held-out 1.0**. (JOURNAL 2026-06-30 18:39.)
- **B3 — emergent global addressing (geometry).** `AttnGatherMemory` re-earns flip/flip/transpose via a
  learned position-attention gather (2×2 coord projection + temperature), **no hand-coded affine** —
  **held-out 1.0, first try**. **NEGATIVE RESULT (recorded):** folding colour onto this into one
  combined memory to retire `OperatorMemory` only moved under hand-staging/β-boost scaffolding = stone
  soup; abandoned. Retiring the operator is a real research step (routed to a better mechanism), not
  tuning. (JOURNAL 2026-06-30 19:10.)
- **Compute — CPU-parallelized ES.** The 2·N antithetic fitness evals per iteration are parallelized
  across cores, **bit-identical** to the sequential loop (same numbers, ~3.5× on the heavy attention
  fit). GPU is **deferred**: the pinned slim `mojo` wheel has no `gpu` package — it needs a
  MAX-platform migration off the hard version pin, worthwhile only when the workload outgrows the
  CPU cores. (JOURNAL 2026-06-30 23:03.)
- **B4 — self-modifying memory.** *Step 1:* `RecolorSelfModMemory` writes its own colour map from the
  demos in one forward pass with a meta-learned read; a fresh recolor permutation adapts in a **single
  pass** (held-out 1.0) vs the ES-fit MLP's ~4000-iter per-task fit. *Fuller block:*
  `DeltaSelfModMemory` self-generates key/η/α and runs a **gated delta-rule self-write**
  `S ← (1−α)S + η(v−S·k)k` (the internal associative objective); on a sequence local-context rule
  `out[i]=f(in[i],in[i-1])` it meta-learns its projections **cold** → a fresh *arbitrary* rule solved
  to **held-out ~0.98 in one adapt pass**. Discovery: unit-normalise the key or the delta write
  diverges. (JOURNAL 2026-06-30 23:42 / 2026-07-01 12:25.)
- **2-D context keys (first ARC-AGI-2 block).** `GridContextSelfModMemory` lifts the self-write to a 2-D
  toroidal grid neighbourhood: the key is `concat(E[center]⊗E[up], E[center]⊗E[left])` (unit-normalised),
  η/α self-generated, a gated delta write over a few epochs. On the additive local rule
  `out[r,c]=h1(center,up)+h2(center,left)` — genuinely 2-D (per-cell/1-D memories can't express it) — it
  meta-learns its neighbourhood projections **cold** → a fresh rule solved to **held-out ~0.985 in one
  adapt pass** (generic seed 0.11). Reuses the meta-fit with zero core change. Expressible class =
  *additive* center↔neighbour rules (disjunctive/count/mod-wrap need a nonlinear read — later).
  (JOURNAL 2026-07-01 14:44.)
- **Richer neighbourhoods + nonlinear read (ARC-AGI-2 block 2).** `GridNbhdSelfModMemory` crosses the
  linear barrier: it keys on a **Moore-8 neighbour-count histogram** (centre-free, mean-aggregated) and
  reads through a **sigmoid THRESHOLD**, expressing the DISJUNCTIVE / COUNT class
  `out = C1 if (#neighbours == P) ≥ t else C2` — an OR / count that a linear `S·k` (GridContext) provably
  cannot be, and that no per-cell / 1-D memory can (needs all 8 neighbours). Self-write is the
  perceptron-style gated delta rule; the ES meta-learns only the small slow vector (embeddings + read
  scalars + gates). Cold meta-fit → a fresh unseen rule (unseen predicate colour) solved **held-out 1.0
  in one adapt pass** (generic seed 0.0). **Honesty control:** the same write with a LINEAR read reaches
  only 0.61 vs the nonlinear 0.995 — the nonlinearity is load-bearing, not scaffolding. Three empirically
  forced design calls (centre-free key; mean not unit-norm; feature-scale + bias slot). Scope = balanced
  `t=2`, fixed output pair, predicate colour inferred in-context; varying `t` and multi-bin count deferred.
  (JOURNAL 2026-07-01 16:05.)
- **Broaden the count class (ARC-AGI-2 block 3).** Generalised `GridNbhdSelfModMemory` (in place; the
  block-2 case is a strict subset) so it infers the **WHOLE 2-level rule** from demos —
  `out = C1 if (#Moore-8 == P) ≥ t else C2` with **P, t, AND both output colours C1/C2 varying per task**.
  Key idea: **decouple colour from threshold** — read the two colours off the demos (`v0/v1 = min/max`
  output, written) and train the salience as a **binary classifier** of which colour a cell outputs, so
  its learned sign handles inverted rules (fire → the smaller colour) and the bias slot self-calibrates
  `t`. Cold meta-fit **from a zero-embedding prior** (the honest "no representation" baseline: `before`
  = best-constant ~0.74) → a fresh arbitrary rule solved **held-out 1.0 in one adapt pass**; linear-read
  control 0.48. (JOURNAL 2026-07-01 17:20.)
- **Multi-bin count map (ARC-AGI-2 block 4).** `GridCountMapSelfModMemory` reads an **arbitrary**
  `out = M(count_P)` (map count → colour, non-contiguous / non-monotone, ≥3 output colours); `P` and `M`
  inferred per task. **Negative result then fix (recorded):** the "obvious" coupled gradient self-write
  (learn salience + value table together) fails — at `S=0` the only bootstrap signal is linear covariance,
  which vanishes for non-monotone maps (identifying *which* colour to count is a discrete selection, not a
  gradient target). The fix holds the emergent bar with a **meta-learned SCORING salience**: per-colour
  demo statistics (variance-reduction, correlation, mean-count) → a learned score picks `P`; the meta-fit
  **learns to weight variance-reduction over correlation** (the feature that fails non-monotone). Read =
  soft count-bin value table. Ckpt A: arbitrary non-monotone maps **1.0** vs the block-3 2-level memory
  **0.41** on the same ≥3-colour map. Cold meta-fit (w=0 seed → uniform → 0.32): fresh unseen map
  **held-out 1.0 in one adapt pass**. (JOURNAL 2026-07-01 18:30.)
- **Compose geometry + colour, single memory — NEGATIVE RESULT (ARC-AGI-2 block 5, recorded).** A single
  ES-fit composed memory (attention gather + colour table, ES on geometry only + closed-form colour) hits
  an **irreconcilable soft/sharp coupling**: the geometry ES needs a SOFT gather (a gradient in `M`),
  recolor needs a SHARP gather (clean colour bins); one temperature + one colour table cannot serve both,
  and a free colour table **absorbs** geometry error at wrong geometries (flattening the ES contrast).
  ~7 mechanism iterations, each trading transpose against recolor; the parts are each proven separately
  (AttnGather: all geometry 1.0; recolor memories: 1.0). Root cause is general, so development paused for
  a literature pass → the reroute is **energy-based composition** (see Next #1 and `RESEARCH-NOTES.md`
  2026-07-02). (JOURNAL 2026-07-02.)

## Next — the path to full ARC-AGI 2

Each is its own block, held to the **cold-fit bar** (a scaffolded pass is a negative result). The
emergent memories are each measured on the subset they express; the north-star metric is the raw
held-out ARC-AGI-2 number.

1. **Compose content + geometry — via energy composition.** The single-jointly-fit-memory route is a
   recorded negative result (block 5: the soft/sharp coupling). The reroute (`RESEARCH-NOTES.md`
   2026-07-02 #1): recast memories as **energies** `E(out | in, demos)`, train geometry- and
   colour-energies **separately** (both already proven as forward memories), compose at inference by
   **summation** `E = E_geom + E_colour`, and solve by minimizing the summed energy over the output
   with an **annealed** schedule — the soft→hard schedule lives on the solver, not inside any module,
   so each energy keeps its own sharpness. Additive and selector-free (a summed energy is a
   conjunction of constraints, not a DSL over memories). This is the honest route to finally
   **retiring `OperatorMemory`** (B3's open thread), and doubles as the composition mechanism for
   the pipeline horizon below.
2. **Shape change.** Handle outputs whose dims ≠ inputs (currently scored 0 by the same-shape guard) —
   a Domain / output-size generalization.
3. **Multi-block CMS chain** (NL §7). Stack memories at multiple frequencies for multi-step /
   object-level reasoning.
4. **Real ARC-AGI-2 re-measure.** With the count/neighbourhood family in hand, re-run the honest held-out
   eval (`arc_solve --report`) to see whether the emergent memories move the raw corpus number off the
   M8 operator ceiling (5/1000).

## Beyond ARC-AGI 2

The long-term goal: an architecture trainable for problems **completely different** from ARC that
require neuro-symbolic reasoning and continuous learning. The ES / two-timescale / self-modifying
core is deliberately **domain-agnostic** (proved by B2); every blocker fix should move toward this
vision, not just past the blocker.

- **The emergent pipeline.** A pipeline of emergent memories whose *composition is itself learned*.
  The theory guardrail (`RESEARCH-NOTES.md` 2026-07-02 #2, Schug et al. ICLR 2024): primitive modules
  are provably recoverable from demonstrations after only O(M) combinations via a **hypernetwork**
  (task weights = per-task code × shared templates — exactly our Reptile fast/slow split) *iff* the
  task distribution has compositional + connected support and the student is **not over-parameterized**
  (excess capacity ⇒ per-task memorization instead of factored primitives). Practical consequences
  today: engineer `synth_tasks.py` families that *share* primitives (e.g. flip∘recolor), and treat
  memory capacity as a lever, not just headroom.
- The fuller **Continuum Memory System** and a self-modifying *optimizer* (the memory shaping its own
  learning rule end-to-end, HOPE §4/§8) remain the far horizon, fleshed out as the ARC-AGI 2 path
  matures.

---

## How planning works (the decoupling)

- **`docs/ROADMAP.md` (this file)** — the durable project *direction*. Update it when a milestone lands
  or the plan changes.
- **Claude Code plan files** (e.g. `~/.claude/plans/…`) — **ephemeral**, per-session *implementation*
  plans. Not the roadmap; not authoritative for direction.
- **`docs/JOURNAL.md`** — the timestamped *narrative* (why the code looks the way it does).
- **`docs/RESEARCH-NOTES.md`** — external literature mapped onto Esper at decision points (the
  *evidence* behind direction changes).
- **`docs/NL-summary.md`** — the *theory* (HOPE / Nested Learning).
- **`CLAUDE.md`** — *conventions* and hard constraints for working in the repo.
