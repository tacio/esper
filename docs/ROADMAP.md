# Esper Roadmap

The durable, canonical view of Esper's **direction**: what it is, the discipline it runs by,
what's done, what's next, and the horizon beyond. This is the source of truth for *where the
project is going*. Companion docs: **`docs/JOURNAL.md`** is the timestamped narrative (the *why*
behind every step — this roadmap links to it for depth), **`docs/NL-summary.md`** distills the
theory (the "Nested Learning"/HOPE paper), and **`CLAUDE.md`** holds the conventions and hard
constraints.

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

## Next — the path to full ARC-AGI 2

Each is its own block, held to the **cold-fit bar** (a scaffolded pass is a negative result). The
emergent memories are each measured on the subset they express; the north-star metric is the raw
held-out ARC-AGI-2 number.

1. **Richer neighbourhoods / nonlinear read.** Grow the 2-D context beyond the additive 2-neighbour
   class — full von Neumann / Moore neighbourhoods, and a nonlinear read for disjunctive/count rules
   ("any neighbour is X", majority) that a linear `S·k` can't express.
2. **Compose content + geometry.** Combine the grid content self-write (`GridContextSelfModMemory` /
   `DeltaSelfModMemory`) with the B3 `AttnGatherMemory` global-read (geometry + content) — the honest
   route to finally **retiring `OperatorMemory`** (B3's open thread).
3. **Shape change.** Handle outputs whose dims ≠ inputs (currently scored 0 by the same-shape guard) —
   a Domain / output-size generalization.
4. **Multi-block CMS chain** (NL §7). Stack memories at multiple frequencies for multi-step /
   object-level reasoning.

## Beyond ARC-AGI 2 (TBD)

The ES / two-timescale / self-modifying core is deliberately **domain-agnostic** (proved by B2), so it
should serve reasoning problems well beyond ARC. The long horizon — a fuller Continuum Memory System
and a self-modifying *optimizer* (the memory shaping its own learning rule end-to-end, HOPE §4/§8) —
is intentionally **TBD** and will be fleshed out once the ARC-AGI 2 path is further along.

---

## How planning works (the decoupling)

- **`docs/ROADMAP.md` (this file)** — the durable project *direction*. Update it when a milestone lands
  or the plan changes.
- **Claude Code plan files** (e.g. `~/.claude/plans/…`) — **ephemeral**, per-session *implementation*
  plans. Not the roadmap; not authoritative for direction.
- **`docs/JOURNAL.md`** — the timestamped *narrative* (why the code looks the way it does).
- **`docs/NL-summary.md`** — the *theory* (HOPE / Nested Learning).
- **`CLAUDE.md`** — *conventions* and hard constraints for working in the repo.
