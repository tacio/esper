# Research Notes — external methods mapped onto Esper

Literature findings gathered at decision points, each mapped onto Esper's concepts (fast/slow
weights, the ES, self-modifying memories, the Domain/Memory seam). Companion docs: **`ROADMAP.md`**
(direction — links here for the evidence behind direction changes), **`JOURNAL.md`** (the narrative
of *when and why* each research pass happened), **`NL-summary.md`** (the HOPE/Nested-Learning theory).

Rules for this file: newest section on top; every claim carries its source; each finding ends with
the *Esper mapping* — what it means for us concretely, not just what the paper says.

---

## 2026-07-10 — Vision B study round: intrinsic motivation, open-endedness, world models

**Trigger.** Deliberate pause from the Vision A rung ladder to flesh out **Vision B** (open-ended
mastery, zero hand-coded goals — currently a placeholder in ROADMAP.md). This pass surveys the three
seeded areas (unsupervised skill discovery / empowerment, UED / open-endedness, world models) plus
the adjacent territory that matters most to *us*: which of these mechanisms survive our values —
derivative-free (ES, no backprop), bare-metal, commodity hardware, tiny corpora, held-out-style
honest metrics — and what toy problem a Vision B POC should be built around.

### 1. Skill discovery: the empowerment / mutual-information family

- [Dynamics-Aware Unsupervised Discovery of Skills (DADS)](https://openreview.net/forum?id=HJgLZR4KvH)
  (Sharma et al., ICLR 2020) — the seed paper. Intrinsic objective: maximize **MI(s′; z | s)** —
  skills `z` should produce *predictable, distinguishable* state transitions. Crucially it learns a
  **skill-dynamics model** `q(s′ | s, z)` as the MI discriminator, and that model then supports
  **zero-shot model-based planning over skills** (MPC in skill space) on downstream tasks with no
  further learning. Skill discovery and world-model learning are the *same* objective here.
- [DIAYN](https://arxiv.org/abs/1802.06070) (Eysenbach et al., 2019) — the simpler ancestor:
  MI(s; z) via a learned discriminator; skills = "be distinguishable from each other". Known failure
  mode: rewards *static* distinguishability (posing, not doing); DADS's transition-based MI fixes this.
- [Empowerment — an Introduction](https://arxiv.org/pdf/1310.1863) (Salge, Glackin, Polani, 2013;
  concept from Klyubin et al. 2005) — empowerment = **channel capacity between n-step actions and
  resulting states**, "how much can I influence my future". In a *small discrete* world it is
  **exactly computable** (Blahut–Arimoto over the action→state channel; [UCT-accelerated for larger
  n](https://arxiv.org/pdf/1803.09866)) — no neural estimator, no variational bound, no backprop.
- ⭐ [Neuroevolution is a Competitive Alternative to Reinforcement Learning for Skill Discovery](https://arxiv.org/abs/2210.03516)
  (Chalumeau et al., ICLR 2023 spotlight) — **the keystone result for us**: across 8 algorithms
  (4 QD, 4 RL) and three evaluation axes (diversity, adaptation, hierarchical planning),
  quality-diversity neuroevolution gives **equal or better skill discovery than DIAYN/DADS-style
  RL**, with less hyperparameter sensitivity and better scalability. The whole
  MI-discriminator/backprop apparatus is *not required* to discover skill repertoires.

**Esper mapping.** Skill discovery does not force a gradient path into the engine: (a) the QD route
is ES-native (next finding); (b) in a small discrete grid world, empowerment is a *closed-form*
intrinsic fitness our existing ES can maximize directly — a candidate first intrinsic signal that is
exact, cheap, and has zero learned parts to go wrong; (c) DADS's deepest idea — **the skill
discriminator IS a world model, and planning over skills is free afterwards** — maps onto our
self-mod memories: a memory fit in-context to predict `next_grid = f(grid, action, z)` is
simultaneously the skill-quality signal and the planner's substrate.

### 2. Novelty, quality-diversity, archives — the ES-native intrinsic-motivation stack

- [NS-ES / NSR-ES / NSRA-ES](https://arxiv.org/abs/1712.06560) (Conti et al., NeurIPS 2018) —
  novelty search **inside OpenAI-style ES**: behavior characterization `b(π)` per candidate, novelty
  = mean kNN distance to an archive, and the ES gradient estimate simply weights perturbations by
  novelty (or a reward+novelty mix) instead of reward. **Drop-in for our `calculate_fitness`** — the
  update rule, workspaces, and SIMD loops are untouched; only the scalar being ranked changes.
- [MAP-Elites / QD](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)
  (Mouret & Clune 2015; Pugh et al. 2016) — grid archive over a hand- or learned-descriptor space,
  keep the best solution per cell; turns one optimization into a **repertoire** of diverse elites.
  The archive *is* the skill library — no z-conditioning needed at discovery time.
- [Go-Explore](https://arxiv.org/pdf/2004.12919) (Ecoffet et al., Nature 2021) — "first return, then
  explore": archive of **cells** (downscaled observations), return to a promising cell
  deterministically, explore from there. Solved all hard-exploration Atari games. The cell
  archive needs no neural network at all; its two named failure modes (**detachment** — forgetting
  reachable frontiers, **derailment** — failing to re-reach them) are the canonical checklist for
  any archive design.

**Esper mapping.** This is the family Vision B's first increment should come from, because it is
*literally our optimizer with a different scalar*: fitness ← novelty (archive-kNN over a behavior
descriptor, e.g. the downscaled final grid — a Go-Explore cell). MAP-Elites gives the persistent
repertoire — note this is exactly the **persistence/consolidation machinery rung #6 built the case
for**, re-hosted where it belongs (Vision B was already flagged in ROADMAP as rung #6's true home):
elites persist across ES runs the way rung #6 wanted decisions to persist across tasks.

### 3. UED and open-endedness: the task generator becomes the learner

- [POET](https://arxiv.org/abs/1901.01753) (Wang et al., 2019) — the seed paper. Co-evolves a
  *population* of (environment, agent) pairs; environments mutate, agents transfer between
  environments, a **minimal criterion** (not too easy, not too hard) curates. Notably POET's agents
  are optimized by **OpenAI-ES** — the original open-endedness loop is already ES-native.
- [PAIRED](http://aima.eecs.berkeley.edu/~russell/papers/neurips20-paired.pdf) (Dennis et al.,
  NeurIPS 2020) — formalizes UED as **minimax regret**: an adversary proposes environments
  maximizing (antagonist return − protagonist return); regret ≈ "solvable but not yet solved" =
  learnability. Needs a trained generator + two agents — heavy.
- [ACCEL](https://accelagent.github.io/) (Parker-Holder et al., 2022) — the frugal UED: **random
  edits** to existing levels + **regret-based curation** of a replay buffer; no learned generator at
  all. Evolution does the proposing, the curation does the intelligence. Caveat from
  [No Regrets](https://arxiv.org/html/2408.15099v1) (2024) and
  [TRACED](https://arxiv.org/html/2506.19997v3) (2025): regret proxies ≠ learnability — high-regret
  levels the agent can't improve on cause stagnation; prefer direct "success-rate near 50% /
  improving" learnability scores.
- [Open-Endedness is Essential for ASI](https://arxiv.org/abs/2406.04268) (Hughes et al., ICML 2024
  oral) — the formal definition Vision B needs: a system is open-ended w.r.t. an observer iff its
  artifacts are **novel** (unpredictable by the observer's model) *and* **learnable** (conditioning
  on history improves prediction). Novelty without learnability is noise (static on a TV screen is
  "novel" forever); learnability without novelty exhausts itself.
- [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (Zhang et al., 2025) — archive-based
  open-ended self-modification (agents rewrite their own code, keep "interestingly new" variants).
  Two takeaways: the **archive** (not a single lineage) is what makes it open-ended, and it
  **gamed its own metrics** (falsified test results, disabled its hallucination checks) — a concrete
  warning that self-generated objectives need an uncheatable outer metric.

**Esper mapping.** UED is the emergent replacement for the last hand-coded thing in our pipeline:
`synth_tasks.py`'s task families. An ACCEL-style loop — mutate task/world parameters, curate by "the
solver's held-out fit is neither 0 nor 1 and improving" — turns curriculum design into a measured
process, directly serving the convergence hypothesis (self-generated curriculum → primitives →
few-shot vocabulary). Hughes et al.'s novelty+learnability pair is the right *definition* of Vision
B's metric; the DGM failure is why the outer metric must stay Vision-A-style held-out transfer
(value #2, "honest measurement"), never a self-scored quantity.

### 4. World models and learning progress: the intrinsic signal with taste

- [World Models](https://arxiv.org/pdf/1803.10122) (Ha & Schmidhuber, 2018) — VAE + MDN-RNN world
  model (SGD-trained), but the **controller is ~870 params trained by CMA-ES** entirely inside the
  learned dream. Proof that "learn in imagination" does not require the *policy* to be a gradient
  citizen — only the world model was, and only because it was pixel-scale.
- [Dreamer v1–v3](https://arxiv.org/pdf/2402.16801) (Hafner et al., 2020–2023) — the seed line:
  RSSM latent dynamics, policy + value learned purely from imagined latent rollouts; V3 is stable
  across 150+ tasks with one hyperparameter set. Architecturally out of reach for us (deep SGD
  end-to-end), but the *decomposition* — dynamics model, then cheap policy search inside it — is the
  transferable content, and it is Ha & Schmidhuber's decomposition too.
- [Compression progress](https://people.idsia.ch/~juergen/creativity.html) (Schmidhuber 1990–2010) &
  [learning progress](http://www.pyoudeyer.com/oudeyerGottliebLopesPBR16Preprint.pdf) (Oudeyer et
  al.) — intrinsic reward = the **first derivative of prediction/compression error**, not its value.
  Chasing high error alone rewards noise (the "noisy TV" trap that also breaks naive RND/novelty);
  chasing error *improvement* selects exactly the learnable-but-not-yet-learned — the same
  novelty+learnability pair as Hughes et al., discovered twice, 15 years apart. Oudeyer's LP-based
  curricula are quasi-optimal for realistic learner models.

**Esper mapping.** We hold an unusually good hand here: **our ES fitness trajectory is a free
learning-progress meter.** When a self-mod memory is fit in-context to predict the world
(`next_grid` from `grid, action`), the *slope of its fitness curve* on a region/skill/task is
Oudeyer's LP signal with zero additional machinery — the engine already computes it every
iteration. A grid world model also needs no VAE: grids are already discrete and tiny, so the world
model is a self-mod memory over the same substrate `grid_substrate.mojo` reads — and "planning in
the dream" is running our existing per-task ES *inside* the learned model, i.e. the Ha–Schmidhuber
split with ES on both sides.

### 5. The POC environment: what the toy world must be

- [Craftax](https://arxiv.org/abs/2402.16801) (Matthews et al., ICML 2024) and
  [XLand-MiniGrid](https://pypi.org/project/xminigrid/) (NeurIPS 2024) — the field's answer to
  "Crafter/NetHack are too slow for small labs": rewrite tiny symbolic gridworlds for speed
  (250×–even more on GPU) and measure **deep exploration** (Craftax: achievement ladder) or **wide
  meta-generalization** (XLand-MG: millions of procedurally composed rule/goal tasks). Lesson: the
  env must be symbolic-tiny AND have *composable depth* (achievements/rules that chain), or nothing
  interesting can emerge to be measured.
- [Amorphous Fortress](https://arxiv.org/pdf/2312.02231) (2023) — QD over 0-player FSM micro-worlds;
  evidence that even degenerate-simple substrates yield behavioral diversity worth archiving.

**Esper mapping — the POC proposal ("the sandbox").** A pure-Mojo deterministic micro-world over our
existing grid types: an avatar on a small `ArcGrid` (≤16×16, ≤10 colours) with ~6 discrete actions
(move×4, paint, toggle/grab) and 2–3 fixed local dynamics rules that give it composable depth
(e.g. blocks fall, same-colour contact merges/annihilates — CA-flavoured, a dozen lines each, and
*parameterizable* so the rule-set itself becomes the UED mutation surface later). No reward channel
exists. This reuses `ArcGrid`, the substrate, the memory families as policy/world-model hosts, and
the GPU-batched ES unchanged; rollouts are microseconds, so commodity hardware is enough by
construction (value #5).

### Synthesis — the ES-native Vision B stack and its rungs

The literature sorts into a dependency-ordered ladder, each rung independently measurable, each
mechanism already proven derivative-free somewhere above:

1. **B-POC-1 — novelty-driven coverage (NS-ES + cells).** Policy = small memory fit by ES; fitness
   = archive-kNN novelty over Go-Explore-style cells (downscaled end-state grid). Metric: distinct
   cells reached vs. random-policy baseline. Proves the intrinsic-fitness seam in `Domain`
   (Example = trajectory, fitness = self-generated) with the fewest new parts.
2. **B-POC-2 — repertoire (MAP-Elites archive).** Same loop, persistent elite-per-cell archive =
   the skill library; optional exact empowerment (Blahut–Arimoto) as a second, learned-part-free
   intrinsic signal to compare against novelty. Metric: repertoire size × distinctness.
   (This is rung #6's persistence machinery, re-hosted.)
3. **B-POC-3 — world model + learning progress.** Self-mod memory fit in-context to predict
   `next_grid`; intrinsic reward = LP (fitness-slope), which auto-avoids the noisy-TV trap; explore
   where LP is highest (Go-Explore return-then-explore over the archive).
4. **B-POC-4 — the convergence test (the uncheatable outer metric).** Freeze the unsupervised
   phase's repertoire/prior; hand it to the Vision A few-shot fitter on *held-out grid→grid tasks
   defined inside the same world* (e.g. "reach this end-state from that start-state" as demo pairs).
   Metric: **held-out adaptation speed/success vs. cold start.** Open-ended exploration is declared
   useful exactly insofar as it buys few-shot generalization — Vision B scored in Vision A's
   currency, per the convergence hypothesis and value #2.
5. **B-POC-5 (later) — UED on the world itself (ACCEL-style).** Mutate the world's rule/topology
   parameters, curate by learnability (solver improving, success neither 0 nor 1) — the generator
   stops being hand-written; POET/ACCEL say curation, not a learned generator, is enough.

**What we deliberately do NOT import:** variational MI discriminators trained by SGD (QD replaces
them — Chalumeau), pixel-scale VAE/RSSM world models (grids are already symbolic), minimax-regret
adversary networks (ACCEL-style curation replaces them), and any self-scored success metric (DGM's
cautionary tale; the outer metric stays held-out transfer).

**Addendum (2026-07-10, B-POC-1 build).** Rung 1's phrase "proves the intrinsic-fitness seam in
`Domain`" resolved, on contact with the code, into a split: the **trajectory-Example seam** is
proven through `Memory.apply` (= a full deterministic rollout whose flat prediction is the
trajectory's behaviour characterization, so the unchanged generic ES core can fit a policy toward a
target end-state — B-POC-4's scoring path), while the **intrinsic fitness itself is driver-hosted**
— `Domain.distance` is a static, per-example, target-based method and cannot see a runtime novelty
archive, exactly as `meta_fit_selfmod`'s meta-fitness could not. Two empirical amendments from
calibration: (1) NS-ES over raw antithetic novelty differences barely moves (BC-distance scale
~0.1 and shrinking as the archive densifies) — **unit-std fitness shaping** of the coefficients is
load-bearing, quadrupling coverage at the same budget; (2) in an open world, **entropy is a strong
raw-visitation baseline** (a uniform random-ACTION controller touches more distinct per-tick cells
than anything else measured) — the honest gated claim for directed search is within the
deterministic policy class, on both visitation AND distinct end-states, the repertoire currency
B-POC-2 consumes.

**Addendum (2026-07-10, B-POC-2 build).** Rung 2 landed the same day (`src/map_elites.mojo`):
elite-per-cell over END-STATE Go-Explore cells, quality = directness (earliest settle tick). The
measured mutation-vs-emitter comparison — the rung's informative by-product — went to the
**ES-emitter** (4,317 stored replayable elites vs. 1,716 for pure mutation vs. 1,372 end-states
the NS-ES baseline ever touched, all at 13,205 rollouts): CMA-ME's insight that emitters beat
undirected mutation reproduces here derivative-free, but the dominant mechanism is *harvesting* —
every antithetic probe deposits into the map, so states the ES was already visiting stop being
thrown away. Calibration surfaced a clean asymmetry: the emitter's best step size is 4× NS-ES's
(α 0.8 vs 0.2) while the mutation arm's best σ is HALF the ES probe σ — products differ (map vs.
centers; a mutation is a move, a probe only measures). **Exact empowerment (Blahut–Arimoto) was
deferred** to a possible B-POC-2.5: the archive-vs-population claim stood alone. The repertoire is
deliberately in-memory; serialization is B-POC-4's seam.

**Addendum (2026-07-10, B-POC-2.5 build).** Exact empowerment measured (`src/empowerment.mojo`).
Theory note first: in a deterministic world **Blahut–Arimoto collapses** — channel capacity =
log₂(#distinct reachable states), so "exact empowerment" is exhaustive 6ⁿ enumeration + distinct-
state counting (n = 4 costs 0.7 ms; entirely affordable at this world size). Measured verdict, all
at 13,205 rollouts: an empowerment-only emitter stores 1,513 elites vs 447 random (3.4×, the gated
claim) but only **ties the novelty emitter at this RNG stream (1,513 vs 1,607)** — and the tie
exposed that the emitter family's repertoire is high-variance across streams (novelty drew 4,317
in `test_repertoire`'s position); cross-signal comparisons should quote spreads, not single draws.
Two qualitative findings: (1) **empowerment concentrates** — at reseed 25 it parks below random
(382 elites); frequent elite-restarts (reseed 5) are what convert an optionality-seeker into an
explorer, whereas novelty self-disperses by construction; (2) **the paint action flattens the
optionality landscape** (mean elite empowerment ≈ 7.9 bits in ALL arms), so empowerment buys no
concentration advantage in THIS world — revisit when B-POC-5 mutates the rules toward worlds with
genuinely constraining states. Cost honesty: the budget stayed rollout-denominated by design
choice; the enumeration spent ~24× the charged ticks (printed in-test as the uncharged caveat).

**Addendum (2026-07-10, B-POC-3 build).** Rung 3 landed (`src/world_model.mojo`): the world model
is fit through the UNCHANGED generic ES core (transitions = `ExamplePair`s of a `TransitionDomain`
— §4's "grid world model needs no VAE" confirmed in practice), and "our ES fitness trajectory is a
free LP meter" survived contact with three amendments:
1. **LP is currency-sensitive.** The fitness (MSE) slope correctly *diagnoses* learnability
   (novel = 22× mastered/unlearnable — gated), but as an *allocator* it chases the noisy TV: error
   scales differ 100× across regions and learning the noise's MEAN is a huge one-off MSE delta.
   Clone-probe LP fails differently (measures batch memorizability). The allocator that works is
   the windowed **discrete-score slope** (held-out changed-cell accuracy per region) — scale-free,
   and mean-learning never moves it. LP-guided beat uniform 0.362 vs 0.153 at equal budget.
2. **The deterministic noisy TV must be constructed carefully**: target-shuffling is LEARNABLE
   (shift-by-one = two gravity ticks ahead); the airtight device is CONTRADICTION (duplicate
   inputs, different targets — unfittable by any function, immune to memorization) for the static
   diagnostic, and pseudo-random targets (TV static) for the collection loop.
3. **The dynamics are selections, not values**: a tanh value head learns saturating gates
   (departures 113/113) but cannot emit a graded copy of a neighbour's colour (arrivals 0/115);
   the landed head is a learned softmax SELECTOR over {patch cells, brush, empty} — the AttnGather
   / Rung CF content-fetch expressivity lesson reproduced at 400-parameter scale, and a hint that
   selection should be the default read primitive across the engine.
"Return-then-explore over the archive" was realized as regions = DYNAMICS CONTEXTS (the gravity
directions — the UED surface); spatial return-then-explore over stored elites folds into
B-POC-4/5. Honest residual: rare agent-write events (paint) stay unlearned; paint-heavy data made
things worse — the fix is curriculum, which is what B-POC-5's UED is for.

**Addendum (2026-07-11, B-POC-4 build).** The convergence test landed (`src/transfer.mojo`) almost
entirely as plumbing — two existing core facts made it free: `fit_operator` takes a caller-prefilled
seed buffer (warm-start = a `memcpy`), and `fill_scale=0` fully freezes a parameter, so
`ComposeMemory` fits a schedule head over K FROZEN primitives through the unchanged `fit_operator`
(the ShapeMemory/GeomColor trick re-hosted). Measured verdicts, in Vision A's held-out currency:
1. **Retrieval is the lever, and it must be INDEXED.** The BC-nearest elite as ES seed reaches
   held-out goals **7.3× closer than cold-start**, while a random-elite seed is *worse* than cold
   (retrieval beats it 29.6×): a generic good init does not transfer — the index does. The
   convergence hypothesis's retrieval half is strongly confirmed, in-sandbox.
2. **Naïve composition DILUTES.** An unbiased schedule head loses even to cold on compositional
   goals — spreading weight across four primitives starves the best one. The fix is a slot-0 bias
   (start as ~nearest, explore mixing *from* that floor): 1.12× over nearest on two-phase goals,
   never underperforming it.
3. **Composition's margin is density-bounded.** At full repertoire density nearest already
   saturates (goals ~0.001–0.002 away), leaving composition little gap to close — the
   compositional signal is clearest exactly when the vocabulary is sparse.
Held-out discipline held physically: the `.rep` serialization splits the unsupervised phase from
the few-shot phase across disk (bit-identical reload, 100 % replay), goal cell keys are provably
absent from the repertoire, and exact-hit stays ≈ 0 so the graded BC-MSE is the honest metric.

**Addendum (2026-07-11, B-POC-5 build).** The final rung landed (`src/ued.mojo`): ACCEL as B-POC-3's
`train_lp_guided` with its **fixed context set replaced by a grown, mutation-fed, learnability-curated
replay buffer** — the generator is now emergent. Curation is the direct changed-cell-slope
learnability score (the No-Regrets caveat honoured). Result at equal budget: the curriculum trains a
world model to **0.154 held-out changed-cell vs 0.0 for domain randomization** (additive gate, à la
B-POC-3 — a ratio is degenerate at DR≈0). Two findings reshaped the plan: (1) **the density→
learnability relation inverts intuition** — moderate initial-grid density (0.08) is the learnable band
(0.44 changed in 400 focused iters), sparse (0.05) teaches ~nothing in-budget, so DR's wasteful tail
is the *sparse* end, not a dense one; (2) **DR churns to zero** because a fresh uniform-density level
each round, under `train_round`'s wide→fine anneal *restart*, lets a sparse round undo a moderate one
— curriculum consistency, not just allocation, is what ACCEL buys (the intrinsic-motivation lesson
once more: a learnable-band-held difficulty lets an incremental learner accumulate). Honest scope
booked: gravity (dir and rate) is held fixed — learning multiple gravity functions exceeds the
per-arm budget, and the world model has no grav_rate feature — so the config axis is effectively ~1-D
(gravity-event density) for a local-receptive model. **This is the wall the paint/optionality
residuals kept pointing at: the sandbox's single dynamics rule is the ceiling on curriculum richness;
the deferred next lever is constraining topology (walls) — new dynamics, not a new curator.** The
five-rung ES-native ladder is complete.

**Addendum (2026-07-14, T-POC-1 build — the first cross-world rung).** The deferred walls lever
landed as **world 2** (a reserved in-grid cell value; strict superset, all six Vision-B proofs
byte-identical) and B-POC-4's machinery was pointed across the world gap: W1 repertoire, few-shot
goals in walls worlds, fits rolling out in the goal's world (`make_demos` generalized,
byte-identical on B-POC-4). Measured verdicts, extending the B-POC-4 list:
1. **The index survives a world change; the basin does not.** B-POC-4's "retrieval must be
   indexed" sharpens: under new topology a *mismatched* skill is **4–8× worse than cold** (random
   arm) — negative transfer is the default — and BC-nearest retrieval rescues exactly that back to
   cold-parity (5.78×/5.19× vs random, seed 0), even landing exact goal keys in the confined world
   (5/24 vs cold ≤ 2/24, both seeds). But the same-world **7.3× warm-start advantage evaporates**
   (nearest-vs-cold ~1×, world/seed-noisy): the retrieved weights' fine structure encodes the OLD
   world's dynamics; only the index's relevance mapping generalizes.
2. **Native builds saturate confined worlds.** An equal-budget repertoire built in the room world
   covers its entire reachable goal space (zero doubly-held-out goals at 60k reference rollouts) —
   coverage-per-budget rises as topology shrinks the reachable set. Consequence: ceiling
   comparisons need doubly-held-out subsets, and "held-out" itself gets scarce in small worlds.
3. **Topology creates the optionality structure paint erased.** The B-POC-2.5 flat-landscape
   residual breaks locally: a sealed pocket collapses exact empowerment to log2(15) ≈ 3.9 bits vs
   the open ~8.3 — walls give empowerment (and any curiosity signal over it) something to see.
The next lever this prescribes is **adaptation, not retrieval**: re-grounding a retrieved skill in
the new world's dynamics (the B-POC-3 world model is the obvious mediator) to reclaim the basin
advantage across the gap — T-POC-2's shape, unscheduled.

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

---

## 2026-07-15 addendum (post-measurement) — T-POC-2: the dream orders but cannot pick

§4 of the 2026-07-14 addendum seeded T-POC-2 as "planning in the dream = our existing per-task ES
inside the learned model" — the Ha–Schmidhuber split with ES on both sides. It was built in full
and **STOPped at its pre-registered increment-0 gate**. The negative is specific, and it moves the
literature mapping rather than closing it.

**What was measured.** A dream whose every half is learned (grid model + pose/brush/write agent
models, blocked-move accuracy 1.0, nothing hand-coded) ranks candidate policies at Kendall tau
~0.4–0.7 against a ~0.1 scrambled-tail control — real signal — but its **top-1 pick** fails
decisively in columns (regret 3.9× vs a pre-registered 2.0×; 5.6× even in world 1's calibration).
Stated precisely: the GO required BOTH walls worlds and **room meets it** (tau 0.55, regret 1.97×),
so the negative rests on columns plus room's absence of margin — room straddles the bar across runs
(1.97×/2.06×/2.17×) rather than clearing it.
Because `fit_dream_policy` follows the dream's **argmax**, ordering alone cannot carry adaptation.

**Why this is an objective mismatch, not a capacity limit.** As the grid model's one-step
changed-cell accuracy ROSE (0.0 → 0.53 → 0.93), the dream's top-1 regret WORSENED (1.0× → 1.6× →
5.6×). At overall ~0.99 the model misplaces ~1% of cells per tick and compounds that over 64 ticks.
The identity collapse we spent the rung removing was, for ranking, partly PROTECTIVE — "nothing
moves" is safe to iterate. This is the literature's known failure and we rediscovered it from our
own numbers: **one-step teacher-forced accuracy does not predict rollout usefulness.** It is exactly
why the model-based RL line (PlaNet, Dreamer) trains on multi-step latent rollouts / scheduled
sampling rather than single-step prediction. Our §4 mapping ("planning in the dream = per-task ES")
holds — but it silently assumed a dream fit for the horizon it is planned over. It was not.

**Route A (scheduled).** Fit the model over K ticks of its OWN rollout, then re-read the same gate.
This is the field's answer to the exact defect we measured, and it is cheap in our setting (K× per
fitness eval; the ES core is untouched — a multi-step rollout is, again, "just another Domain").

**The identity basin as an objective pathology (transferable beyond this rung).** A transition moves
~0.2–0.7% of the grid; at ~200:1 static-to-changed, unweighted MSE makes "predict keep everywhere"
near-optimal, and the fit lands there on 1/3 of world-1 draws and ALL walls-world draws. This is the
class-imbalance problem in regression clothing, and the fix is the standard one: weight the changed
cells (`WeightedWMMemory`). It removes the basin outright (9/9 restarts; world 1 held-out changed
0.625 → 0.927 on the UNCHANGED unweighted ruler) and is backported into B-POC-3's gate 1, whose 0.4
bar rises to 0.75. Two wrong diagnoses died on the way and are worth remembering as a pattern: it
was NOT the sigma schedule (widening it CAUSES the collapse — a deterministic attractor, bit-
identical scores across seeds) and NOT the data scale (4× the transitions rescued nothing and
lowered density). When an ES fit stops learning a rare event, suspect the objective's balance before
the optimizer.

**Method lesson for future gates.** Match the metric to its CONSUMER (tau vs regret was the whole
finding); controls earn their cost (the scrambled tail proved no backdoor, the identity-grid arm
refuted our own leading hypothesis); and one run is not a reading (tau swung 0.31–0.57 for a single
world across runs at fixed bars; regret is a heavy-tailed mean of ratios — prefer medians).

## 2026-07-16 addendum (post-measurement) — Routes A and C: multi-step imagination vs. Intelligent Trial and Error

**Route A ran and confirmed the field's prescription is real but not free.** Fitting the world
model over K=8 ticks of its own rollout (the PlaNet/Dreamer multi-step lesson, §above) fixed the
measured defect exactly where per-tick error signal is dense: columns' top-1 regret 3.93× →
1.44×/1.05× across two seeds, both pre-registered bars cleared. Where the signal is thinnest (room,
transition-event density 0.0027 — the sparsest of the three worlds) it did NOT transfer at either
seed (4.64×/2.39× vs the 2.0× bar). The literature's multi-step fix presumes there is per-tick
residual to chase; at low enough event density the K-step objective has almost nothing to grade.
The gate stays a booked negative; the room-specific data-scale lever is named but unscheduled.

**Route C is Cully, Clune, Tarapore & Mouret 2015 ("Robots that can adapt like animals" —
Intelligent Trial and Error), mapped one-to-one onto our stack, minus the surrogate.** ITE's
ingredients: (1) a behaviour-performance map built offline by MAP-Elites = our B-POC-2/4 repertoire
(`EliteMap`); (2) at deployment, when the world has changed (their broken leg = our walls
topology), a handful of REAL trials on the robot, ranked by measured performance = our
`pool_trials` (one deterministic 64-tick rollout per candidate — their Bayesian-optimization
surrogate exists to handle noisy, expensive trials, and is degenerate at pool size 9 with
noise-free rollouts, so we drop it); (3) commit to the best-performing behaviour = our few-shot fit
seeded from the trial winner. Two honest divergences from ITE: we FIT from the winner rather than
just executing it (ITE is pure selection; we are selection + tuning), and our budget accounting
overcharges the trials (one full ES iteration dropped from the ITE arm's fit) so the comparison can
never be won with compute.

**What Route C measured (the durable findings).** (1) **The cross-world index is good but not
best-of-pool**: in the home world the BC-nearest elite IS the true best of the 9-member pool
(oracle probe gain exactly 1.0, zero picks changed — a clean control), while across the world gap
the true best differs from nearest-1 for 67–83% of goals (median pre-fit headroom 1.7–6.2×). BC
distance ranks the OLD world's behaviours; the new world's dynamics re-shuffle them, and only real
contact with the new world sees the re-shuffle. (2) **A few real trials reclaim much of that gap,
and the advantage SURVIVES the few-shot fit** — the first seed-quality effect that does: ITE beats
cold 1.89–3.49× and a uniform pool pick 2.65–5.23× at both seeds in both walls worlds, and improves
exact goal hits everywhere (room 6–8/24 vs nearest-1's 4/24). T-POC-1's wash-out finding
(nearest-vs-cold ~1×) is thereby refined: the fit washes out a MARGINAL seed advantage, not a real
one — nearest-1's seed was near-parity with cold, the trial-picked seed is not. (3) Still a
PARTIAL: room's margin over nearest-1 straddles its 1.3× bar across seeds (1.21×/2.07×), so under
the AND rule the GO is unearned — booked with the positives gated and the GO asserted FALSE.

**The axis-level reading after three rungs (dream STOP → Route A partial → Route C partial).**
Selection-by-real-contact is cheap, model-free, and insensitive to the event-density axis that
limits learned models here — but it is bounded by what the pool already contains; it cannot
re-ground a skill, only find the least-mismatched one. The learned-model route re-grounds in
principle but pays in event density. The two failure modes are complementary, which is itself
evidence for the ITE paper's architecture (map + trials + model only where trials are expensive):
in a world where real ticks are cheap, imagination must beat 576 ticks of ground truth to earn its
keep — and at this scale it does not.
