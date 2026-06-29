# Nested Learning & HOPE — distilled summary

Distilled from "Nested Learning: The Illusion of Deep Learning Architecture" (Behrouz,
Razaviyayn, Zhong, Mirrokni — Google Research, NeurIPS 2025), available at
<https://abehrouz.github.io/files/NL.pdf>, so future work can skip re-reading the 52-page
paper. Covers the conceptual spine (§1–§9.1); the late
experiment tables and appendices (B/C, exact ablations) are not reproduced here — go to the
PDF for those. This is the theory behind Esper's architecture; for how it maps onto the
codebase and the roadmap, see the "Direction" section of `CLAUDE.md`.

---

## TL;DR

A deep model is **not** one architecture trained by one optimizer. It is an **ordered system
of nested optimization problems**, each with its own *context flow* (the data it compresses)
and its own *update frequency*. "Deep learning" is the flattened, lossy view of that system —
the "illusion" of the title. Three contributions follow:

1. **Optimizers are associative memories.** SGD-momentum, Adam, AdaGrad, Muon are all memory
   modules that *compress gradients* into their state — so they can be made more expressive.
2. **Self-modifying modules.** A module can generate its own keys/values/learning-rates and
   learn (part of) its own update rule — true in-context self-improvement.
3. **Continuum Memory System (CMS).** Replace "short-term vs long-term memory" with a
   *spectrum* of memory blocks updated at different frequencies. Combining (2)+(3) gives
   **HOPE**, a continual-learning module.

Guiding definition (§3.1): **"Memory is a neural update caused by an input; learning is the
process of acquiring effective memory."**

---

## 1. Associative memory — the atom (§3.1)

An associative memory `M` maps a set of keys `K` to values `V`, found by minimizing a loss
that measures mapping quality: `M* = argmin_M  L̂(M(K); V)`. Keys/values need not be tokens —
they can be gradients, sub-sequences, error signals, anything. Everything else in the paper is
built from this one operator. **Memorization** = forming the mapping; **learning** = acquiring
an *effective* (generalizing) mapping.

## 2. The Nested Learning paradigm (§3.2–3.3)

- **Nested system (Def 3):** an ordered set of `K` levels; level `k` is a set of optimization
  problems, each minimizing its objective over its parameters on its context `C`, by gradient
  descent. Components are ordered by **update frequency** — higher level = lower frequency.
- **NSAM (Def 4):** the case where every level's optimization problem is an associative memory.
  Modern architectures + their optimizers are instances of NSAM.
- **Stacking levels** adds a genuinely new axis of *computational depth* (beyond stacking
  layers): higher-order in-context learning, latent computation, more expressive optimizers.
- **Knowledge transfer between levels (§3.3)** is a first-class design choice, with several
  forms: direct parametric/non-parametric conditioning, backprop across levels, **initialization
  (MAML-style: the slow level meta-learns the fast level's init)**, and weight/context
  generation (hypernetworks, learned optimizers). *Two design choices define a module: (1) the
  optimization problems and their frequencies, (2) the knowledge transfer between levels.*

## 3. Optimizers ARE associative memories (§4)

- **Backprop (§4.1):** training a net by backprop is a *compression* process — each layer is an
  associative memory mapping its input `x̂` to its local surprise/error signal `δ`. "A network
  trained with backprop memorizes how surprising its predicted outputs are."
- **Momentum (§4.2):** the momentum term is an associative memory that *compresses past
  gradients* into its parameters; GD-with-momentum is a 2-level process (inner learns momentum,
  outer uses it). Adam, RMSProp, AdaGrad, Lion, Shampoo are all instances. Standard momentum is
  a weak low-pass filter (β=0.9 → last ~43 gradients carry >99% weight), which causes
  catastrophic forgetting of the gradient subspace on orthogonal continual-learning tasks (§4.3).
- **More expressive optimizers (§4.4–4.5):** give momentum a real objective (L2 regression →
  **Delta Momentum**, delta-rule), an MLP-valued memory (**DMGD**), higher-order feature maps,
  or a nonlinearity (Newton–Schulz → recovers **Muon**). **Delta Gradient Descent (DGD)** is a
  state-dependent gradient rule (uses the *current weight state*, not just the current input) —
  better for correlated/sequential data where the i.i.d. assumption breaks.

## 4. Architectures as NSAM (§5)

- **Softmax attention** = the *non-parametric* optimal solution of an L2 regression objective
  over the context (Nadaraya–Watson). It is a "perfect memory" (caches all tokens, frequency ∞)
  — so it won't be beaten by parametric memory at scale on the *same* objective, but it has
  limited computational depth and static projections.
- **Linear attention / RNNs (Hebbian, Delta, Oja, Omega rules)** = *parametric* solutions of
  the same mapping objective with constant-size state — i.e. fast-weight memories.
- **Takeaway:** modern architectures only *look* heterogeneous; under NL they're all feedforward
  memory blocks differing only in level, objective, and update rule. The heterogeneity is "the
  illusion."

## 5. Continuum Memory System (§7)

Generalize long-/short-term memory to a **chain/spectrum of MLP memory blocks**, each updated
every `C^(ℓ)` steps (its frequency). High-frequency blocks = fast adaptation / short-term;
low-frequency blocks = persistent knowledge. A Transformer's MLP is the special case `k=1`.
Variants: nested (level `s` meta-learns level `s+1`'s init), sequential, head-wise (parallel
blocks aggregated). **Why it fights forgetting:** when one block is overwritten, the old
knowledge still lives in other-frequency blocks and can circle back via cross-level transfer.
Updates are cheap (only blocks at their scheduled time update) and parallelizable.
**M3 optimizer** = Adam + Muon + CMS, a proof-of-concept multi-timescale-momentum optimizer.

## 6. Self-modifying Titans + HOPE (§8)

- **Limitation of Transformers:** projections `W_k/W_v/W_q` are fixed after pre-training, so
  contextualization is bounded by the first level — they can't modify themselves in-context.
- **Self-modifying memory (Eqs 83–97):** a module that **generates its own** keys, values,
  learning-rates `η`, retention/forget gates `α`, *and its own values* per input, and optimizes
  its memory with an internal objective — it controls its own learning process. Trained with a
  chunk-wise parallelizable rule (DGD with weight decay).
- **HOPE** = self-modifying Titans **followed by a CMS chain**: a small-but-expressive
  self-modifying block (fast adaptation) feeding a multi-frequency memory stack (persistent
  knowledge). The two are complementary (small capacity + rich rule vs. large capacity + simple
  rule).

## 7. Revisited terms (§6)

- **In-context learning** is not emergent — it's a *direct consequence* of having multiple
  frequency levels; any level adapting to its context is doing ICL.
- **Pre-training** is just the lowest-frequency level (ICL with the whole dataset as context).
- **Test-time training / test-time memorization** = parametric in-context learning whose
  acquired knowledge vanishes when the context is removed.
- **No train/test boundary** in a neural learning module — it only has two states: receiving
  input, or being an isolated learning system.
- *Models have more parameters than we knew:* momentum/optimizer state and RNN hidden state are
  knowledge stores too, not just the backprop-trained weights.

## 8. Experimental claims (§9, high level)

HOPE-enhanced architectures beat in-context learning, EWC, and an external in-context learner
(InCA) on **class-incremental continual learning** (CLINC/Banking/DBpedia) and on **long-context
understanding** (RULER needle-in-a-haystack, LongHealth, QASPER). Adding memory *levels*
monotonically improves ICL accuracy and lowers perplexity. (Exact numbers/ablations: see the
PDF figures 6–7 and the language-modeling tables.)

---

## How this grounds Esper

| Paper concept | Esper realization |
|---|---|
| Slow (low-frequency) level = meta-learned **prior / init** (MAML-style transfer, §3.3) | `HopeNode.slow` weights + the L2 anchor in `update_fast_weights` |
| Fast (high-frequency) memory adapted **in-context** (§5, §8) | `HopeNode.fast` weights, fit per-task |
| Optimizer as a derivative-free learning rule (NL says learning ≠ backprop) | the antithetic-sampling **Evolution Strategy** (`evolve_fast_weights`) |
| "Memory is a neural update caused by an input; learning is acquiring *effective* memory" | the engine must **learn the grid→grid transformation** from demonstrations — never be handed a symbolic DSL |
| Continuum Memory System — a spectrum of frequencies | north-star: stack fast/slow into a multi-frequency memory; Phase B of the roadmap |
| Self-modifying memory (generates its own values/rates) | the fully-emergent end state after the structured-operator "training wheels" come off |

The roadmap's spine — *fast weights = a learned operator fit in-context by ES, slow weights =
the meta-learned prior, no hand-coded DSL, measured by held-out generalization* — is the direct
application of NL's two-timescale, self-improving-memory thesis to ARC.
