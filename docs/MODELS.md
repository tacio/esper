# Model selection & subagent delegation (Esper)

How to spend model capability well on this project. The principle: **match the model to
the judgment the task actually requires**, and **push read-only fan-out into subagents**
so the main thread stays focused and cheap. This is guidance for any Claude Code session
working in Esper; the `.claude/agents/*` definitions encode it via their `model:` frontmatter.

## The heuristic

| Task class | Model | Why |
|---|---|---|
| Deterministic / mechanical: idiom lint, grep-and-report, locating code, file audits | **Haiku** | No novel reasoning; speed and cost dominate. |
| Scoped, well-specified build: implement a change whose design is already decided; routine doc upkeep | **Sonnet** | Competent execution of a known plan; the ambiguity is already resolved. |
| Judgment / novelty: emergence & architecture decisions, gate/metric/control design, discipline review, wall-survey and reroute, ambiguous multi-file design | **Opus** | These are where a wrong call is expensive — the honest-measurement and no-DSL values live here. |

When unsure, ask: *would a wrong answer here quietly corrupt the evidence trail or the
architecture?* If yes → Opus. If it's just typing out a decided change → Sonnet. If it's
"find/scan/report" → Haiku.

## Subagent delegation

- **Fan out read-only exploration** to `esper-explorer` (Haiku) — "where is X / how does Y
  work / which tests cover Z". Keep the answer in the main thread; keep the file dumps out.
- **Batch idiom audits** go to `mojo-linter` (Haiku) — the whole-repo counterpart to the
  per-edit hook.
- **Keep the emergence verdict and the gate design on Opus.** Delegate discipline review
  to `discipline-reviewer` (Opus) for any diff touching the ES core, a memory family, the
  `Domain`/`Memory` seams, or a claimed result; design gates with `gate-designer` (Opus).
- **Doc upkeep** (JOURNAL append, ROADMAP status reflection) goes to `roadmap-scribe`
  (Sonnet).
- Prefer **one capable agent over many shallow ones**; spawn a subagent when the work is
  a genuine read-only fan-out or a separable specialist task, not to split a single line
  of reasoning. A subagent starts cold — give it the module-map context it needs.

## Notes
- Esper is CPU/GPU-frugal by design; the model-cost frugality here is the same value
  applied to the tooling.
- The agents' `model:` frontmatter is the source of truth for defaults; this doc explains
  the *why* so a human (or a session) can override deliberately.
