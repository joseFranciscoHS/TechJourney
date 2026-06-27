---
name: dwmri-implementation-planning
description: >-
  Interview the user relentlessly to plan a new DWMRI implementation before
  coding. Use when the user wants to plan, design, or stress-test a new
  feature, model, baseline, eval harness, or experiment workflow in this repo,
  or uses plan/planning/grill trigger phrases for implementation work.
---

# DWMRI implementation planning

## When to use

Apply when the user wants to **plan or stress-test** a new implementation before writing code — not when they are ready to implement immediately.

## Workflow

1. **Plan mode only** — do not edit code or create files. Explore the repo to answer what you can without asking.
2. **One question at a time** — wait for the user's answer before the next question. Multiple questions at once is bewildering.
3. **Recommended answer** — with each question, state your recommended choice and why (grounded in repo patterns when possible).
4. **Walk the decision tree** — cover every relevant branch in [reference.md](reference.md). Skip branches that clearly do not apply; mark them as N/A with a one-line reason.
5. **Codebase first** — if a question can be answered by reading the repo, read it (start with [AGENTS.md](../../../AGENTS.md), [src/AGENTS.md](../../../src/AGENTS.md), [experiments/AGENTS.md](../../../experiments/AGENTS.md)) instead of asking.
6. **J-invariance branch** — when the plan involves self-supervised, blind-spot, or hybrid training, read and apply [j-invariance-dwmri-reviewer](../j-invariance-dwmri-reviewer/SKILL.md) during planning (assumptions, J-set, threats).
7. **Completion criterion** — do not finish until every relevant branch has an explicit decision or is marked **deferred** with a reason. Vague "TBD" without rationale is not done.
8. **Deliverable** — when grilling is complete, produce the implementation plan using the template in [reference.md](reference.md).

## Repo anchors

| Area | Read first |
| --- | --- |
| Package layout, module pattern | [src/AGENTS.md](../../../src/AGENTS.md) |
| Manifests, driver, baselines | [experiments/AGENTS.md](../../../experiments/AGENTS.md), [BASELINES_REGISTRY.md](../../../experiments/BASELINES_REGISTRY.md) |
| Agent conventions | [AGENTS.md](../../../AGENTS.md) |
| Tech debt marking | [.cursor/rules/tech-debt-todos.mdc](../../../.cursor/rules/tech-debt-todos.mdc) |
| J-invariance / hybrid methods | [j-invariance-dwmri-reviewer](../j-invariance-dwmri-reviewer/SKILL.md) |

## After planning

Suggest the user switch to **Agent mode** to implement. Optionally run **dwmri-handoff** if the session will continue in a fresh chat.
