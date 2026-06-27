---
name: dwmri-handoff
description: Compact the current conversation into a handoff document for another agent to continue DWMRI work.
argument-hint: "What will the next session be used for?"
disable-model-invocation: true
---

# DWMRI session handoff

Write a handoff document summarising the current conversation so a fresh agent can continue the work.

## Output location

Save to `.cursor/handoffs/YYYY-MM-DD_<slug>.md` (create the directory if it does not exist). Use a short slug derived from the focus of the next session.

## Document structure

```markdown
# Handoff: [title]

## Next session focus
[What the next agent should do first]

## Context
[Compressed background: goal, constraints, key decisions already made]

## Current state
[What exists now: files touched, plans written, blockers]

## References (do not duplicate)
- [path or URL] — one-line description

## Suggested skills
- dwmri-implementation-planning — if more planning is needed
- j-invariance-dwmri-reviewer — if self-supervised / hybrid / J-invariance applies
- .cursor/rules/dl-expert.mdc — architecture and training choices

## Sensitive data
[Note if anything was redacted]
```

## Rules

- **Do not duplicate** content already captured in other artifacts (PRDs, Cursor plans, ADRs, issues, commits, diffs, checklists). Reference them by path instead.
- **Redact** API keys, passwords, tokens, and personally identifiable information.
- If the user passed arguments, treat them as the description of what the next session will focus on and tailor the doc accordingly.
