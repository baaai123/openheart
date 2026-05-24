# Domain Docs

Single-context repo.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root
- **`docs/adr/`** at the repo root — read ADRs that touch the area you're about to work in

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront.

## File structure

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-*.md
│   └── 0002-*.md
└── src/
```

## Use the glossary's vocabulary

When naming a domain concept, use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 — but worth reopening because…_
