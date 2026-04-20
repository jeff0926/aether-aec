# AEC Example Knowledge Graphs

Five example KGs covering all agent types and verification behaviors.

| Example | Agent Type | What it Tests |
|---|---|---|
| frontend-design | Skill (Anthropic) | AntiPattern blacklist, Rule compliance, edge traversal |
| docx | Skill (Anthropic) | Document structure rules, Technique verification |
| jefferson | Scholar | Factual grounding, numeric/date extraction |
| ceo | Executive Advisor | Scope boundary enforcement via avoids/contradicts edges |
| domain-sap-cap | Domain | SAP-specific Rule compliance |

Run any example:
```bash
python cli.py verify "Your text here" --kg examples/frontend-design/kg.jsonld
```
