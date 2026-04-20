# aether-aec

**Agent Education Calibration — Compiled Runtime Verification for LLM Agent Responses**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![Dependencies: stdlib + optional anthropic](https://img.shields.io/badge/deps-stdlib%20only-brightgreen.svg)]()
[![Tests: 93/93](https://img.shields.io/badge/tests-93%2F93%20passing-brightgreen.svg)]()
[![Paper: Zenodo](https://img.shields.io/badge/Paper-Zenodo%20DOI-blue.svg)](https://zenodo.org/record/19212829)

---

## What is AEC?

AEC answers one question about every LLM agent response:

> **Does this agent know what it just claimed — or is it making things up?**

Most evaluation frameworks answer this with embedding similarity or LLM-as-judge scoring — probabilistic, non-deterministic, and insufficient for regulated environments. AEC takes a different approach: it **compiles** your knowledge graph into an executable verification program at load time, then runs every response through it using set intersection.

No embeddings. No vector databases. No GPU. No external services required.

The result: per-statement accountability with 0.000 score variance across runs, sub-millisecond verification for deterministic layers, and structured gap output that feeds directly into an autonomous self-education loop.

**AEC moves LLM reliability toward an enforceable contract.**

---

## How It Works

```
Knowledge Graph (JSON-LD)
        │
        ▼  compile_kg() — once at load
 ┌─────────────────────────────────────┐
 │  Compiled Structure                 │
 │  • pattern detectors per node       │
 │  • AntiPattern blacklist            │
 │  • edge policy checkers             │
 └─────────────────────────────────────┘
        │
        ▼  verify() — per response
 ┌──────────────────────────────────────────────┐
 │  L0: Factual extraction    <1ms  deterministic│
 │  L1: Compiled patterns     <1ms  deterministic│
 │  L2: Type-driven LLM       2-8s  optional     │
 │  L3: Edge policy traverse  <1ms  deterministic│
 └──────────────────────────────────────────────┘
        │
        ├─ score ≥ 0.80 → PASS
        └─ score < 0.80 → structured gap list → education loop
```

AEC treats your knowledge graph as a compiled constraint program executed against natural language outputs. The node's `@type` determines which verification question fires: Rules trigger compliance checking, AntiPatterns trigger violation detection, Techniques trigger application confirmation. Edge policies catch compositional violations that token matching alone cannot detect.

**AEC does not require the full AETHER framework.** Point it at any typed JSON-LD knowledge graph.

---

## Installation

```bash
git clone https://github.com/jeff0926/aether-aec.git
cd aether-aec

# Core verification: no external dependencies
# Layer 2 LLM verification (optional)
pip install anthropic

# REST/CAP adapters (optional)
pip install flask

# Redis queue backend (optional)
pip install redis

# Negation detection (optional — regex fallback always available)
pip install spacy && python -m spacy download en_core_web_sm

# Copy config template
cp .env.example .env
```

Core verification (Layers 0, 1, 3) runs on Python stdlib only. Layer 2 requires the Anthropic SDK or any OpenAI-compatible endpoint. For air-gapped deployments, local models via [Ollama](https://ollama.ai) work with Layer 2.

---

## Quick Start

### CLI

```bash
# Verify any text against any JSON-LD knowledge graph
python cli.py verify "Jefferson was born in 1743 in Virginia" \
  --kg examples/jefferson/kg.jsonld

# Expected output:
# Score: 1.000 | PASS
# Grounded: 2 | Ungrounded: 0 | Persona: 0

# Test an AntiPattern violation
python cli.py verify "Use Inter for body text" \
  --kg examples/frontend-design/kg.jsonld

# Expected output:
# Score: 0.000 | FAIL
# Gaps: AntiPattern violation — overused_fonts
```

### Python

```python
from aec_concept import compile_kg
from kg import load_kg, get_nodes

# Compile your knowledge graph once at load time
kg = load_kg("examples/frontend-design/kg.jsonld")
compiled = compile_kg(get_nodes(kg))

# Verify any response
from aec_concept import verify_concept
result = verify_concept("Use CSS variables for design tokens", compiled)

print(f"Score:  {result['score']}")    # 0.0 – 1.0
print(f"Passed: {result['passed']}")   # True / False
print(f"Gaps:   {result['gaps']}")     # structured gap list
```

### REST API

```bash
# Start the REST adapter
python adapters/rest.py --port 8080

# Verify via HTTP
curl -X POST http://localhost:8080/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Use Inter for body text",
    "kg_path": "examples/frontend-design/kg.jsonld"
  }'

# Response: {"score": 0.0, "passed": false, "gaps": [...]}

# Health check
curl http://localhost:8080/health
# Response: {"status": "ok", "version": "1.0.0"}
```

### MCP Tool

```bash
# List available tools
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | \
  python adapters/mcp.py

# Call aec_verify
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
  "name":"aec_verify",
  "arguments":{
    "text":"Use CSS variables for design tokens",
    "kg_path":"examples/frontend-design/kg.jsonld"
  }}}' | python adapters/mcp.py
```

### SAP CAP

```bash
# Start the CAP adapter
python adapters/cap.py --port 8080

# OData action call
curl -X POST http://localhost:8080/AECService/verify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Use CSS variables for design tokens",
    "kg_path": "examples/frontend-design/kg.jsonld"
  }'

# Response: {"d": {"score": 1.0, "passed": true, "gaps": "[]"}}
```

---

## Adapters

AEC is transport-agnostic. The core verification engine never changes — adapters are protocol translation only.

| Adapter | File | Input | Output | Use Case |
|---|---|---|---|---|
| **CLI** | `cli.py` | Command line args | Formatted text | Development, scripting |
| **REST** | `adapters/rest.py` | HTTP POST JSON | HTTP JSON response | Any HTTP client, SAP CAP |
| **MCP** | `adapters/mcp.py` | MCP JSON-RPC (stdio) | MCP tool result | Claude Code, Copilot, any MCP client |
| **SAP CAP** | `adapters/cap.py` | OData action / JSON POST | OData result / JSON | SAP CAP services, BTP |
| **Redis Queue** | `adapters/redis_queue.py` | AEC failure event | Redis queue entry | Multi-agent, cross-service education |

Each adapter is 15–214 lines. Zero business logic — only protocol translation.

### REST Endpoints

```
POST /verify          — verify single text
POST /verify-batch    — verify list of texts
GET  /health          — health check
```

### SAP CAP OData Definition

```cds
// Equivalent CAP CDS
action verify(text: String, kg_path: String, threshold: Decimal)
  returns { score: Decimal, passed: Boolean, gaps: String };
```

### Redis Queue

Replaces file-based `education-queue.json` for multi-agent and cross-service deployments. Implements the same interface as `education.py` — swap the backend via `AEC_QUEUE_BACKEND=redis` in `.env`.

```python
from adapters.redis_queue import RedisQueue

queue = RedisQueue(capsule_id="jefferson")
queue.queue_failure(query, response, aec_result)
pending = queue.get_pending()
```

---

## Verification Layers

| Layer | Method | Speed | Dependencies |
|---|---|---|---|
| **L0** | Factual extraction (numbers, dates, magnitudes, named entities) | <1ms | stdlib |
| **L1** | Compiled concept matching (Rules, Techniques, AntiPatterns) | <1ms | stdlib |
| **L2** | Type-driven LLM verification (ambiguous cases only, max 3 calls) | 2–8s | anthropic SDK (optional) |
| **L3** | Edge policy traversal (avoids, requires, contradicts) | <1ms | stdlib |

Layers 0, 1, and 3 are **fully deterministic** — identical results across all runs. Layer 2 is **bounded** — at most 3 LLM calls per response, 0 in the common case where L1 resolves everything.

When Layer 2 is unavailable, AEC degrades gracefully to a fully deterministic verifier using L0, L1, and L3.

---

## The Scoring Formula

```
score = grounded / (grounded + ungrounded)
```

**Persona statements** — those with no verifiable content — are excluded from both numerator and denominator. An agent's stylistic framing does not penalize its score. Only verifiable claims are evaluated.

Default threshold: **0.80** — at least 80% of verifiable content must be grounded.

---

## Gap List Output

When a response scores below threshold, AEC returns a **structured gap list** — a machine-readable specification of exactly what the agent does not know, consumed directly by the education loop:

```json
{
  "score": 0.143,
  "passed": false,
  "grounded_statements": 1,
  "ungrounded_statements": 8,
  "gaps": [
    {
      "text": "The Purchase was negotiated by Robert Livingston",
      "node_id": "concept:louisiana_purchase",
      "violation_type": "ungrounded_claim"
    },
    {
      "text": "Napoleon agreed to sell the territory",
      "node_id": "concept:louisiana_purchase",
      "violation_type": "ungrounded_claim"
    }
  ],
  "method": "concept_3layer_compiled"
}
```

In the full [AETHER framework](https://github.com/jeff0926/aether/tree/v1.0.0), this gap list feeds directly into the autonomous self-education loop — the agent researches its gaps, validates new knowledge through AEC, and integrates survivors into the knowledge graph without human intervention.

---

## Examples

Five example knowledge graphs covering all agent types and all verification behaviors:

| Example | Agent Type | Nodes | What it Tests |
|---|---|---|---|
| `frontend-design` | Skill (Anthropic) | 73 | AntiPattern blacklist, Rule compliance, L3 edge traversal |
| `docx` | Skill (Anthropic) | 75 | Document structure rules, Technique verification |
| `jefferson` | Scholar | 51 | Factual grounding, numeric/date extraction (L0) |
| `ceo` | Executive Advisor | 60 | Scope boundary enforcement via avoids/contradicts edges |
| `domain-sap-cap` | Domain | 15 | SAP-specific Rule compliance |

```bash
# Run any example
python cli.py verify "Your text here" --kg examples/frontend-design/kg.jsonld
python cli.py verify "Your text here" --kg examples/jefferson/kg.jsonld
python cli.py verify "Your text here" --kg examples/ceo/kg.jsonld
```

---

## Knowledge Graph Format

AEC expects typed [JSON-LD](https://json-ld.org/) knowledge graphs. The `@type` field drives the verification engine — it is not optional.

| Type | AEC Behavior |
|---|---|
| `aether:Rule` | Compliance checking — does the statement FOLLOW or VIOLATE? |
| `aether:AntiPattern` | Violation detection — does the statement USE forbidden content? |
| `aether:Technique` | Application confirmation — does the statement APPLY or REFERENCE? |
| `aether:Concept` | Relevance matching — is this concept present? |
| `aether:Tool` | Usage detection — is this tool referenced? |
| `aether:Trait` | Tone matching — does this characteristic appear? |

---

## Use Cases

**Enterprise AI compliance.** Every agent response is per-statement auditable against your authoritative knowledge base. Score variance: 0.000. Suitable for SOC 2, FDA, and EU AI Act environments where probabilistic evaluation is insufficient.

**Scope boundary enforcement.** Use `avoids` and `contradicts` edges to enforce role boundaries structurally — not via fragile prompt engineering. A CEO agent that starts prescribing Kubernetes configurations fails AEC even if every technical claim is factually correct.

**SAP CAP integration.** The CAP adapter wraps AEC as an OData-compatible external service. CAP calls `POST /AECService/verify`, AEC returns score and gap list in OData format. Plugs into any existing CAP service with ~40 lines of adapter code.

**Self-educating agents.** Pair with the full [AETHER framework](https://github.com/jeff0926/aether/tree/v1.0.0) to convert every AEC failure into autonomous knowledge acquisition via the six-stage education loop.

**Any LLM pipeline.** AEC works standalone against any typed JSON-LD KG. No agent framework required. `compile_kg()` is the only prerequisite.

---

## Performance

Measured on MacBook Pro M-series (16GB RAM, 8-core CPU), single-threaded Python:

| Operation | Time |
|---|---|
| `compile_kg()` — 73 nodes | 0.62ms |
| L1 verification — per statement | 0.1–0.3ms |
| Blacklist check | <0.05ms |
| L3 edge traversal — per statement | <0.5ms |
| Full verification — 6 statements, no L2 | <2ms |

---

## Running Tests

```bash
# Install test dependencies
pip install flask fakeredis

# Run all test suites
python tests/test_aec_standalone.py   # 26/26
python tests/test_rest_adapter.py     # 23/23
python tests/test_mcp_adapter.py      # 22/22
python tests/test_redis_queue.py      # 22/22

# Total: 93/93 passing
```

---

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# LLM (required for Layer 2)
ANTHROPIC_API_KEY=your_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6

# Verification
AEC_THRESHOLD=0.80
AEC_LAYER2_ENABLED=true

# REST adapter
AEC_REST_PORT=8080
AEC_REST_HOST=0.0.0.0

# Education queue backend
AEC_QUEUE_BACKEND=file          # file | redis
REDIS_URL=redis://localhost:6379
```

---

## Paper

**Agent Education Calibration: A Compiled Runtime Verification Engine for Knowledge-Grounded AI Agent Responses**  
Jeff Conn, 864 Zeros LLC, April 2026  
DOI: [10.5281/zenodo.19212829](https://doi.org/10.5281/zenodo.19212829)

Companion framework paper (AETHER):  
[AETHER: Self-Educating Agent Skills through Compiled Knowledge Graph Verification](https://doi.org/10.5281/zenodo.19212829)

---

## Relationship to AETHER

AEC is the verification engine at the core of the [AETHER framework](https://github.com/jeff0926/aether/tree/v1.0.0). It can be used standalone (this repo) or as part of the full AETHER capsule system.

```
AETHER (full framework)           github.com/jeff0926/aether
├── aether-aec    ← this repo — standalone verification engine
├── aether-engram ← working memory / KG persistence
└── aether-ui     ← agent projection layer
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*864 Zeros LLC · jeff.m.conn@gmail.com*  
*"The model did not change. The skill did."*