"""
AEC - Aether Entailment Check
Deterministic verification gate for response validation against KG subgraph.
"""

import re
import json

DEFAULT_THRESHOLD = 0.8


def split_statements(response: str) -> list[str]:
    """Split response into sentences. Simple regex, no NLP."""
    if not response or not response.strip():
        return []

    text = " ".join(response.split())

    # Protect abbreviations
    for abbr in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Inc.", "Jr.", "Sr.", "vs.", "etc.", "e.g.", "i.e."]:
        text = text.replace(abbr, abbr.replace(".", "<<DOT>>"))

    # Split on sentence endings followed by capital letter
    parts = re.split(r'[.!?]+\s+(?=[A-Z])', text)

    statements = []
    for part in parts:
        part = part.replace("<<DOT>>", ".").strip().rstrip(".!?").strip()
        if len(part) > 10:
            statements.append(part)
    return statements


def _extract_values(text: str) -> list[tuple[str, any, str]]:
    """Extract verifiable values: (original, normalized, type)."""
    values = []

    # Numbers with magnitude words (million, billion, trillion)
    magnitude = {"million": 1e6, "billion": 1e9, "trillion": 1e12, "thousand": 1e3}
    mag_pattern = r'\$?\s*(\d+(?:[.,]\d+)?)\s*(million|billion|trillion|thousand)\b'
    for m in re.finditer(mag_pattern, text, re.I):
        try:
            num = float(m.group(1).replace(",", ""))
            mult = magnitude[m.group(2).lower()]
            values.append((m.group(0).strip(), num * mult, "number"))
        except (ValueError, KeyError):
            pass

    # Numbers (integers and decimals)
    for m in re.finditer(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b', text):
        try:
            values.append((m.group(1), float(m.group(1).replace(",", "")), "number"))
        except ValueError:
            pass

    # Percentages
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\b', text, re.I):
        try:
            values.append((m.group(0), float(m.group(1)), "percentage"))
        except ValueError:
            pass

    # Years (1700-2099)
    for m in re.finditer(r'\b(1[7-9]\d{2}|20\d{2})\b', text):
        values.append((m.group(0), m.group(0), "date"))

    # Full dates: Month Day, Year
    months = "January|February|March|April|May|June|July|August|September|October|November|December"
    for m in re.finditer(rf'({months})\s+(\d{{1,2}}),?\s+(1[7-9]\d{{2}}|20\d{{2}})', text):
        month_map = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                     "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
        month_num = month_map[m.group(1).lower()]
        normalized = f"{m.group(3)}-{month_num:02d}-{int(m.group(2)):02d}"
        values.append((m.group(0), normalized, "date"))

    # Capitalized names (skip common words and pronouns)
    skip = {"The", "This", "That", "These", "Those", "It", "He", "She", "They", "We", "You", "I",
            "His", "Her", "Its", "Their", "Our", "Your", "My", "And", "But", "Or", "So", "If"}
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        name = m.group(1)
        if name not in skip:
            values.append((name, name, "name"))

    return values


def _flatten_kg(kg_nodes: list[dict]) -> dict[str, list]:
    """Flatten KG nodes into {key: [values]} for searching."""
    result = {}

    def extract(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith("@"):
                    if k == "@id":
                        result.setdefault("id", []).append(str(v))
                else:
                    extract(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, prefix)
        else:
            result.setdefault(prefix, []).append(obj)

    for node in kg_nodes:
        extract(node)
    return result


def _match_in_kg(value, value_type: str, kg_flat: dict, tolerance: float = 0.01) -> tuple[bool, str | None]:
    """Check if value exists in flattened KG. Returns (matched, key)."""
    for key, kg_vals in kg_flat.items():
        for kv in kg_vals:
            # String match (case-insensitive, partial for names)
            if isinstance(value, str) and isinstance(kv, str):
                if value.lower() == kv.lower() or value.lower() in kv.lower() or kv.lower() in value.lower():
                    return True, key

            # Numeric match with tolerance
            if isinstance(value, (int, float)):
                kv_num = kv
                if isinstance(kv, str):
                    try:
                        kv_num = float(kv.replace(",", ""))
                    except (ValueError, AttributeError):
                        continue
                if isinstance(kv_num, (int, float)):
                    if abs(value - kv_num) <= abs(kv_num * tolerance) + 0.001:
                        return True, key

    return False, None


def deterministic_gate(statement: str, kg_nodes: list[dict]) -> dict:
    """
    Deterministic verification: extract values from statement, match against KG.
    Returns {matched: bool, entity: str|None, method: str, values_found: int}.
    """
    if not statement or not kg_nodes:
        return {"matched": False, "entity": None, "method": "no_input", "values_found": 0}

    values = _extract_values(statement)
    if not values:
        return {"matched": False, "entity": None, "method": "no_values", "values_found": 0}

    kg_flat = _flatten_kg(kg_nodes)
    matches = []

    for original, normalized, vtype in values:
        matched, key = _match_in_kg(normalized, vtype, kg_flat)
        if matched:
            matches.append((vtype, original, key))

    if matches:
        return {
            "matched": True,
            "entity": matches[0][1],
            "method": f"deterministic_{matches[0][0]}",
            "values_found": len(values),
            "matches": len(matches),
        }
    return {
        "matched": False,
        "entity": None,
        "method": "deterministic_unmatched",
        "values_found": len(values),
        "matches": 0,
    }


def verify(response: str, kg_nodes: list[dict], threshold: float = DEFAULT_THRESHOLD,
           compiled_kg: dict = None, llm_fn=None) -> dict:
    """
    Verify response against KG subgraph.

    Statement categories:
    - GROUNDED: has extractable values that match KG
    - UNGROUNDED: has extractable values that DON'T match KG
    - PERSONA: no extractable values (qualitative/interpretive)

    Score = grounded / (grounded + ungrounded)
    PERSONA statements excluded from ratio - they're expected behavior.

    If compiled_kg is provided (typed KG with Rule/Technique/AntiPattern nodes),
    always runs concept matching and merges results. Concept takes priority over
    factual name-only matches; factual takes priority for numbers/dates.

    If llm_fn is provided, Layer 2 type-driven LLM verification runs for
    ambiguous/persona statements against Rules, AntiPatterns, and Techniques.
    """
    statements = split_statements(response)

    if not statements:
        return {
            "score": 0.0, "threshold": threshold, "passed": False,
            "total_statements": 0, "grounded_statements": 0,
            "ungrounded_statements": 0, "persona_statements": 0, "persona_ratio": 0.0,
            "statements": [], "gaps": [{"text": response[:100], "reason": "no_statements"}],
        }

    # Phase 1: Run factual AEC on all statements
    factual_results = []
    for stmt in statements:
        gate = deterministic_gate(stmt, kg_nodes)
        if gate["values_found"] == 0:
            factual_results.append({"text": stmt, "grounded": None, "method": "persona", "category": "persona"})
        elif gate["matched"]:
            factual_results.append({"text": stmt, "grounded": True, "method": gate["method"], "category": "grounded"})
        else:
            factual_results.append({"text": stmt, "grounded": False, "method": gate["method"], "category": "ungrounded"})

    # Phase 2: If typed KG, always run concept matching and merge
    concept_applied = False
    if compiled_kg is not None:
        concept_applied = True
        from aec_concept import tokenize, match_statement

        results = []
        gaps = []
        grounded, ungrounded, persona = 0, 0, 0

        for factual in factual_results:
            stmt = factual["text"]
            factual_method = factual.get("method", "")

            # Run concept matching on this statement (Layer 1 + optional Layer 2)
            stmt_tokens = tokenize(stmt)
            concept_result = match_statement(stmt_tokens, stmt, compiled_kg, llm_fn=llm_fn)

            # Merge logic: decide which result to use
            # Hard facts (numbers, dates, percentages) → factual wins
            # Name-only matches, persona, ungrounded → concept can override
            use_factual = False
            if factual["category"] == "grounded":
                # Check if factual matched on hard facts (not just names)
                hard_fact_methods = ("deterministic_number", "deterministic_date", "deterministic_percentage")
                if factual_method in hard_fact_methods:
                    use_factual = True  # Hard facts take priority

            if use_factual:
                # Keep factual result
                results.append(factual)
                grounded += 1
            elif concept_result['category'] == 'concept_grounded':
                # Concept found a match (Layer 1 or Layer 2 LLM)
                grounded += 1
                llm_res = concept_result.get('llm_result')
                if llm_res and llm_res.get('category') == 'grounded':
                    # Layer 2 LLM found a grounded match
                    results.append({
                        "text": stmt,
                        "grounded": True,
                        "method": llm_res.get('method', 'llm_grounded'),
                        "category": "grounded",
                        "matched_node": llm_res.get('node_id', ''),
                        "matched_label": llm_res.get('node_label', ''),
                        "reasoning": llm_res.get('reasoning', ''),
                    })
                else:
                    # Layer 1 compiled pattern match
                    top = concept_result['matches'][0]
                    results.append({
                        "text": stmt,
                        "grounded": True,
                        "method": f"concept:{top['node_type']}",
                        "category": "grounded",
                        "matched_node": top['node_id'],
                        "matched_label": top['label'],
                        "coverage": top['coverage'],
                    })
            elif concept_result['category'] == 'antipattern_violation':
                # Concept found a violation (Layer 1 blacklist)
                v = concept_result['violation']
                ungrounded += 1
                results.append({
                    "text": stmt,
                    "grounded": False,
                    "method": "antipattern_violation",
                    "category": "ungrounded",
                    "matched_node": v['node_id'],
                    "matched_label": v.get('label', ''),
                    "violation_terms": v['hits'],
                })
                gaps.append({
                    "text": f"VIOLATION: '{', '.join(v['hits'])}' matches antipattern",
                    "reason": "antipattern_violation",
                    "node_id": v['node_id'],
                })
            elif concept_result['category'] == 'concept_ungrounded':
                # Layer 2 LLM found a violation
                llm_res = concept_result.get('llm_result', {})
                ungrounded += 1
                results.append({
                    "text": stmt,
                    "grounded": False,
                    "method": llm_res.get('method', 'llm_violation'),
                    "category": "ungrounded",
                    "matched_node": llm_res.get('node_id', ''),
                    "matched_label": llm_res.get('node_label', ''),
                    "reasoning": llm_res.get('reasoning', ''),
                })
                gaps.append({
                    "text": f"LLM VIOLATION: {llm_res.get('reasoning', '')}",
                    "reason": "llm_violation",
                    "node_id": llm_res.get('node_id', ''),
                })
            else:
                # Concept didn't match - fall back to factual or persona
                if factual["category"] == "grounded":
                    # Factual had a name match, concept didn't find anything stronger
                    # Keep as grounded but note it was name-only
                    grounded += 1
                    results.append({**factual, "method": f"{factual_method}(weak)"})
                elif factual["category"] == "ungrounded":
                    # Factual found values but no match, concept also no match
                    ungrounded += 1
                    results.append(factual)
                    gaps.append({"text": stmt, "reason": "values_not_in_kg"})
                else:
                    # Pure persona
                    persona += 1
                    results.append(factual)

    else:
        # No compiled KG - use factual results only
        results = factual_results
        gaps = []
        grounded = sum(1 for r in results if r["category"] == "grounded")
        ungrounded = sum(1 for r in results if r["category"] == "ungrounded")
        persona = sum(1 for r in results if r["category"] == "persona")
        for r in results:
            if r["category"] == "ungrounded":
                gaps.append({"text": r["text"], "reason": "values_not_in_kg"})

    # Calculate final score
    verifiable = grounded + ungrounded
    score = grounded / verifiable if verifiable > 0 else 1.0
    persona_ratio = persona / len(statements) if statements else 0.0

    return {
        "score": round(score, 3),
        "threshold": threshold,
        "passed": score >= threshold,
        "total_statements": len(statements),
        "grounded_statements": grounded,
        "ungrounded_statements": ungrounded,
        "persona_statements": persona,
        "persona_ratio": round(persona_ratio, 3),
        "statements": results,
        "gaps": gaps,
        "concept_applied": concept_applied,
    }


if __name__ == "__main__":
    # Test with mix of grounded, ungrounded, and persona statements
    test_response = """
    Thomas Jefferson was born on April 13, 1743 in Virginia.
    He was a brilliant thinker and visionary leader.
    Jefferson served as the 3rd President of the United States.
    His ideas continue to inspire people around the world.
    He authored the Declaration of Independence in 1776.
    """
    test_kg = [
        {"@id": "person:jefferson", "rdfs:label": "Thomas Jefferson", "birth_year": 1743, "birth_date": "1743-04-13", "role": "3rd President"},
        {"@id": "doc:declaration", "rdfs:label": "Declaration of Independence", "year": 1776, "author": "Thomas Jefferson"},
    ]

    result = verify(test_response, test_kg)
    print(f"Score: {result['score']} (threshold: {result['threshold']})")
    print(f"Passed: {result['passed']}")
    print(f"Grounded: {result['grounded_statements']}, Ungrounded: {result['ungrounded_statements']}, Persona: {result['persona_statements']}")
    print(f"Persona ratio: {result['persona_ratio']} (expected ~20% for good responses)")
    print(f"Gaps: {len(result['gaps'])}")
    print("\nStatements:")
    for s in result["statements"]:
        cat = s["category"].upper()
        print(f"  [{cat:10}] {s['text'][:55]}...")
