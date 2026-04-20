"""
AEC Concept Layer 1 - Compiled Deterministic Matching
Adds concept-level statement matching using COMPILED token patterns from KG node labels.
The KG compiles into detector functions at load time. Runtime matching is set intersection.
"""

import re
from collections import Counter

# Stopwords removed from token sets
STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
             'it', 'its', 'this', 'that', 'these', 'those', 'not', 'no', 'do', 'does',
             'did', 'has', 'have', 'had', 'will', 'would', 'could', 'should', 'may',
             'can', 'if', 'then', 'than', 'so', 'as', 'from', 'into', 'up', 'out',
             'about', 'over', 'such', 'each', 'every', 'all', 'any', 'both', 'few',
             'more', 'most', 'other', 'some', 'very', 'just', 'also', 'like', 'use',
             'using', 'used',
             # Question words and pronouns
             'how', 'what', 'when', 'where', 'why', 'who', 'which', 'whom', 'whose',
             'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
             'she', 'her', 'they', 'them', 'their'}

# Type-specific configuration for matching
TYPE_CONFIG = {
    'rules':        {'match_threshold': 0.50, 'weight': 1.0},
    'techniques':   {'match_threshold': 0.55, 'weight': 1.0},
    'antipatterns': {'match_threshold': 0.40, 'weight': 1.0},
    'concepts':     {'match_threshold': 0.30, 'weight': 0.5},
    'tools':        {'match_threshold': 0.60, 'weight': 0.3},
    'traits':       {'match_threshold': 0.70, 'weight': 0.2},
}

# Layer 2: Type-driven verification operators
TYPE_OPERATORS = {
    'Rule': {
        'strategy': 'compliance',
        'llm_prompt_template': (
            'You are a strict compliance evaluator.\n\n'
            'RULE: "{label}"\n'
            'STATEMENT: "{statement}"\n\n'
            'Does the statement FOLLOW, VIOLATE, or have NO RELATION to the rule?\n\n'
            'Reasoning steps:\n'
            '1. What does the rule require or forbid?\n'
            '2. What does the statement do?\n'
            '3. Is there direct compliance or violation?\n\n'
            'If the statement is merely "good advice" but not explicitly supported by the rule, classify as UNRELATED.\n\n'
            'Return ONLY one JSON object:\n'
            '{{"classification": "FOLLOW"|"VIOLATE"|"UNRELATED", "reasoning": "one sentence"}}'
        ),
    },
    'AntiPattern': {
        'strategy': 'violation_detect',
        'llm_prompt_template': (
            'You are a strict violation detector.\n\n'
            'FORBIDDEN PATTERN: "{label}"\n'
            'STATEMENT: "{statement}"\n\n'
            'Does the statement USE or RECOMMEND what the forbidden pattern describes?\n\n'
            'If the statement explicitly uses, recommends, or includes elements described '
            'in the forbidden pattern, classify as VIOLATES.\n'
            'If the statement avoids or does not mention the forbidden elements, classify as CLEAN.\n'
            'If there is no connection, classify as UNRELATED.\n\n'
            'Return ONLY one JSON object:\n'
            '{{"classification": "VIOLATES"|"CLEAN"|"UNRELATED", "reasoning": "one sentence"}}'
        ),
    },
    'Technique': {
        'strategy': 'application',
        'llm_prompt_template': (
            'You are a strict technique evaluator.\n\n'
            'TECHNIQUE: "{label}"\n'
            'STATEMENT: "{statement}"\n\n'
            'Does the statement APPLY, REFERENCE, or have NO RELATION to this technique?\n\n'
            'APPLY means the statement describes using this technique or recommends it.\n'
            'REFERENCE means it mentions the technique in passing.\n'
            'UNRELATED means no meaningful connection.\n\n'
            'Return ONLY one JSON object:\n'
            '{{"classification": "APPLY"|"REFERENCE"|"UNRELATED", "reasoning": "one sentence"}}'
        ),
    },
    # Concept, Tool, Trait — no LLM, handled by Layer 1 only
    'Concept': {'strategy': 'relevance', 'llm_prompt_template': None},
    'Tool': {'strategy': 'usage', 'llm_prompt_template': None},
    'Trait': {'strategy': 'tone', 'llm_prompt_template': None},
}


def tokenize(text: str) -> set:
    """Extract content words as lowercase set, minus stopwords."""
    return set(re.findall(r'\b\w+\b', text.lower())) - STOPWORDS


def dice_bigram(s1: str, s2: str) -> float:
    """Sørensen-Dice coefficient using word bigrams. Secondary check."""
    def bigrams(text):
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 2:
            return Counter()
        return Counter(f"{words[i]} {words[i+1]}" for i in range(len(words) - 1))
    bg1, bg2 = bigrams(s1), bigrams(s2)
    intersection = sum((bg1 & bg2).values())
    total = sum(bg1.values()) + sum(bg2.values())
    return (2 * intersection) / total if total > 0 else 0.0


# -----------------------------------------------------------------------------
# Layer 2: Type-Driven Verification
# -----------------------------------------------------------------------------

def get_type_operator(node_type: str) -> dict | None:
    """Get the verification operator for a node type."""
    for type_key, operator in TYPE_OPERATORS.items():
        if type_key in node_type:
            return operator
    return None


def _extract_json_block(text: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    import json
    # Try direct parse first
    text = text.strip()
    if text.startswith('{'):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try to find JSON in markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find any JSON object
    match = re.search(r'\{[^{}]*"classification"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def llm_classify_statement(statement: str, node: dict, llm_fn) -> dict:
    """
    Use LLM to classify statement against a specific KG node.
    Only called when Layer 1 is ambiguous.

    Returns: {'classification': str, 'category': str, 'reasoning': str, 'node_id': str, ...}
    """
    node_type = node.get('@type', '')
    node_id = node.get('@id', '')
    label = node.get('rdfs:label', '')

    operator = get_type_operator(node_type)
    if not operator or not operator.get('llm_prompt_template'):
        return {
            'classification': 'UNRELATED',
            'category': 'persona',
            'reasoning': 'No LLM operator for this type',
            'node_id': node_id,
            'node_label': label,
            'method': 'no_operator',
        }

    prompt = operator['llm_prompt_template'].format(
        label=label,
        statement=statement,
    )

    try:
        result = llm_fn(prompt)
        text = result.get('text', result) if isinstance(result, dict) else str(result)

        parsed = _extract_json_block(text)
        if isinstance(parsed, dict):
            classification = parsed.get('classification', 'UNRELATED').upper()
            reasoning = parsed.get('reasoning', '')

            # Normalize classification to category
            if classification in ('FOLLOW', 'APPLY', 'REFERENCE', 'CLEAN'):
                category = 'grounded'
            elif classification in ('VIOLATE', 'VIOLATES'):
                category = 'ungrounded'
            else:
                category = 'persona'

            return {
                'classification': classification,
                'category': category,
                'reasoning': reasoning,
                'node_id': node_id,
                'node_label': label,
                'method': f'llm_type:{node_type}',
            }
    except Exception:
        pass

    return {
        'classification': 'UNRELATED',
        'category': 'persona',
        'reasoning': 'LLM call failed',
        'node_id': node_id,
        'node_label': label,
        'method': 'llm_error',
    }


def type_driven_check(statement: str, stmt_tokens: set, layer1_result: dict,
                      compiled: dict, llm_fn) -> dict:
    """
    Layer 2: Type-driven verification for ambiguous Layer 1 results.

    Only runs when:
    - Layer 1 classified as 'concept_persona' (no match found)
    - AND there are ambiguous candidates or we find candidates via broader search

    Returns updated result dict with LLM classification.
    """
    if not llm_fn:
        return layer1_result

    # Collect candidates for LLM checking
    candidates = []

    # From Layer 1 ambiguous matches (dice 0.3-0.7)
    for match in layer1_result.get('ambiguous', []):
        dice = match.get('dice', 0)
        if 0.3 <= dice <= 0.7:
            candidates.append(match)

    # If no ambiguous candidates, do broader search against Rules/AntiPatterns/Techniques
    if not candidates:
        for detector in compiled['detectors']:
            if detector['node_type'] not in ('rules', 'antipatterns', 'techniques'):
                continue
            # Check keyword coverage
            overlap = stmt_tokens & detector['patterns']
            coverage = len(overlap) / detector['pattern_count'] if detector['pattern_count'] > 0 else 0
            # Broad threshold for LLM candidates - let LLM decide relevance
            dice = dice_bigram(statement, detector['label'])
            if dice > 0.08 or coverage > 0.12 or len(overlap) >= 1:
                candidates.append({
                    'node_id': detector['node_id'],
                    'label': detector['label'],
                    'node_type': detector['node_type'],
                    'dice': dice,
                    'coverage': coverage,
                    'node': detector['node'],
                })
        # Keep top 3 by combined score (dice + coverage)
        candidates = sorted(candidates, key=lambda c: c.get('dice', 0) + c.get('coverage', 0), reverse=True)[:3]

    if not candidates:
        return layer1_result

    # Run LLM classification against candidates
    best_result = None
    for candidate in candidates:
        node = candidate.get('node', {})
        if not node or '@type' not in node:
            continue

        llm_result = llm_classify_statement(statement, node, llm_fn)

        if llm_result['category'] == 'grounded':
            best_result = llm_result
            break  # First grounded match wins
        elif llm_result['category'] == 'ungrounded' and not best_result:
            best_result = llm_result  # Record violation, but keep checking for grounded

    if best_result:
        return {
            'category': f"concept_{best_result['category']}",
            'matches': layer1_result.get('matches', []),
            'violation': best_result if best_result['category'] == 'ungrounded' else None,
            'ambiguous': [],
            'llm_result': best_result,
        }

    return layer1_result


def compile_kg(kg_nodes: list) -> dict:
    """
    Compile KG into executable detector structures.
    Called ONCE at capsule load. All runtime matching uses the compiled output.

    Layer 1: Node pattern detectors and antipattern blacklist
    Layer 3: Edge policy checkers (avoids, requires, contradicts)

    Returns:
        {
            'detectors': [...],      # compiled pattern detectors
            'blacklist': set,        # anti-pattern forbidden tokens
            'blacklist_map': dict,   # token -> node_id mapping for violation attribution
            'edge_policies': [...],  # Layer 3: compiled edge policies
            'node_lookup': dict,     # Layer 3: @id -> node for edge resolution
        }
    """
    detectors = []
    blacklist = set()
    blacklist_map = {}  # token -> {'node_id': ..., 'label': ...}

    # Layer 3: Build node lookup for edge resolution
    node_lookup = {node.get('@id', ''): node for node in kg_nodes if node.get('@id')}

    for node in kg_nodes:
        ntype = node.get('@type', '')
        label = node.get('rdfs:label', '')
        node_id = node.get('@id', '')
        if not label or not node_id:
            continue

        # Classify node type
        if 'Rule' in ntype:
            node_type = 'rules'
        elif 'AntiPattern' in ntype:
            node_type = 'antipatterns'
        elif 'Technique' in ntype:
            node_type = 'techniques'
        elif 'Concept' in ntype:
            node_type = 'concepts'
        elif 'Tool' in ntype:
            node_type = 'tools'
        elif 'Trait' in ntype:
            node_type = 'traits'
        else:
            continue

        config = TYPE_CONFIG.get(node_type, {'match_threshold': 0.5, 'weight': 0.5})

        # COMPILE: Extract content tokens from label
        patterns = tokenize(label)

        detectors.append({
            'node_id': node_id,
            'label': label,
            'node_type': node_type,
            'patterns': patterns,
            'pattern_count': len(patterns),
            'threshold': config['match_threshold'],
            'weight': config['weight'],
            'node': node,
        })

        # COMPILE: Anti-pattern blacklist
        if node_type == 'antipatterns':
            # Extract specific terms from parentheses: "Overused fonts (Inter, Roboto)" -> {inter, roboto}
            paren_terms = re.findall(r'\(([^)]+)\)', label)
            for match in paren_terms:
                for term in re.split(r'[,/]', match):
                    term = term.strip().lower()
                    if len(term) > 2:
                        blacklist.add(term)
                        blacklist_map[term] = {'node_id': node_id, 'label': label}

    # Layer 3: COMPILE edge policies
    edge_policies = []
    EDGE_PREDICATES = {
        'skill:avoids': 'avoids',
        'skill:requires': 'requires',
        'skill:contradicts': 'contradicts',
    }

    for node in kg_nodes:
        source_id = node.get('@id', '')
        source_label = node.get('rdfs:label', '')
        source_patterns = tokenize(source_label)

        if not source_id or not source_patterns:
            continue

        for prop_key, edge_type in EDGE_PREDICATES.items():
            targets = node.get(prop_key)
            if not targets:
                continue
            if isinstance(targets, str):
                targets = [targets]

            for target_id in targets:
                if not isinstance(target_id, str):
                    continue
                target_node = node_lookup.get(target_id)
                if not target_node:
                    continue

                target_label = target_node.get('rdfs:label', '')
                target_patterns = tokenize(target_label)

                # Extract blacklist tokens from target (for avoids edges)
                target_blacklist = set()
                if edge_type == 'avoids':
                    # Get parentheses terms
                    paren_terms = re.findall(r'\(([^)]+)\)', target_label)
                    for match in paren_terms:
                        for term in re.split(r'[,/]', match):
                            term = term.strip().lower()
                            if len(term) > 2:
                                target_blacklist.add(term)
                    # Also add significant words from target patterns
                    target_blacklist |= {w for w in target_patterns if len(w) > 3}

                edge_policies.append({
                    'source_id': source_id,
                    'source_label': source_label,
                    'source_patterns': source_patterns,
                    'target_id': target_id,
                    'target_label': target_label,
                    'target_patterns': target_patterns,
                    'target_blacklist': target_blacklist,
                    'edge_type': edge_type,
                })

    return {
        'detectors': detectors,
        'blacklist': blacklist,
        'blacklist_map': blacklist_map,
        'edge_policies': edge_policies,
        'node_lookup': node_lookup,
    }


# -----------------------------------------------------------------------------
# Layer 3: Compiled Edge Policy Execution
# -----------------------------------------------------------------------------

def execute_edge_policies(stmt_tokens: set, matched_node_ids: list, compiled: dict) -> list:
    """
    Execute compiled edge policies for matched nodes.

    For each edge policy where the source node was matched by Layer 1/2:
      - avoids: check if statement also contains target's forbidden tokens → VIOLATION
      - contradicts: check if statement also matches target patterns → CONTRADICTION
      - requires: informational only in v1 (logged but not scored)

    Returns: List of violation dicts
    """
    violations = []
    matched_set = set(matched_node_ids)
    edge_policies = compiled.get('edge_policies', [])

    for policy in edge_policies:
        # Only fire policies where source node was matched by Layer 1/2
        if policy['source_id'] not in matched_set:
            continue

        if policy['edge_type'] == 'avoids':
            # Check if statement contains target's forbidden tokens
            hits = stmt_tokens & policy['target_blacklist']
            if hits:
                violations.append({
                    'type': 'edge_avoids_violation',
                    'source_id': policy['source_id'],
                    'source_label': policy['source_label'],
                    'target_id': policy['target_id'],
                    'target_label': policy['target_label'],
                    'path': f"{policy['source_id']} --avoids--> {policy['target_id']}",
                    'evidence': list(hits),
                    'severity': 'high',
                    'explanation': (
                        f"Statement addresses '{policy['source_label']}' "
                        f"but contains '{', '.join(hits)}' which is avoided per "
                        f"'{policy['target_label']}'"
                    ),
                })

        elif policy['edge_type'] == 'contradicts':
            # Check if statement matches BOTH source and target patterns
            target_overlap = stmt_tokens & policy['target_patterns']
            coverage = len(target_overlap) / len(policy['target_patterns']) if policy['target_patterns'] else 0
            if coverage > 0.4:
                violations.append({
                    'type': 'edge_contradicts_violation',
                    'source_id': policy['source_id'],
                    'source_label': policy['source_label'],
                    'target_id': policy['target_id'],
                    'target_label': policy['target_label'],
                    'path': f"{policy['source_id']} --contradicts--> {policy['target_id']}",
                    'evidence': list(target_overlap),
                    'severity': 'medium',
                    'explanation': (
                        f"Statement matches '{policy['source_label']}' "
                        f"which contradicts '{policy['target_label']}'"
                    ),
                })

        elif policy['edge_type'] == 'requires':
            # Informational only in v1 — log but don't score
            # Future: check if target technique/rule is present in full response
            pass

    return violations


def check_violation(stmt_tokens: set, compiled: dict) -> dict | None:
    """Check if statement tokens hit the anti-pattern blacklist. O(1) per token."""
    hits = stmt_tokens & compiled['blacklist']
    if not hits:
        return None

    # Find best matching anti-pattern node
    node_scores = {}
    for hit in hits:
        mapped = compiled['blacklist_map'].get(hit)
        if mapped:
            nid = mapped['node_id']
            node_scores[nid] = node_scores.get(nid, 0) + 1

    if not node_scores:
        return None

    best_id = max(node_scores, key=node_scores.get)
    best_node = compiled['blacklist_map'].get(list(hits)[0], {})

    return {
        'node_id': best_id,
        'label': best_node.get('label', ''),
        'hits': list(hits),
        'type': 'antipattern_violation',
    }


def match_statement(stmt_tokens: set, statement: str, compiled: dict, llm_fn=None) -> dict:
    """
    Match a single statement against compiled KG detectors.

    Layer 1: set intersection (O(1) per detector) + Dice bigram for ambiguous
    Layer 2: LLM type-driven verification (only for persona results with candidates)

    Returns:
        {
            'category': 'concept_grounded' | 'antipattern_violation' | 'concept_persona',
            'matches': [...],
            'violation': {...} or None,
            'ambiguous': [...]  # candidates for Layer 2 LLM check
            'llm_result': {...} or None  # Layer 2 result if invoked
        }
    """
    # Step 1: Anti-pattern violation check (highest priority, fastest)
    violation = check_violation(stmt_tokens, compiled)
    if violation:
        return {
            'category': 'antipattern_violation',
            'matches': [],
            'violation': violation,
            'ambiguous': [],
        }

    # Step 2: Run compiled detectors via set intersection
    matches = []
    ambiguous = []

    for detector in compiled['detectors']:
        if detector['pattern_count'] == 0:
            continue

        # PRIMARY: Set intersection
        overlap = stmt_tokens & detector['patterns']
        coverage = len(overlap) / detector['pattern_count']

        if coverage >= detector['threshold']:
            # High confidence match
            matches.append({
                'node_id': detector['node_id'],
                'label': detector['label'],
                'node_type': detector['node_type'],
                'coverage': round(coverage, 3),
                'overlap_tokens': list(overlap),
                'weight': detector['weight'],
                'method': 'compiled_pattern',
            })
        elif coverage >= detector['threshold'] * 0.5:
            # Ambiguous range - secondary Dice check
            dice = dice_bigram(statement, detector['label'])
            if dice >= 0.3:
                ambiguous.append({
                    'node_id': detector['node_id'],
                    'label': detector['label'],
                    'node_type': detector['node_type'],
                    'coverage': round(coverage, 3),
                    'dice': round(dice, 3),
                    'weight': detector['weight'],
                    'node': detector['node'],
                    'method': 'dice_ambiguous',
                })

    # Step 3: Determine category
    if matches:
        strong = [m for m in matches if m['weight'] >= 0.5]
        if strong:
            return {
                'category': 'concept_grounded',
                'matches': sorted(matches, key=lambda m: m['coverage'], reverse=True),
                'violation': None,
                'ambiguous': ambiguous,
            }

    # Layer 1 result: persona (no high-confidence match)
    layer1_result = {
        'category': 'concept_persona',
        'matches': matches,
        'violation': None,
        'ambiguous': sorted(ambiguous, key=lambda a: a.get('dice', 0), reverse=True)[:3],
    }

    # Step 4: Layer 2 — Type-driven LLM verification for persona results
    if llm_fn and layer1_result['category'] == 'concept_persona':
        return type_driven_check(statement, stmt_tokens, layer1_result, compiled, llm_fn)

    return layer1_result


def concept_verify(response_text: str, kg_nodes: list, compiled: dict = None,
                   llm_fn=None) -> dict:
    """
    Run concept-level AEC on a response.

    Layer 1: Compiled deterministic matching (always runs)
    Layer 2: LLM type-driven verification (only for ambiguous/persona, requires llm_fn)
    Layer 3: Compiled edge policy execution (fires on matched nodes)

    If compiled dict is provided (from capsule load), uses it directly.
    Otherwise compiles on the fly (for standalone verify).
    """
    from aec import split_statements

    if compiled is None:
        compiled = compile_kg(kg_nodes)

    statements = split_statements(response_text)

    grounded = 0
    ungrounded = 0
    persona = 0
    details = []
    gaps = []
    llm_calls = 0
    edge_violations = []

    for stmt in statements:
        stmt_tokens = tokenize(stmt)
        result = match_statement(stmt_tokens, stmt, compiled, llm_fn=llm_fn)

        # Track LLM calls
        if result.get('llm_result'):
            llm_calls += 1

        # Layer 3: Execute edge policies on matched nodes
        matched_ids = [m['node_id'] for m in result.get('matches', [])]
        if matched_ids and compiled.get('edge_policies'):
            e_violations = execute_edge_policies(stmt_tokens, matched_ids, compiled)
            if e_violations:
                # Edge violation overrides any grounded classification
                edge_violations.extend(e_violations)
                ungrounded += 1
                top_v = e_violations[0]
                details.append({
                    'statement': stmt,
                    'category': 'ungrounded',
                    'method': f"edge:{top_v['type']}",
                    'path': top_v['path'],
                    'evidence': top_v['evidence'],
                    'explanation': top_v['explanation'],
                    'severity': top_v['severity'],
                })
                gaps.append({
                    'text': top_v['explanation'],
                    'node_id': top_v['target_id'],
                    'violation_type': top_v['type'],
                })
                continue  # Edge violation processed — don't double-count

        # Standard classification (no edge violation)
        if result['category'] == 'concept_grounded':
            grounded += 1
            top = result['matches'][0]
            details.append({
                'statement': stmt,
                'category': 'grounded',
                'method': f"concept:{top['node_type']}",
                'matched_node': top['node_id'],
                'matched_label': top['label'],
                'coverage': top['coverage'],
                'overlap_tokens': top.get('overlap_tokens', []),
            })
        elif result['category'] == 'antipattern_violation':
            ungrounded += 1
            v = result['violation']
            details.append({
                'statement': stmt,
                'category': 'ungrounded',
                'method': 'antipattern_violation',
                'matched_node': v['node_id'],
                'matched_label': v['label'],
                'violation_terms': v.get('hits', []),
            })
            gaps.append({
                'text': f"VIOLATION: '{', '.join(v.get('hits', []))}' matches antipattern '{v['label']}'",
                'node_id': v['node_id'],
            })
        elif result['category'] == 'concept_ungrounded':
            # Layer 2 found a violation via LLM
            ungrounded += 1
            llm_res = result.get('llm_result', {})
            details.append({
                'statement': stmt,
                'category': 'ungrounded',
                'method': llm_res.get('method', 'llm_violation'),
                'matched_node': llm_res.get('node_id', ''),
                'matched_label': llm_res.get('node_label', ''),
                'reasoning': llm_res.get('reasoning', ''),
            })
            gaps.append({
                'text': f"LLM VIOLATION: {llm_res.get('reasoning', '')}",
                'node_id': llm_res.get('node_id', ''),
            })
        else:
            # Check if Layer 2 upgraded to grounded
            llm_res = result.get('llm_result')
            if llm_res and llm_res.get('category') == 'grounded':
                grounded += 1
                details.append({
                    'statement': stmt,
                    'category': 'grounded',
                    'method': llm_res.get('method', 'llm_grounded'),
                    'matched_node': llm_res.get('node_id', ''),
                    'matched_label': llm_res.get('node_label', ''),
                    'reasoning': llm_res.get('reasoning', ''),
                })
            else:
                # Pure persona
                persona += 1
                details.append({
                    'statement': stmt,
                    'category': 'persona',
                    'method': 'no_concept_match',
                    'ambiguous_candidates': [
                        {'node_id': a['node_id'], 'dice': a.get('dice', 0)}
                        for a in result.get('ambiguous', [])
                    ],
                })

    total = grounded + ungrounded
    score = grounded / total if total > 0 else 1.0

    return {
        'score': round(score, 4),
        'grounded_statements': grounded,
        'ungrounded_statements': ungrounded,
        'persona_statements': persona,
        'total_statements': grounded + ungrounded + persona,
        'details': details,
        'gaps': gaps,
        'edge_violations': edge_violations,
        'method': 'concept_3layer_compiled',
        'llm_calls': llm_calls,
    }


def has_typed_nodes(kg_nodes: list) -> bool:
    """Check if KG has nodes with Rule/AntiPattern/Technique/Concept/Tool types."""
    type_markers = ('Rule', 'AntiPattern', 'Technique', 'Concept', 'Tool')
    for node in kg_nodes:
        ntype = node.get('@type', '')
        if ntype and any(t in ntype for t in type_markers):
            return True
    return False


if __name__ == "__main__":
    import time
    import json

    # Test with frontend-design KG
    kg_path = 'examples/frontend-design-v1.0.0-ff6ab491/frontend-design-v1.0.0-ff6ab491-kg.jsonld'
    try:
        with open(kg_path) as f:
            kg = json.load(f)
        nodes = kg.get('@graph', [])
    except FileNotFoundError:
        print(f"KG not found: {kg_path}")
        nodes = []

    if nodes:
        # Test compilation performance
        start = time.time()
        compiled = compile_kg(nodes)
        compile_time = (time.time() - start) * 1000

        print(f"Compile time: {compile_time:.2f}ms")
        print(f"Detectors: {len(compiled['detectors'])}")
        print(f"Blacklist tokens: {len(compiled['blacklist'])}")
        print(f"Blacklist sample: {list(compiled['blacklist'])[:10]}")

        # Test statements
        tests = [
            ("Use CSS variables for consistency", "should be GROUNDED (rule match)"),
            ("Implement staggered reveals using animation-delay", "should be GROUNDED (technique match)"),
            ("Use Inter for the body text", "should be UNGROUNDED (antipattern violation)"),
            ("Your typography should feel handcrafted", "should be PERSONA (no match)"),
        ]

        print("\n--- Statement Tests ---")
        for stmt, expected in tests:
            tokens = tokenize(stmt)
            result = match_statement(tokens, stmt, compiled)
            print(f"\nStatement: {stmt}")
            print(f"Expected: {expected}")
            print(f"Category: {result['category']}")
            if result['matches']:
                top = result['matches'][0]
                print(f"Match: {top['label']} (coverage={top['coverage']}, tokens={top['overlap_tokens']})")
            if result['violation']:
                print(f"Violation: {result['violation']['label']} (hits={result['violation']['hits']})")
