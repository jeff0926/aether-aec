"""
KG - Knowledge Graph loader for Aether capsules.
JSON-LD based with 5 knowledge origin types.
"""

import json
from pathlib import Path
from datetime import datetime

ORIGIN_TYPES = ["core", "acquired", "updated", "deprecated", "provenance"]

EMPTY_KG = {
    "@context": {"rdfs": "http://www.w3.org/2000/01/rdf-schema#", "aether": "http://aether.dev/ontology#"},
    "@graph": [],
}


def load_kg(path: str | Path) -> dict:
    """Load kg.jsonld. Returns empty graph if missing."""
    path = Path(path)
    if not path.exists():
        return {"@context": EMPTY_KG["@context"].copy(), "@graph": []}

    with open(path, "r", encoding="utf-8") as f:
        kg = json.load(f)

    # Normalize to @graph format
    if "@graph" not in kg and "@id" in kg:
        kg = {"@context": kg.get("@context", EMPTY_KG["@context"]), "@graph": [kg]}
    elif "@graph" not in kg:
        kg["@graph"] = []
    return kg


def get_nodes(kg: dict) -> list[dict]:
    """Extract all nodes from @graph."""
    if "@graph" in kg:
        return kg["@graph"]
    return [kg] if "@id" in kg else []


def query_nodes(kg: dict, entities: list[str]) -> list[dict]:
    """Find nodes matching any entity name (case-insensitive)."""
    if not entities:
        return []

    entities_lower = [e.lower() for e in entities]
    matches = []

    for node in get_nodes(kg):
        if _node_matches(node, entities_lower):
            matches.append(node)
    return matches


def _node_matches(node: dict, entities_lower: list[str]) -> bool:
    """Check if node matches any entity."""
    # Check @id and common label fields
    for key in ["@id", "rdfs:label", "name", "label", "title"]:
        val = node.get(key, "")
        if isinstance(val, str) and any(e in val.lower() for e in entities_lower):
            return True
    # Search all string values
    return _search_values(node, entities_lower)


def _search_values(obj, entities_lower: list[str]) -> bool:
    """Recursively search for entity matches in string values."""
    if isinstance(obj, str):
        return any(e in obj.lower() for e in entities_lower)
    if isinstance(obj, dict):
        return any(_search_values(v, entities_lower) for k, v in obj.items() if not k.startswith("@"))
    if isinstance(obj, list):
        return any(_search_values(item, entities_lower) for item in obj)
    return False


def add_knowledge(kg: dict, triple: dict, origin: str = "acquired") -> dict:
    """
    Add knowledge with specified origin type and provenance metadata.
    Triple: {subject, predicate, object, confidence, aec_trigger}
    Origin must be one of: core, acquired, updated, deprecated, provenance
    """
    if origin not in ORIGIN_TYPES:
        raise ValueError(f"Unknown origin: {origin}. Must be one of {ORIGIN_TYPES}")

    nodes = kg.setdefault("@graph", [])
    subject = triple.get("subject", "unknown")
    node_id = f"aether:{origin}/{subject.replace(' ', '_').lower()}"

    # Find or create node
    existing = next((n for n in nodes if n.get("@id") == node_id), None)

    if existing:
        existing[triple["predicate"]] = triple["object"]
        existing["aether:updated"] = datetime.now().isoformat()
    else:
        nodes.append({
            "@id": node_id,
            "rdfs:label": subject,
            triple["predicate"]: triple["object"],
            "aether:origin": origin,
            "aether:confidence": triple.get("confidence", 0.5),
            "aether:acquired_date": datetime.now().isoformat(),
            "aether:aec_trigger": triple.get("aec_trigger", "unknown"),
            "aether:last_accessed": datetime.now().isoformat(),
            "aether:access_count": 0,
        })
    return kg


def add_acquired(kg: dict, triple: dict) -> dict:
    """Convenience wrapper for add_knowledge with origin='acquired'."""
    return add_knowledge(kg, triple, origin="acquired")


def mark_deprecated(kg: dict, node_id: str, reason: str = "") -> dict:
    """Mark an existing node as deprecated. Does NOT delete it."""
    for node in get_nodes(kg):
        if node.get("@id") == node_id:
            node["aether:origin"] = "deprecated"
            node["aether:deprecated_date"] = datetime.now().isoformat()
            node["aether:deprecated_reason"] = reason
            break
    return kg


def mark_updated(kg: dict, node_id: str, updates: dict) -> dict:
    """Update an existing node's values and mark as updated."""
    for node in get_nodes(kg):
        if node.get("@id") == node_id:
            for k, v in updates.items():
                node[k] = v
            node["aether:origin"] = "updated"
            node["aether:updated_date"] = datetime.now().isoformat()
            break
    return kg


def touch_node(kg: dict, node_id: str) -> dict:
    """Increment access_count and update last_accessed on a node."""
    for node in get_nodes(kg):
        if node.get("@id") == node_id:
            node["aether:last_accessed"] = datetime.now().isoformat()
            node["aether:access_count"] = node.get("aether:access_count", 0) + 1
            break
    return kg


def get_nodes_by_origin(kg: dict, origin: str) -> list[dict]:
    """Return nodes with specified origin type."""
    if origin == "core":
        return [n for n in get_nodes(kg) if n.get("aether:origin", "core") == "core"]
    return [n for n in get_nodes(kg) if n.get("aether:origin") == origin]


def get_core_nodes(kg: dict) -> list[dict]:
    """Return nodes with origin: 'core' or no origin (original knowledge)."""
    return get_nodes_by_origin(kg, "core")


def get_acquired_nodes(kg: dict) -> list[dict]:
    """Return nodes with origin: 'acquired' (learned through AEC)."""
    return get_nodes_by_origin(kg, "acquired")


def get_deprecated_nodes(kg: dict) -> list[dict]:
    """Return nodes marked as deprecated."""
    return get_nodes_by_origin(kg, "deprecated")


def save_kg(kg: dict, path: str | Path) -> None:
    """Write KG to file with pretty printing."""
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)


def stats(kg: dict) -> dict:
    """Return KG statistics for all 5 origin types."""
    nodes = get_nodes(kg)
    return {
        "total": len(nodes),
        "core": len([n for n in nodes if n.get("aether:origin", "core") == "core"]),
        "acquired": len([n for n in nodes if n.get("aether:origin") == "acquired"]),
        "updated": len([n for n in nodes if n.get("aether:origin") == "updated"]),
        "deprecated": len([n for n in nodes if n.get("aether:origin") == "deprecated"]),
        "provenance": len([n for n in nodes if n.get("aether:origin") == "provenance"]),
    }


if __name__ == "__main__":
    # Start with empty KG for testing
    kg = {"@context": EMPTY_KG["@context"].copy(), "@graph": [
        {"@id": "test:core1", "rdfs:label": "Core Node 1", "value": 100},
        {"@id": "test:core2", "rdfs:label": "Core Node 2", "value": 200},
    ]}
    print(f"Initial: {stats(kg)}")

    # Test add_acquired (convenience wrapper)
    kg = add_acquired(kg, {
        "subject": "Acquired Fact",
        "predicate": "rdfs:comment",
        "object": "Learned via AEC",
        "confidence": 0.85,
        "aec_trigger": "verification",
    })
    print(f"After acquired: {stats(kg)}")

    # Test add_knowledge with provenance origin
    kg = add_knowledge(kg, {
        "subject": "Source Document",
        "predicate": "rdfs:comment",
        "object": "External reference",
        "confidence": 1.0,
    }, origin="provenance")
    print(f"After provenance: {stats(kg)}")

    # Test mark_updated
    kg = mark_updated(kg, "test:core1", {"value": 150, "rdfs:comment": "Value updated"})
    print(f"After update: {stats(kg)}")

    # Test mark_deprecated
    kg = mark_deprecated(kg, "test:core2", reason="Outdated information")
    print(f"After deprecate: {stats(kg)}")

    # Show final state
    print(f"\nFinal stats: {stats(kg)}")
    print(f"Core nodes: {[n['rdfs:label'] for n in get_core_nodes(kg)]}")
    print(f"Acquired nodes: {[n['rdfs:label'] for n in get_acquired_nodes(kg)]}")
    print(f"Deprecated nodes: {[n['rdfs:label'] for n in get_deprecated_nodes(kg)]}")
    print(f"All origin types: {ORIGIN_TYPES}")
