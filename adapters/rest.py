#!/usr/bin/env python3
"""
AEC REST Adapter - Flask HTTP endpoint for AEC verification.

Endpoints:
    POST /verify       - Verify single text against a KG
    POST /verify-batch - Verify list of texts against a KG
    GET  /health       - Health check

Usage:
    python adapters/rest.py --port 8080 --kg examples/frontend-design/kg.jsonld
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

from kg import load_kg, get_nodes
from aec import verify, DEFAULT_THRESHOLD
from aec_concept import compile_kg, has_typed_nodes

app = Flask(__name__)

# Cache for compiled KGs (path -> compiled)
_kg_cache: dict = {}

VERSION = "1.0.0"


def _get_compiled_kg(kg_path: str) -> tuple:
    """Load and compile KG, with caching."""
    if kg_path in _kg_cache:
        return _kg_cache[kg_path]

    kg = load_kg(kg_path)
    nodes = get_nodes(kg)

    compiled = None
    if has_typed_nodes(nodes):
        compiled = compile_kg(nodes)

    _kg_cache[kg_path] = (nodes, compiled)
    return nodes, compiled


def _verify_text(text: str, kg_path: str, threshold: float) -> dict:
    """Run verification and return result dict."""
    nodes, compiled = _get_compiled_kg(kg_path)

    result = verify(
        response=text,
        kg_nodes=nodes,
        threshold=threshold,
        compiled_kg=compiled,
        llm_fn=None,
    )

    # Return subset matching spec
    return {
        "score": result["score"],
        "passed": result["passed"],
        "grounded_statements": result["grounded_statements"],
        "ungrounded_statements": result["ungrounded_statements"],
        "persona_statements": result["persona_statements"],
        "gaps": result["gaps"],
        "method": "concept_3layer_compiled" if compiled else "factual_only",
    }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "version": VERSION})


@app.route("/verify", methods=["POST"])
def verify_single():
    """Verify single text against a KG."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    text = data.get("text")
    if not text:
        return jsonify({"error": "Missing required field: text"}), 400

    kg_path = data.get("kg_path") or os.getenv("AEC_DEFAULT_KG")
    if not kg_path:
        return jsonify({"error": "Missing kg_path (set AEC_DEFAULT_KG env var or provide in request)"}), 400

    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    try:
        result = _verify_text(text, kg_path, threshold)
    except FileNotFoundError:
        return jsonify({"error": f"KG not found: {kg_path}"}), 400
    except Exception as e:
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

    # HTTP status based on pass/fail
    status_code = 200 if result["passed"] else 422
    return jsonify(result), status_code


@app.route("/verify-batch", methods=["POST"])
def verify_batch():
    """Verify list of texts against a KG."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    texts = data.get("texts")
    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Missing required field: texts (must be a list)"}), 400

    kg_path = data.get("kg_path") or os.getenv("AEC_DEFAULT_KG")
    if not kg_path:
        return jsonify({"error": "Missing kg_path (set AEC_DEFAULT_KG env var or provide in request)"}), 400

    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    try:
        results = []
        for text in texts:
            result = _verify_text(text, kg_path, threshold)
            result["text"] = text[:100]  # Include truncated text for reference
            results.append(result)
    except FileNotFoundError:
        return jsonify({"error": f"KG not found: {kg_path}"}), 400
    except Exception as e:
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

    # Aggregate stats
    all_passed = all(r["passed"] for r in results)
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0

    response = {
        "results": results,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "avg_score": round(avg_score, 3),
        },
    }

    status_code = 200 if all_passed else 422
    return jsonify(response), status_code


def main():
    parser = argparse.ArgumentParser(description="AEC REST Adapter")
    parser.add_argument("--port", type=int, default=int(os.getenv("AEC_REST_PORT", 8080)))
    parser.add_argument("--host", default=os.getenv("AEC_REST_HOST", "0.0.0.0"))
    parser.add_argument("--kg", help="Default KG path (sets AEC_DEFAULT_KG)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.kg:
        os.environ["AEC_DEFAULT_KG"] = args.kg
        print(f"Default KG: {args.kg}")

    print(f"Starting AEC REST adapter on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
