#!/usr/bin/env python3
"""
AEC CAP Adapter - SAP CAP OData-style service for AEC verification.

Wraps AEC as a CAP external service, supporting both OData action calls
and plain JSON POST requests.

CAP CDS equivalent:
    action verify(text: String, kg_path: String, threshold: Decimal)
        returns { score: Decimal, passed: Boolean, gaps: String };

Endpoints:
    POST /AECService/verify     - OData action style
    POST /verify                - Plain JSON (REST-compatible)
    GET  /health                - Health check

Usage:
    python adapters/cap.py --port 8081 --kg examples/domain-sap-cap/kg.jsonld
"""

import argparse
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, make_response

from kg import load_kg, get_nodes
from aec import verify, DEFAULT_THRESHOLD
from aec_concept import compile_kg, has_typed_nodes

app = Flask(__name__)

# Cache for compiled KGs
_kg_cache: dict = {}

VERSION = "1.0.0"

# SAP correlation header name
SAP_CORRELATION_HEADER = os.getenv("SAP_CORRELATION_HEADER", "X-CorrelationID")


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

    return {
        "score": result["score"],
        "passed": result["passed"],
        "grounded_statements": result["grounded_statements"],
        "ungrounded_statements": result["ungrounded_statements"],
        "persona_statements": result["persona_statements"],
        "gaps": json.dumps(result["gaps"]),  # Serialize gaps for OData
        "method": "concept_3layer_compiled" if compiled else "factual_only",
    }


def _sap_error(code: str, message: str, status_code: int = 400):
    """Return SAP-compatible error response."""
    response = {
        "error": {
            "code": str(code),
            "message": {
                "value": message,
            },
        },
    }
    return jsonify(response), status_code


def _add_correlation_header(response):
    """Add correlation ID header if present in request."""
    correlation_id = request.headers.get(SAP_CORRELATION_HEADER)
    if correlation_id:
        response.headers[SAP_CORRELATION_HEADER] = correlation_id
    return response


@app.after_request
def after_request(response):
    """Add correlation header to all responses."""
    return _add_correlation_header(response)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "version": VERSION, "adapter": "cap"})


@app.route("/AECService/verify", methods=["POST"])
def odata_verify():
    """OData-style verify action."""
    data = request.get_json()

    if not data:
        return _sap_error("400", "Request body required")

    # Support both flat params and nested OData action format
    if "parameters" in data:
        params = data["parameters"]
    else:
        params = data

    text = params.get("text")
    if not text:
        return _sap_error("400", "Missing required parameter: text")

    kg_path = params.get("kg_path") or os.getenv("AEC_DEFAULT_KG")
    if not kg_path:
        return _sap_error("400", "Missing kg_path (set AEC_DEFAULT_KG env var or provide in request)")

    threshold = params.get("threshold", DEFAULT_THRESHOLD)
    if isinstance(threshold, str):
        try:
            threshold = float(threshold)
        except ValueError:
            return _sap_error("400", f"Invalid threshold value: {threshold}")

    try:
        result = _verify_text(text, kg_path, threshold)
    except FileNotFoundError:
        return _sap_error("404", f"KG not found: {kg_path}", 404)
    except Exception as e:
        return _sap_error("500", f"Verification failed: {str(e)}", 500)

    # OData action result format
    response = {
        "d": result,
    }

    status_code = 200 if result["passed"] else 422
    return jsonify(response), status_code


@app.route("/verify", methods=["POST"])
def plain_verify():
    """Plain JSON verify endpoint (REST-compatible)."""
    data = request.get_json()

    if not data:
        return _sap_error("400", "Request body required")

    text = data.get("text")
    if not text:
        return _sap_error("400", "Missing required field: text")

    kg_path = data.get("kg_path") or os.getenv("AEC_DEFAULT_KG")
    if not kg_path:
        return _sap_error("400", "Missing kg_path")

    threshold = data.get("threshold", DEFAULT_THRESHOLD)

    try:
        result = _verify_text(text, kg_path, threshold)
        # Parse gaps back to JSON for plain endpoint
        result["gaps"] = json.loads(result["gaps"])
    except FileNotFoundError:
        return _sap_error("404", f"KG not found: {kg_path}", 404)
    except Exception as e:
        return _sap_error("500", f"Verification failed: {str(e)}", 500)

    status_code = 200 if result["passed"] else 422
    return jsonify(result), status_code


def main():
    parser = argparse.ArgumentParser(description="AEC CAP Adapter")
    parser.add_argument("--port", type=int, default=int(os.getenv("AEC_CAP_PORT", 8081)))
    parser.add_argument("--host", default=os.getenv("AEC_CAP_HOST", "0.0.0.0"))
    parser.add_argument("--kg", help="Default KG path (sets AEC_DEFAULT_KG)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.kg:
        os.environ["AEC_DEFAULT_KG"] = args.kg
        print(f"Default KG: {args.kg}")

    print(f"Starting AEC CAP adapter on {args.host}:{args.port}")
    print(f"OData endpoint: POST /AECService/verify")
    print(f"REST endpoint:  POST /verify")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
