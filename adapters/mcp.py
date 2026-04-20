#!/usr/bin/env python3
"""
AEC MCP Adapter - Model Context Protocol tool for AEC verification.

Exposes AEC verify as an MCP tool via stdio transport.
Implements JSON-RPC protocol directly (no external MCP SDK needed).

Usage:
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | python adapters/mcp.py
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg import load_kg, get_nodes
from aec import verify, DEFAULT_THRESHOLD
from aec_concept import compile_kg, has_typed_nodes

# MCP Protocol constants
JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "aec-verify"
SERVER_VERSION = "1.0.0"

# Tool definition
AEC_VERIFY_TOOL = {
    "name": "aec_verify",
    "description": "Verify text against a knowledge graph. Returns grounding score and gap list.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to verify against the knowledge graph",
            },
            "kg_path": {
                "type": "string",
                "description": "Path to the JSON-LD knowledge graph file",
            },
            "threshold": {
                "type": "number",
                "description": f"Grounding threshold (0.0-1.0, default: {DEFAULT_THRESHOLD})",
                "default": DEFAULT_THRESHOLD,
            },
        },
        "required": ["text", "kg_path"],
    },
}


def _run_verify(text: str, kg_path: str, threshold: float) -> dict:
    """Run AEC verification and return result."""
    kg = load_kg(kg_path)
    nodes = get_nodes(kg)

    compiled = None
    if has_typed_nodes(nodes):
        compiled = compile_kg(nodes)

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
        "gaps": result["gaps"],
        "method": "concept_3layer_compiled" if compiled else "factual_only",
    }


def handle_initialize(params: dict) -> dict:
    """Handle initialize request."""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {},
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
    }


def handle_tools_list(params: dict) -> dict:
    """Handle tools/list request."""
    return {
        "tools": [AEC_VERIFY_TOOL],
    }


def handle_tools_call(params: dict) -> dict:
    """Handle tools/call request."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name != "aec_verify":
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                }
            ],
            "isError": True,
        }

    text = arguments.get("text")
    kg_path = arguments.get("kg_path")
    threshold = arguments.get("threshold", DEFAULT_THRESHOLD)

    if not text:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": "Missing required parameter: text"}),
                }
            ],
            "isError": True,
        }

    if not kg_path:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": "Missing required parameter: kg_path"}),
                }
            ],
            "isError": True,
        }

    try:
        result = _run_verify(text, kg_path, threshold)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2),
                }
            ],
        }
    except FileNotFoundError:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": f"KG not found: {kg_path}"}),
                }
            ],
            "isError": True,
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"error": f"Verification failed: {str(e)}"}),
                }
            ],
            "isError": True,
        }


def handle_request(request: dict) -> dict:
    """Route request to appropriate handler."""
    method = request.get("method", "")
    params = request.get("params", {})
    request_id = request.get("id")

    handlers = {
        "initialize": handle_initialize,
        "tools/list": handle_tools_list,
        "tools/call": handle_tools_call,
    }

    handler = handlers.get(method)
    if handler:
        result = handler(params)
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "result": result,
        }
    else:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }


def main():
    """Main loop - read JSON-RPC from stdin, write responses to stdout."""
    # Read from stdin line by line
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": JSONRPC_VERSION,
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}",
                },
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    main()
