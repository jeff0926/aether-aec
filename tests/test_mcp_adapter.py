#!/usr/bin/env python3
"""
Tests for MCP adapter - testing JSON-RPC handling directly.

Usage: python tests/test_mcp_adapter.py
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.mcp import handle_request, AEC_VERIFY_TOOL

PASSED = 0
FAILED = 0

# Test KG path
TEST_KG = "examples/frontend-design/kg.jsonld"


def check(name: str, condition: bool, details: str = ""):
    """Record test result."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        print(f"  FAIL: {name} - {details}")


def test_tools_list():
    """tools/list returns aec_verify in tool list."""
    print("\nTest: tools_list")

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }

    response = handle_request(request)

    check("has result", "result" in response, "missing result")
    check("has tools", "tools" in response.get("result", {}), "missing tools")

    tools = response.get("result", {}).get("tools", [])
    tool_names = [t.get("name") for t in tools]
    check("aec_verify in tools", "aec_verify" in tool_names, f"got {tool_names}")


def test_tools_call_verify():
    """tools/call with valid input returns score."""
    print("\nTest: tools_call_verify")

    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "aec_verify",
            "arguments": {
                "text": "Use CSS variables for design tokens.",
                "kg_path": TEST_KG,
            },
        },
    }

    response = handle_request(request)

    check("has result", "result" in response, "missing result")
    check("no error", "error" not in response, f"got error: {response.get('error')}")

    result = response.get("result", {})
    check("has content", "content" in result, "missing content")

    content = result.get("content", [])
    check("content is list", isinstance(content, list), f"got {type(content)}")
    check("has content item", len(content) > 0, "empty content")

    if content:
        text = content[0].get("text", "")
        try:
            data = json.loads(text)
            check("score in result", "score" in data, f"got {list(data.keys())}")
            check("passed in result", "passed" in data, f"got {list(data.keys())}")
        except json.JSONDecodeError:
            check("valid JSON", False, f"not valid JSON: {text[:50]}")


def test_tools_call_missing_params():
    """missing text returns error."""
    print("\nTest: tools_call_missing_params")

    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "aec_verify",
            "arguments": {
                "kg_path": TEST_KG,
                # missing text
            },
        },
    }

    response = handle_request(request)

    check("has result", "result" in response, "missing result")
    result = response.get("result", {})
    check("isError is True", result.get("isError") == True, f"got {result.get('isError')}")


def test_output_is_valid_json():
    """result content is parseable JSON."""
    print("\nTest: output_is_valid_json")

    request = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "aec_verify",
            "arguments": {
                "text": "The design uses modern typography.",
                "kg_path": TEST_KG,
            },
        },
    }

    response = handle_request(request)
    content = response.get("result", {}).get("content", [])

    if content:
        text = content[0].get("text", "")
        try:
            data = json.loads(text)
            check("JSON parses correctly", True, "")
            check("is dict", isinstance(data, dict), f"got {type(data)}")
        except json.JSONDecodeError as e:
            check("JSON parses correctly", False, str(e))
    else:
        check("has content", False, "empty content")


def test_score_in_result():
    """result contains score field."""
    print("\nTest: score_in_result")

    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "aec_verify",
            "arguments": {
                "text": "Use CSS variables for theming and consistency.",
                "kg_path": TEST_KG,
                "threshold": 0.8,
            },
        },
    }

    response = handle_request(request)
    content = response.get("result", {}).get("content", [])

    if content:
        text = content[0].get("text", "")
        try:
            data = json.loads(text)
            check("score is number", isinstance(data.get("score"), (int, float)), f"got {type(data.get('score'))}")
            check("score in range", 0 <= data.get("score", -1) <= 1, f"got {data.get('score')}")
        except json.JSONDecodeError:
            check("score in result", False, "not valid JSON")
    else:
        check("score in result", False, "empty content")


def test_initialize():
    """initialize returns server info."""
    print("\nTest: initialize")

    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "initialize",
        "params": {},
    }

    response = handle_request(request)

    check("has result", "result" in response, "missing result")
    result = response.get("result", {})
    check("has protocolVersion", "protocolVersion" in result, "missing protocolVersion")
    check("has capabilities", "capabilities" in result, "missing capabilities")
    check("has serverInfo", "serverInfo" in result, "missing serverInfo")


def test_unknown_method():
    """unknown method returns error."""
    print("\nTest: unknown_method")

    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "unknown/method",
        "params": {},
    }

    response = handle_request(request)

    check("has error", "error" in response, "should have error for unknown method")
    error = response.get("error", {})
    check("error code is -32601", error.get("code") == -32601, f"got {error.get('code')}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AEC MCP Adapter Tests")
    print("=" * 60)

    # Check test KG exists
    if not os.path.exists(TEST_KG):
        print(f"Test KG not found: {TEST_KG}")
        return 1

    test_tools_list()
    test_tools_call_verify()
    test_tools_call_missing_params()
    test_output_is_valid_json()
    test_score_in_result()
    test_initialize()
    test_unknown_method()

    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"SUMMARY: {PASSED}/{total} passed")
    if FAILED > 0:
        print(f"         {FAILED} failed")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
