#!/usr/bin/env python3
"""
Tests for REST adapter using Flask test client.

Usage: python tests/test_rest_adapter.py
"""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def get_test_client():
    """Get Flask test client."""
    from adapters.rest import app
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint():
    """GET /health returns 200."""
    print("\nTest: health_endpoint")
    client = get_test_client()

    response = client.get("/health")

    check("status code is 200", response.status_code == 200, f"got {response.status_code}")
    data = json.loads(response.data)
    check("status is ok", data.get("status") == "ok", f"got {data.get('status')}")
    check("version present", "version" in data, "missing version")


def test_verify_pass():
    """POST /verify with grounded text returns 200 + passed:true."""
    print("\nTest: verify_pass")
    client = get_test_client()

    response = client.post(
        "/verify",
        json={
            "text": "Use CSS variables for design tokens and theming consistency.",
            "kg_path": TEST_KG,
        },
        content_type="application/json",
    )

    check("status code is 200", response.status_code == 200, f"got {response.status_code}")
    data = json.loads(response.data)
    check("passed is True", data.get("passed") == True, f"got {data.get('passed')}")
    check("score >= 0.8", data.get("score", 0) >= 0.8, f"got {data.get('score')}")


def test_verify_fail():
    """POST /verify with ungrounded text returns 422 + passed:false."""
    print("\nTest: verify_fail")
    client = get_test_client()

    response = client.post(
        "/verify",
        json={
            "text": "Use Inter font for the body text and Arial for headings.",
            "kg_path": TEST_KG,
        },
        content_type="application/json",
    )

    check("status code is 422", response.status_code == 422, f"got {response.status_code}")
    data = json.loads(response.data)
    check("passed is False", data.get("passed") == False, f"got {data.get('passed')}")


def test_verify_missing_kg():
    """POST /verify without kg_path returns 400."""
    print("\nTest: verify_missing_kg")
    client = get_test_client()

    # Clear env var if set
    old_val = os.environ.pop("AEC_DEFAULT_KG", None)

    response = client.post(
        "/verify",
        json={"text": "Some text to verify"},
        content_type="application/json",
    )

    # Restore env var
    if old_val:
        os.environ["AEC_DEFAULT_KG"] = old_val

    check("status code is 400", response.status_code == 400, f"got {response.status_code}")
    data = json.loads(response.data)
    check("error present", "error" in data, "missing error field")


def test_verify_missing_text():
    """POST /verify without text returns 400."""
    print("\nTest: verify_missing_text")
    client = get_test_client()

    response = client.post(
        "/verify",
        json={"kg_path": TEST_KG},
        content_type="application/json",
    )

    check("status code is 400", response.status_code == 400, f"got {response.status_code}")
    data = json.loads(response.data)
    check("error present", "error" in data, "missing error field")


def test_verify_batch():
    """POST /verify-batch with two texts returns 200 + list."""
    print("\nTest: verify_batch")
    client = get_test_client()

    response = client.post(
        "/verify-batch",
        json={
            "texts": [
                "Use CSS variables for theming.",
                "Use Inter font everywhere.",
            ],
            "kg_path": TEST_KG,
        },
        content_type="application/json",
    )

    # One passes, one fails, so overall is 422
    check("status code is 422", response.status_code == 422, f"got {response.status_code}")
    data = json.loads(response.data)
    check("results is list", isinstance(data.get("results"), list), f"got {type(data.get('results'))}")
    check("two results", len(data.get("results", [])) == 2, f"got {len(data.get('results', []))}")
    check("summary present", "summary" in data, "missing summary")


def test_verify_custom_threshold():
    """POST /verify with threshold:0.5 uses custom threshold."""
    print("\nTest: verify_custom_threshold")
    client = get_test_client()

    response = client.post(
        "/verify",
        json={
            "text": "The design feels elegant.",
            "kg_path": TEST_KG,
            "threshold": 0.5,
        },
        content_type="application/json",
    )

    # Persona-only statement passes at any threshold
    check("status code is 200", response.status_code == 200, f"got {response.status_code}")


def test_response_shape():
    """Response contains score, passed, gaps, method."""
    print("\nTest: response_shape")
    client = get_test_client()

    response = client.post(
        "/verify",
        json={
            "text": "Use CSS variables.",
            "kg_path": TEST_KG,
        },
        content_type="application/json",
    )

    data = json.loads(response.data)
    check("score present", "score" in data, "missing score")
    check("passed present", "passed" in data, "missing passed")
    check("gaps present", "gaps" in data, "missing gaps")
    check("method present", "method" in data, "missing method")
    check("grounded_statements present", "grounded_statements" in data, "missing grounded_statements")
    check("ungrounded_statements present", "ungrounded_statements" in data, "missing ungrounded_statements")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AEC REST Adapter Tests")
    print("=" * 60)

    # Check Flask is available
    try:
        import flask
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        return 1

    # Check test KG exists
    if not os.path.exists(TEST_KG):
        print(f"Test KG not found: {TEST_KG}")
        return 1

    test_health_endpoint()
    test_verify_pass()
    test_verify_fail()
    test_verify_missing_kg()
    test_verify_missing_text()
    test_verify_batch()
    test_verify_custom_threshold()
    test_response_shape()

    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"SUMMARY: {PASSED}/{total} passed")
    if FAILED > 0:
        print(f"         {FAILED} failed")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
