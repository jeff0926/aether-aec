"""
AEC Standalone Validation Tests

Proves that aec.verify() works completely independent of the Aether pipeline,
with arbitrary response/KG combinations. No pytest needed - just run directly.

Usage: python tests/test_aec_standalone.py
"""

import sys
sys.path.insert(0, ".")

from aec import verify, split_statements, deterministic_gate

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, details: str = ""):
    """Record test result."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        print(f"  FAIL: {name} - {details}")


def test_1_perfect_grounding():
    """All values in response match KG exactly."""
    print("\nTest 1: Perfect grounding")
    response = "The company was founded in 2015 and has 500 employees."
    kg = [{"@id": "co:acme", "founded": 2015, "employees": 500}]

    result = verify(response, kg)

    check("score is 1.0", result["score"] == 1.0, f"got {result['score']}")
    check("passed is True", result["passed"] == True, f"got {result['passed']}")
    check("grounded >= 1", result["grounded_statements"] >= 1, f"got {result['grounded_statements']}")
    check("ungrounded is 0", result["ungrounded_statements"] == 0, f"got {result['ungrounded_statements']}")


def test_2_complete_failure():
    """All values in response contradict KG (values >1% apart to avoid tolerance match)."""
    print("\nTest 2: Complete failure")
    # Use 1990 vs 2015 (~1.25% apart) and 1000 vs 500 (100% apart) to ensure no tolerance match
    response = "The company was founded in 1990 and has 1000 employees."
    kg = [{"@id": "co:acme", "founded": 2015, "employees": 500}]

    result = verify(response, kg)

    check("score is 0.0", result["score"] == 0.0, f"got {result['score']}")
    check("passed is False", result["passed"] == False, f"got {result['passed']}")
    check("ungrounded >= 1", result["ungrounded_statements"] >= 1, f"got {result['ungrounded_statements']}")


def test_3_mixed_grounding():
    """Some values match, some don't (Anne Hidalgo not in KG)."""
    print("\nTest 3: Mixed grounding")
    response = "Paris has a population of 2.1 million. The mayor is Anne Hidalgo. The Eiffel Tower was built in 1889."
    kg = [{"@id": "city:paris", "population": 2100000, "landmark": "Eiffel Tower", "landmark_year": 1889}]

    result = verify(response, kg)

    check("score > 0", result["score"] > 0, f"got {result['score']}")
    check("score < 1", result["score"] < 1.0, f"got {result['score']}")
    check("grounded >= 1", result["grounded_statements"] >= 1, f"got {result['grounded_statements']}")


def test_4_all_persona():
    """Response has no verifiable values - all persona statements."""
    print("\nTest 4: All persona (no verifiable values)")
    response = "This is a fascinating and complex topic that deserves careful consideration."
    kg = [{"@id": "any:thing", "value": 42}]

    result = verify(response, kg)

    check("score is 1.0", result["score"] == 1.0, f"got {result['score']}")
    check("passed is True", result["passed"] == True, f"got {result['passed']}")
    check("persona >= 1", result["persona_statements"] >= 1, f"got {result['persona_statements']}")
    check("grounded is 0", result["grounded_statements"] == 0, f"got {result['grounded_statements']}")
    check("ungrounded is 0", result["ungrounded_statements"] == 0, f"got {result['ungrounded_statements']}")


def test_5_numeric_tolerance():
    """Numbers within 1% tolerance should match."""
    print("\nTest 5: Numeric tolerance")
    response = "The temperature is 72.5 degrees."
    kg = [{"@id": "weather:today", "temp": 72}]

    result = verify(response, kg)

    # 72.5 is within ~0.7% of 72, should be grounded
    check("grounded (within tolerance)", result["grounded_statements"] >= 1 or result["score"] > 0,
         f"score={result['score']}, grounded={result['grounded_statements']}")


def test_6_date_normalization():
    """Dates in different formats should match."""
    print("\nTest 6: Date normalization")
    response = "The event occurred on March 15, 1944."
    kg = [{"@id": "event:x", "date": "1944-03-15"}]

    result = verify(response, kg)

    # Year 1944 should be extractable and match
    check("date year grounded", result["grounded_statements"] >= 1 or result["score"] > 0.5,
         f"score={result['score']}, grounded={result['grounded_statements']}")


def test_7_empty_response():
    """Empty response should fail."""
    print("\nTest 7: Empty response")
    response = ""
    kg = [{"@id": "any:thing", "value": 123}]

    result = verify(response, kg)

    check("score is 0.0", result["score"] == 0.0, f"got {result['score']}")
    check("passed is False", result["passed"] == False, f"got {result['passed']}")


def test_8_empty_kg():
    """Response with facts but empty KG - all statements become persona (cannot verify)."""
    print("\nTest 8: Empty KG")
    response = "Jefferson was born in 1743."
    kg = []

    result = verify(response, kg)

    # With no KG to verify against, deterministic_gate returns values_found=0
    # This classifies all statements as PERSONA, and all-persona = score 1.0
    check("all persona (no KG to verify)", result["persona_statements"] >= 1,
         f"persona={result['persona_statements']}")
    check("no grounded (nothing to match)", result["grounded_statements"] == 0,
         f"grounded={result['grounded_statements']}")
    check("no ungrounded (no KG = no mismatch)", result["ungrounded_statements"] == 0,
         f"ungrounded={result['ungrounded_statements']}")


def test_9_large_kg_single_match():
    """Large KG with 20 nodes, response mentions only one."""
    print("\nTest 9: Large KG, single match")

    # Create 20 nodes, only one matches
    kg = [{"@id": f"entity:{i}", "value": i * 100, "name": f"Entity{i}"} for i in range(20)]
    kg[5]["special_year"] = 1776  # Add specific value to node 5

    response = "The declaration was signed in 1776. This was a pivotal moment in history."

    result = verify(response, kg)

    check("grounded >= 1", result["grounded_statements"] >= 1, f"got {result['grounded_statements']}")
    check("passed is True", result["passed"] == True, f"got {result['passed']}")


def test_10_formatted_numbers():
    """Numbers with formatting (commas, currency) should match."""
    print("\nTest 10: Formatted numbers")
    response = "The project cost $3,500,000 and took 18 months."
    kg = [{"@id": "proj:x", "cost": 3500000, "duration_months": 18}]

    result = verify(response, kg)

    check("grounded >= 1", result["grounded_statements"] >= 1, f"got {result['grounded_statements']}")
    check("score > 0.5", result["score"] > 0.5, f"got {result['score']}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AEC Standalone Validation Tests")
    print("=" * 60)

    test_1_perfect_grounding()
    test_2_complete_failure()
    test_3_mixed_grounding()
    test_4_all_persona()
    test_5_numeric_tolerance()
    test_6_date_normalization()
    test_7_empty_response()
    test_8_empty_kg()
    test_9_large_kg_single_match()
    test_10_formatted_numbers()

    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"SUMMARY: {PASSED}/{total} passed")
    if FAILED > 0:
        print(f"         {FAILED} failed")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
