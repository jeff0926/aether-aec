#!/usr/bin/env python3
"""
AEC CLI - Standalone verification command.
Verify any text against any JSON-LD knowledge graph.

Usage:
    python cli.py verify "Your text here" --kg path/to/kg.jsonld
    python cli.py verify "Your text here" --kg path/to/kg.jsonld --threshold 0.9
    python cli.py verify "Your text here" --kg path/to/kg.jsonld --json
"""

import argparse
import json
import sys

from kg import load_kg, get_nodes
from aec import verify, DEFAULT_THRESHOLD
from aec_concept import compile_kg, has_typed_nodes


def cmd_verify(args):
    """Verify text against a knowledge graph."""
    # Load KG
    try:
        kg = load_kg(args.kg)
        nodes = get_nodes(kg)
    except Exception as e:
        print(f"Error loading KG: {e}", file=sys.stderr)
        return 1

    if not nodes:
        print(f"Warning: KG is empty or invalid: {args.kg}", file=sys.stderr)

    # Check if typed KG (for concept matching)
    compiled = None
    if has_typed_nodes(nodes):
        compiled = compile_kg(nodes)

    # Run verification
    result = verify(
        response=args.text,
        kg_nodes=nodes,
        threshold=args.threshold,
        compiled_kg=compiled,
        llm_fn=None,  # No LLM in CLI mode by default
    )

    # Output
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # Human-readable output
        status = "PASS" if result["passed"] else "FAIL"
        print(f"Score: {result['score']:.3f} (threshold: {result['threshold']})")
        print(f"Status: {status}")
        print(f"Statements: {result['total_statements']} total")
        print(f"  Grounded:   {result['grounded_statements']}")
        print(f"  Ungrounded: {result['ungrounded_statements']}")
        print(f"  Persona:    {result['persona_statements']}")

        if result.get("concept_applied"):
            print(f"Concept matching: enabled")

        if result["gaps"]:
            print(f"\nGaps ({len(result['gaps'])}):")
            for gap in result["gaps"]:
                reason = gap.get("reason", "unknown")
                text = gap.get("text", "")[:60]
                print(f"  - [{reason}] {text}...")

    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(
        description="AEC - Agent Education Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py verify "The company was founded in 2015." --kg data/company.jsonld
  python cli.py verify "Use CSS variables for consistency." --kg examples/frontend-design/demo-kg.jsonld
  python cli.py verify "Use Inter font." --kg examples/frontend-design/demo-kg.jsonld --json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify text against a KG")
    verify_parser.add_argument("text", help="Text to verify")
    verify_parser.add_argument("--kg", required=True, help="Path to JSON-LD knowledge graph")
    verify_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Grounding threshold (default: {DEFAULT_THRESHOLD})",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON",
    )

    args = parser.parse_args()

    if args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
