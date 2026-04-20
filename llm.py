"""
LLM - Simple wrapper for LLM API calls.
No abstractions, no retry, no async. Just call and return.
"""

import json
import os
from pathlib import Path


def _load_env():
    """Load .env file from project root into os.environ."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env()  # Run at import time

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}

# Cost per 1K tokens (USD) - input/output
COST_PER_1K_TOKENS = {
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimate cost in USD for a given model and token counts."""
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0, "output": 0})
    return (tokens_in / 1000 * rates["input"]) + (tokens_out / 1000 * rates["output"])


def resolve_model(
    capability: str,
    preferred_provider: str = None,
    preferred_model: str = None,
    registry_path: str = None
) -> tuple:
    """
    Resolve provider and model from capability using model_registry.json.

    Resolution chain:
    1. Load registry from registry_path or default location
    2. If both preferred_provider and preferred_model given, validate and return
    3. If only preferred_provider given, find matching capability_map entry
    4. Look up capability in capability_map
    5. Fall back to capability_map["default"]
    6. Hard fallback to anthropic defaults

    Never raises. Always returns (provider, model) tuple.
    """
    # Step 1: Load registry
    registry = None
    if registry_path:
        reg_file = Path(registry_path)
    else:
        reg_file = Path(__file__).parent / "model_registry.json"

    if reg_file.exists():
        try:
            registry = json.loads(reg_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            print(f"[LLM] Warning: Could not parse {reg_file}")
            registry = None

    if not registry:
        # Skip to step 6
        print("[LLM] resolve_model fallback to default")
        return (
            preferred_provider or "anthropic",
            preferred_model or DEFAULT_MODELS.get("anthropic")
        )

    providers = registry.get("providers", {})
    capability_map = registry.get("capability_map", {})

    def is_disabled(prov: str) -> bool:
        return providers.get(prov, {}).get("disabled", False)

    # Step 2: Both preferred_provider and preferred_model provided
    if preferred_provider and preferred_model:
        if preferred_provider in providers and not is_disabled(preferred_provider):
            return (preferred_provider, preferred_model)

    # Step 3: Only preferred_provider provided
    if preferred_provider and not preferred_model:
        if not is_disabled(preferred_provider):
            for cap, entry in capability_map.items():
                if entry.get("provider") == preferred_provider:
                    return (preferred_provider, entry.get("model"))

    # Step 4: Look up capability
    if capability in capability_map:
        entry = capability_map[capability]
        prov = entry.get("provider")
        if not is_disabled(prov):
            return (prov, entry.get("model"))

    # Step 5: Fall back to default
    if "default" in capability_map:
        entry = capability_map["default"]
        prov = entry.get("provider")
        if not is_disabled(prov):
            return (prov, entry.get("model"))

    # Step 6: Hard fallback
    print("[LLM] resolve_model fallback to default")
    return (
        preferred_provider or "anthropic",
        preferred_model or DEFAULT_MODELS.get("anthropic")
    )


def call_llm(prompt: str, provider: str = "anthropic", model: str = None,
             api_key: str = None, max_tokens: int = 1024) -> dict:
    """
    Call an LLM and return response dict with text and token counts.
    Returns: {"text": str, "tokens_in": int, "tokens_out": int, "model": str, "cost": float}
    Never raises - returns error text on failure.
    """
    model = model or DEFAULT_MODELS.get(provider, "unknown")

    if provider == "stub":
        # Estimate tokens: ~4 chars per token
        tokens_in = len(prompt) // 4
        tokens_out = 10
        return {
            "text": f"[Stub response for {len(prompt)} char prompt]",
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "model": "stub",
            "cost": 0.0,
        }

    try:
        if provider == "anthropic":
            result = _call_anthropic(prompt, model, api_key, max_tokens)
        elif provider == "openai":
            result = _call_openai(prompt, model, api_key, max_tokens)
        else:
            return {"text": f"[LLM Error: Unknown provider '{provider}']",
                    "tokens_in": 0, "tokens_out": 0, "model": model, "cost": 0.0}

        result["cost"] = estimate_cost(model, result["tokens_in"], result["tokens_out"])
        return result
    except Exception as e:
        return {"text": f"[LLM Error: {e}]",
                "tokens_in": 0, "tokens_out": 0, "model": model, "cost": 0.0}


def _call_anthropic(prompt: str, model: str, api_key: str, max_tokens: int) -> dict:
    import anthropic
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"text": "[LLM Error: No ANTHROPIC_API_KEY]",
                "tokens_in": 0, "tokens_out": 0, "model": model}

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.content[0].text,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
        "model": model,
    }


def _call_openai(prompt: str, model: str, api_key: str, max_tokens: int) -> dict:
    import openai
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"text": "[LLM Error: No OPENAI_API_KEY]",
                "tokens_in": 0, "tokens_out": 0, "model": model}

    client = openai.OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "text": response.choices[0].message.content,
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
        "model": model,
    }


def make_llm_fn(provider: str = "anthropic", model: str = None, api_key: str = None) -> callable:
    """Return a callable matching Capsule's llm_fn interface. Returns dict with text and tokens."""
    def llm_fn(prompt: str, **kwargs) -> dict:
        return call_llm(prompt, provider=provider, model=model, api_key=api_key, **kwargs)
    return llm_fn


if __name__ == "__main__":
    # Test stub provider
    result = call_llm("Hello world", provider="stub")
    print(f"Stub: {result}")

    # Test missing API key handling
    result = call_llm("Hello", provider="anthropic", api_key=None)
    if "ANTHROPIC_API_KEY" in os.environ:
        print(f"Anthropic: {result['text'][:50]}...")
        print(f"  Tokens: {result['tokens_in']} in, {result['tokens_out']} out")
        print(f"  Cost: ${result['cost']:.6f}")
    else:
        print(f"Anthropic (no key): {result['text']}")

    # Test make_llm_fn
    stub_fn = make_llm_fn(provider="stub")
    print(f"make_llm_fn: {stub_fn('Test prompt')}")

    # Test cost estimation
    print(f"\nCost estimate (1000 in, 500 out, Claude Sonnet):")
    print(f"  ${estimate_cost('claude-sonnet-4-20250514', 1000, 500):.4f}")
