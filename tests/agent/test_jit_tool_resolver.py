"""Unit tests for JIT Tool Surface: Capability Resolution & Lazy Schema Hydration."""

from __future__ import annotations

import pytest
from plugins.context_engine.jit.tool_resolver import JitToolResolver, DOMAIN_TAXONOMY
from plugins.context_engine import load_context_engine


@pytest.fixture
def mock_tools():
    tools = []
    for domain, info in DOMAIN_TAXONOMY.items():
        for tool_name in info["tools"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Executable function {tool_name} under {domain} with parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {"arg": {"type": "string"}},
                    },
                },
            })
    return tools


def test_tool_resolver_speculative_hydration(mock_tools):
    resolver = JitToolResolver("test-session-1")
    hydrated, metrics = resolver.resolve_active_tools(
        mock_tools, user_message="Fix bug in models.py and add pytest tests"
    )
    names = {t["function"]["name"] for t in hydrated}
    
    # Must contain core.code tools
    assert "read_file" in names
    assert "write_file" in names
    assert "terminal" in names
    
    # Must NOT eagerly contain heavy browser/kanban tools
    assert "browser_exec" not in names
    assert "kanban_create" not in names
    
    # Asymptotic reduction must exceed 60% on full catalog
    assert metrics["savings_pct"] > 60.0


def test_tool_resolver_explicit_activation(mock_tools):
    resolver = JitToolResolver("test-session-2")
    # Explicitly activate web.browser capability
    res = resolver.activate_capability("web.browser")
    assert res["status"] == "activated"
    assert "browser_exec" in res["tools"]
    
    hydrated, metrics = resolver.resolve_active_tools(
        mock_tools, user_message="Check status"
    )
    names = {t["function"]["name"] for t in hydrated}
    assert "browser_exec" in names


def test_capability_index_prompt_generation(mock_tools):
    resolver = JitToolResolver("test-session-3")
    prompt = resolver.get_capability_index_prompt(mock_tools)
    assert "<JIT_CAPABILITY_INDEX>" in prompt
    assert "core.code" in prompt
    assert "HYDRATED" in prompt
