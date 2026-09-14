"""Comprehensive unit tests for JIT Capability Resolution, Lazy Schema Hydration,
and the Request Budget Gate (10/10 test suite).
"""

from __future__ import annotations

import pytest
from plugins.context_engine.jit.tool_resolver import JitToolResolver, DOMAIN_TAXONOMY
from plugins.context_engine.jit.budget_gate import RequestBudgetGate


@pytest.fixture
def mock_tools():
    tools = []
    for domain, data in DOMAIN_TAXONOMY.items():
        for name in data["tools"]:
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Tool {name} under domain {domain}",
                    "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
                }
            })
    return tools


def test_tool_resolver_initial_taxonomy(mock_tools):
    resolver = JitToolResolver("test-session-1")
    hydrated, metrics = resolver.resolve_active_tools(mock_tools)
    assert metrics["total_registered"] == len(mock_tools)
    assert metrics["hydrated_count"] < len(mock_tools)
    assert metrics["savings_pct"] > 50.0


def test_tool_resolver_speculative_hydration_for_coding(mock_tools):
    resolver = JitToolResolver("test-session-code")
    hydrated, _ = resolver.resolve_active_tools(mock_tools, user_message="Fix bug in models.py and run pytest")
    names = {t["function"]["name"] for t in hydrated}
    assert "patch" in names
    assert "read_file" in names
    assert "terminal" in names


def test_tool_resolver_explicit_activation(mock_tools):
    resolver = JitToolResolver("test-session-2")
    res = resolver.activate_capability("web.browser")
    assert res["status"] == "activated"
    assert "browser_exec" in res["tools"]
    
    hydrated, _ = resolver.resolve_active_tools(mock_tools)
    names = {t["function"]["name"] for t in hydrated}
    assert "browser_exec" in names


def test_capability_index_prompt_generation(mock_tools):
    resolver = JitToolResolver("test-session-3")
    prompt = resolver.get_capability_index_prompt(mock_tools)
    assert "<JIT_CAPABILITY_INDEX>" in prompt
    assert "core.code" in prompt
    assert "HYDRATED" in prompt


def test_ttl_turn_eviction_lifecycle(mock_tools):
    resolver = JitToolResolver("test-session-ttl")
    resolver.activate_capability("web.browser", ttl=2)
    assert "web.browser" in resolver.active_domains
    
    # Step 1
    resolver.step_turn_eviction()
    assert "web.browser" in resolver.active_domains
    
    # Step 2 -> TTL reaches 0 -> evicted
    resolver.step_turn_eviction()
    assert "web.browser" not in resolver.active_domains


def test_permanent_core_tools_never_evicted():
    resolver = JitToolResolver("test-session-permanent")
    for _ in range(10):
        resolver.step_turn_eviction()
    assert "core.code" in resolver.active_domains


def test_request_budget_gate_permits_when_within_budget():
    gate = RequestBudgetGate(reserved_output=2048, safety_margin=512)
    messages = [{"role": "system", "content": "You are Hermes"}, {"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    
    # 16K window: budget is 16384 - 2560 = 13824
    permitted, metrics = gate.validate_request(
        messages=messages,
        tools=tools,
        runtime_context_window=16384,
    )
    assert permitted is True
    assert metrics["remaining_headroom"] > 10000


def test_request_budget_gate_refuses_when_overflowing_window():
    gate = RequestBudgetGate(reserved_output=2048, safety_margin=512)
    # Huge message payload (15k tokens)
    huge_msg = "A" * 60000  # ~15k tokens
    messages = [{"role": "user", "content": huge_msg}]
    tools = [{"type": "function", "function": {"name": "tool", "parameters": {}}}]
    
    permitted, metrics = gate.validate_request(
        messages=messages,
        tools=tools,
        runtime_context_window=16384,
    )
    assert permitted is False
    assert metrics["remaining_headroom"] < 0


def test_request_budget_gate_detailed_token_accounting():
    gate = RequestBudgetGate(reserved_output=2048, safety_margin=512)
    messages = [{"role": "user", "content": "12345678"}]  # ~2 tokens
    tools = [{"name": "t"}]
    
    _, metrics = gate.validate_request(
        messages=messages,
        tools=tools,
        runtime_context_window=32768,
    )
    assert "message_tokens" in metrics
    assert "tool_schema_tokens" in metrics
    assert "provider_overhead" in metrics
    assert "total_request_tokens" in metrics
    assert metrics["total_request_tokens"] == metrics["message_tokens"] + metrics["tool_schema_tokens"] + metrics["provider_overhead"]


def test_qwen_16k_with_jit_hydration_fits_headroom(mock_tools):
    resolver = JitToolResolver("qwen-16k-session")
    hydrated, _ = resolver.resolve_active_tools(mock_tools, user_message="Build models.py")
    
    gate = RequestBudgetGate(reserved_output=2048, safety_margin=512)
    messages = [
        {"role": "system", "content": "SOTA Hermes Agent system prompt" * 10},  # ~100 tokens
        {"role": "user", "content": "Implement backend models and tests"},
    ]
    
    permitted, metrics = gate.validate_request(
        messages=messages,
        tools=hydrated,
        runtime_context_window=16384,
    )
    assert permitted is True
    # Asserts that with JIT hydration, Qwen 16K has > 10,000 tokens of free working memory!
    assert metrics["remaining_headroom"] > 10000
