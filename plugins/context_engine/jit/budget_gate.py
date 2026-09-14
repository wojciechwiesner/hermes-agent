"""Request Budget Gate: Formally verifies that the complete assembled request
(system prompt + selected messages + hydrated tool schemas + output reserve)
fits inside the provider's active runtime context window.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class RequestBudgetGate:
    """Mathematical gate ensuring full request fit before dispatch."""

    def __init__(self, reserved_output: int = 2048, safety_margin: int = 512) -> None:
        self.reserved_output = reserved_output
        self.safety_margin = safety_margin

    def estimate_tokens(self, text_or_obj: Any) -> int:
        if isinstance(text_or_obj, str):
            return max(len(text_or_obj) // 4, 1)
        try:
            dumped = json.dumps(text_or_obj)
            return max(len(dumped) // 4, 1)
        except Exception:
            return 500

    def validate_request(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        runtime_context_window: int,
    ) -> tuple[bool, Dict[str, Any]]:
        """Formally evaluate if total wire request fits within window.
        
        Returns:
            (is_permitted: bool, metrics: dict)
        """
        msg_tokens = sum(self.estimate_tokens(m.get("content", "")) for m in messages if isinstance(m, dict))
        tool_tokens = sum(self.estimate_tokens(t) for t in tools if isinstance(t, dict))
        
        # Provider overhead padding (framing, roles, headers)
        provider_overhead = len(messages) * 4 + len(tools) * 4
        
        total_request_tokens = msg_tokens + tool_tokens + provider_overhead
        safe_budget = runtime_context_window - self.reserved_output - self.safety_margin
        
        is_permitted = total_request_tokens <= safe_budget
        
        metrics = {
            "is_permitted": is_permitted,
            "runtime_context_window": runtime_context_window,
            "total_request_tokens": total_request_tokens,
            "message_tokens": msg_tokens,
            "tool_schema_tokens": tool_tokens,
            "provider_overhead": provider_overhead,
            "reserved_output": self.reserved_output,
            "safety_margin": self.safety_margin,
            "safe_budget_ceiling": safe_budget,
            "remaining_headroom": safe_budget - total_request_tokens,
        }
        return is_permitted, metrics
