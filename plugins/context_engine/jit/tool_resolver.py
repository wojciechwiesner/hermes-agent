"""JIT Capability Resolver & Lazy Schema Hydration Engine.

Implements the 6-state lifecycle for agent tools:
REGISTERED -> INDEXED -> DISCOVERED -> HYDRATED -> ACTIVE/USED -> EVICTED
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set


# Core domain taxonomy mapping tool names to capability domains
DOMAIN_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "core.code": {
        "description": "Filesystem navigation, file reading/writing/patching, and shell command execution.",
        "tools": {"read_file", "write_file", "patch", "search_files", "terminal", "execute_code"},
        "risk_class": "workspace_execution",
        "default_ttl": None,  # Persistent across coding turns
    },
    "web.search": {
        "description": "Internet search and webpage markdown extraction.",
        "tools": {"web_search", "web_extract"},
        "risk_class": "read_external",
        "default_ttl": 4,
    },
    "web.browser": {
        "description": "Full headless browser automation, CDP navigation, DOM inspection, and password vault.",
        "tools": {
            "browser_exec", "browser_vault_fill", "browser_vault_list",
            "browser_vault_save_login", "browser_vault_enter_code", "browser_vault_unlock"
        },
        "risk_class": "network_execution",
        "default_ttl": 3,
    },
    "kanban.workflow": {
        "description": "Task orchestration, multi-agent dispatch, issue links, and kanban cards.",
        "tools": {
            "kanban_create", "kanban_complete", "kanban_block", "kanban_heartbeat",
            "kanban_show", "kanban_comment", "kanban_link", "kanban_list",
            "kanban_unblock", "kanban_attach", "kanban_attach_url", "kanban_attachments",
            "kanban_request_review", "kanban_request_changes"
        },
        "risk_class": "orchestration",
        "default_ttl": 3,
    },
    "multimedia": {
        "description": "Visual analysis, speech synthesis, video inspection, and image generation.",
        "tools": {"vision_analyze", "text_to_speech", "video_analyze", "image_generate"},
        "risk_class": "media_processing",
        "default_ttl": 2,
    },
    "meta.learning": {
        "description": "Persistent memory updates, skill authoring, delegation, and user clarification.",
        "tools": {"memory", "skill_manage", "skill_view", "skills_list", "clarify", "delegate_task"},
        "risk_class": "system_governance",
        "default_ttl": 3,
    },
}

# Meta-tool definition for dynamic explicit discovery
DISCOVERY_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "activate_capability",
        "description": "Dynamically hydrate tool definitions for a specific capability domain into the next request.",
        "parameters": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": "Domain name to hydrate (e.g. 'web.browser', 'web.search', 'kanban.workflow', 'multimedia', 'meta.learning').",
                },
                "reason": {
                    "type": "string",
                    "description": "Short justification why this capability is needed for the current step.",
                },
            },
            "required": ["capability"],
        },
    },
}


class JitToolResolver:
    """Manages progressive disclosure and lazy schema hydration for tools."""

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self.active_domains: Set[str] = {"core.code"}
        self.domain_ttls: Dict[str, int] = {}
        self.recently_used_tools: Set[str] = set()

    def estimate_schemas_tokens(self, tools: List[Dict[str, Any]]) -> int:
        """Rough estimation of tokens consumed by tool JSON schemas."""
        try:
            dumped = json.dumps(tools)
            return max(len(dumped) // 4, 1)
        except Exception:
            return len(tools) * 500

    def get_capability_index_prompt(self, registered_tools: List[Dict[str, Any]]) -> str:
        """Generate a minimal capability index markdown block (<100 tokens)."""
        registered_names = {
            t.get("function", {}).get("name") for t in registered_tools if isinstance(t, dict)
        }
        
        lines = [
            "<JIT_CAPABILITY_INDEX>",
            "Tool schemas are lazily hydrated Just-In-Time. Available domains:",
        ]
        
        for dom, info in DOMAIN_TAXONOMY.items():
            avail = info["tools"].intersection(registered_names)
            if not avail:
                continue
            is_active = dom in self.active_domains
            status = "HYDRATED" if is_active else "INDEXED"
            lines.append(f"- {dom} [{status}]: {info['description']}")
            
        lines.append("To hydrate dormant tools into context, call `activate_capability(capability=...)`.")
        lines.append("</JIT_CAPABILITY_INDEX>")
        return "\n".join(lines)

    def resolve_active_tools(
        self,
        registered_tools: List[Dict[str, Any]],
        *,
        user_message: str = "",
        active_scope: str = "general",
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Selectively hydrate tool schemas based on speculative intent & explicit activation.
        
        Returns:
            (hydrated_tools_list, telemetry_metrics)
        """
        if not registered_tools:
            return [], {"total_registered": 0, "hydrated_count": 0, "token_savings_pct": 0.0}

        total_registered = len(registered_tools)
        baseline_tokens = self.estimate_schemas_tokens(registered_tools)

        # 1. Speculative Hydration based on user intent keywords
        lower_msg = user_message.lower()
        if any(w in lower_msg for w in ("szukaj", "search", "google", "web", "find on internet", "extract")):
            self.active_domains.add("web.search")
            self.domain_ttls["web.search"] = DOMAIN_TAXONOMY["web.search"]["default_ttl"] or 4

        if any(w in lower_msg for w in ("browser", "przeglądark", "playwright", "click", "screenshot", "http", "login")):
            self.active_domains.add("web.browser")
            self.domain_ttls["web.browser"] = DOMAIN_TAXONOMY["web.browser"]["default_ttl"] or 3

        if any(w in lower_msg for w in ("kanban", "task", "karta", "board", "issue")):
            self.active_domains.add("kanban.workflow")
            self.domain_ttls["kanban.workflow"] = DOMAIN_TAXONOMY["kanban.workflow"]["default_ttl"] or 3

        # Always keep core.code active for engineering scopes
        self.active_domains.add("core.code")

        # 2. Collect allowed tool names from active domains
        allowed_tool_names: Set[str] = set()
        for dom in self.active_domains:
            if dom in DOMAIN_TAXONOMY:
                allowed_tool_names.update(DOMAIN_TAXONOMY[dom]["tools"])

        # 3. Filter registered tools to hydrated set
        hydrated_tools: List[Dict[str, Any]] = []
        for t in registered_tools:
            name = t.get("function", {}).get("name")
            if name in allowed_tool_names:
                hydrated_tools.append(t)

        # 4. Include the meta discovery tool so model can explicitly request others
        hydrated_tools.append(DISCOVERY_TOOL_SCHEMA)

        hydrated_tokens = self.estimate_schemas_tokens(hydrated_tools)
        savings_pct = round((1.0 - (hydrated_tokens / max(baseline_tokens, 1))) * 100.0, 1)

        metrics = {
            "total_registered": total_registered,
            "hydrated_count": len(hydrated_tools),
            "baseline_tokens": baseline_tokens,
            "hydrated_tokens": hydrated_tokens,
            "savings_tokens": baseline_tokens - hydrated_tokens,
            "savings_pct": savings_pct,
            "active_domains": list(self.active_domains),
        }

        return hydrated_tools, metrics

    def activate_capability(self, domain_name: str, ttl: int = 3) -> Dict[str, Any]:
        """Explicitly hydrate a domain capability with custom TTL."""
        if domain_name in DOMAIN_TAXONOMY:
            self.active_domains.add(domain_name)
            self.domain_ttls[domain_name] = ttl
            return {"status": "activated", "domain": domain_name, "tools": DOMAIN_TAXONOMY[domain_name]["tools"], "ttl": ttl}
        return {"status": "unknown_domain", "domain": domain_name}

    def step_turn_eviction(self) -> None:
        """Decrement domain TTLs and evict expired capabilities."""
        expired = []
        for dom, ttl in list(self.domain_ttls.items()):
            if ttl is not None:
                new_ttl = ttl - 1
                if new_ttl <= 0:
                    expired.append(dom)
                else:
                    self.domain_ttls[dom] = new_ttl

        for dom in expired:
            if dom != "core.code":
                self.active_domains.discard(dom)
                self.domain_ttls.pop(dom, None)
