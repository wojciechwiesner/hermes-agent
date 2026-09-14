# Empirical Benchmark: JIT Tool Surface & Lazy Schema Hydration (Qwen 3.8 9B)

## 1. Overview
In standard AI agent runtimes, tool schemas are eagerly loaded into every prompt. In Hermes Agent, the default tool surface (25 native active tools, 42.6KB JSON) consumes over 10,000 tokens on Turn 1 before user instructions are processed.

This benchmark measures the impact of **JIT Tool Surface (Lazy Schema Hydration)** against local Qwen 3.8 9B running on Apple Silicon Metal (Ollama runtime).

---

## 2. Quantitative Results

Measured on Apple Silicon M2 Pro (Qwen 3.8 9B, Q4_K_M quantization via Ollama):

| Metric | Baseline: Eager Tools (25 tools) | JIT Tool Surface (5 core.code tools) | Delta |
|---|---|---|---|
| **Tool Schemas Injected** | 25 tools (42,661 chars) | **5 tools** (6,120 chars) | **-80.0% tools** |
| **Turn 1 Prompt Tokens** | 10,142 tokens | **2,220 tokens** | **-78.1% tokens** |
| **Prefill Latency (Prompt Eval)** | 44.96 s | **9.26 s** | **4.85× faster** |
| **Total Turn 1 Wall Time** | 63.10 s | **13.54 s** | **4.66× faster** |
| **16K Window Feasibility** | ⚠️ High Risk (61.9% consumed) |  Safe (13.5% consumed, >14k headroom) | **Enabled** |

---

## 3. The 3-Tier Tool Surface Comparison

To provide complete accounting across tool loading strategies:

1. **Unbounded Eager (58 native tools, no deferred tiering):**
   * Schema Tokens: ~34,812 tokens
   * Turn 1 Request: ~36,512 tokens
   * Feasibility on 16K Context: **0% (Immediate context length overflow)**

2. **Hermes Built-in Tool Search (Tier 1 Catalog):**
   * Active Tools: 25 native tools (27 deferred behind tool_search)
   * Prompt Tokens: **10,142 tokens**
   * Prefill Time on Metal: **44.96 seconds**

3. **JIT Tool Surface Hydration (Active Working Set):**
   * Active Tools: 5 `core.code` tools (`read_file`, `write_file`, `patch`, `search_files`, `terminal`)
   * Prompt Tokens: **2,220 tokens**
   * Prefill Time on Metal: **9.26 seconds** (**4.85x speedup over Tier 1**)

---

## 4. RequestBudgetGate Proof

Under a 16,384 (`16K`) runtime window:
* Reserved output: 2,048 tokens
* Safety margin: 512 tokens
* Safe budget ceiling: **13,824 tokens**

### Gate Evaluation:
* **Eager / Tier 1 (10,142 tokens):** Approaching ceiling; only 3,682 tokens available for file reads and code generation.
* **JIT Hydrated (2,220 tokens):** Formally validated by `RequestBudgetGate`. Yields **11,604 tokens of clean headroom**.
* Result on Wynajmujemy.xyz production benchmark: **Autonomous completion in 14 turns, 7.6 minutes, 25/25 tests passing ($0.00 cost)**.
