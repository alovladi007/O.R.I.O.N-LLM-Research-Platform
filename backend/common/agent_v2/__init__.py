"""Phase 7 / Session 7.3 — LLM-driven agent loop.

The "LLM Research Platform" name earns itself here. An :class:`AgentLoop`
runs the cycle:

    while not halted:
        1. snapshot the campaign state (results, remaining budget,
           objectives, available tools)
        2. ask the LLM (Claude Opus 4.7 by default) for a structured
           ``Plan`` of next tool calls
        3. validate every tool call against the registered schema
        4. execute each tool call (or skip in dry-run mode)
        5. log the round to the :class:`AgentLog` (atomic JSONL)
        6. update budget, check cost guard / kill switch

The module ships:

- :class:`Tool` + :class:`ToolCatalog` — strongly-typed tool registry
  with JSON-schema validation of arguments and a Python handler that
  the loop calls. The four roadmap tools (structure_generator,
  run_dft, run_ml_predict, suggest_bo) are pre-registered with stub
  handlers in :mod:`tools` — real handlers wire in for 7.3b.
- :class:`Planner` — wraps the Anthropic SDK with retry + temperature=0
  for replay-determinism. Supports a ``dry_run`` mode that skips the
  HTTP call and returns a deterministic mock plan, used by the
  acceptance tests.
- :class:`AgentLog` — append-only JSONL on disk. Each entry records
  one LLM round: messages, tool calls, validation outcome, token
  usage, latency, cost estimate, kill-switch state.
- :class:`CostGuard` + :class:`KillSwitch` — safety rails. The cost
  guard halts before any tool call whose estimated dollar cost would
  exceed the configured budget. The kill switch is a thread-safe
  flag the API endpoint flips to immediately halt the loop.

Configuration model is the same JSON-state-store pattern as Session
7.2 — the loop snapshot lives on disk so a process crash doesn't lose
the agent's plan history. DB promotion (``agent_logs`` table +
Alembic) is Session 7.3b.
"""

from .agent import (
    AgentConfig,
    AgentLog,
    AgentLogEntry,
    AgentLoop,
    AgentRoundResult,
    CostGuard,
    KillSwitch,
    Plan,
    Planner,
    PlanValidationError,
    Tool,
    ToolCall,
    ToolCallResult,
    ToolCatalog,
)
from .tools import default_tool_catalog

__all__ = [
    "AgentConfig",
    "AgentLog",
    "AgentLogEntry",
    "AgentLoop",
    "AgentRoundResult",
    "CostGuard",
    "KillSwitch",
    "Plan",
    "Planner",
    "PlanValidationError",
    "Tool",
    "ToolCall",
    "ToolCallResult",
    "ToolCatalog",
    "default_tool_catalog",
]
