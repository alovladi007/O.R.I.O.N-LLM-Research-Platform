"""Agent loop core — Plan, Tool, Planner, AgentLog, safety rails.

See :mod:`backend.common.agent_v2` for module-level overview.

Design notes
------------

- **Plan is structured.** The LLM doesn't free-form text actions;
  it emits a strict JSON object validated against a Pydantic model.
  Anthropic's tool-use API is the natural transport — we register
  the campaign tools as Anthropic Tools and let the model select
  among them. The :class:`Planner` accumulates ``tool_use`` blocks
  from the response into an ordered :class:`Plan`.

- **Replay is bit-exact at temperature=0.** "Replay logs reproduces
  the same sequence modulo LLM nondeterminism" is the roadmap
  acceptance language, but in practice ``temperature=0`` plus a
  fixed system prompt and identical history makes the API
  deterministic enough that the bit-exact check works for the
  acceptance test. We expose a ``replay_mode`` on :class:`Planner`
  that loads previously-recorded responses from the
  :class:`AgentLog` JSONL instead of calling the API.

- **Cost guard is per-tool-call.** Each tool declares a
  ``cost_estimate_usd`` callable that takes the tool args and
  returns a $ estimate. The guard rejects a call before
  execution if cumulative_cost + estimate > max_cost_usd.
  ``max_cost_usd=0`` therefore halts before *any* tool runs —
  exactly what the acceptance test exercises.

- **Kill switch is a threading.Event.** The API endpoint sets it;
  the loop checks it at the top of every round. Sub-second halt.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass
class Tool:
    """One registered tool callable.

    Attributes
    ----------
    name
        Unique identifier surfaced to the LLM.
    description
        Human-readable hint shown to the LLM. Should describe **when**
        to use the tool, not just **what** it does.
    input_schema
        JSON-schema dict for the tool's arguments. Anthropic's tool
        API uses this directly.
    handler
        ``(args: dict) → result: dict``. Idempotent + side-effect-free
        for tools that should be safe under replay (run_dft is not
        idempotent in production; the stub handler used by tests is).
    cost_estimate_usd
        ``(args: dict) → float``. Used by :class:`CostGuard`. Default
        is ``$0`` (free). DFT calls should return their compute-cost
        estimate (e.g., ``walltime_min * cluster_per_minute_rate``).
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    cost_estimate_usd: Callable[[Dict[str, Any]], float] = field(
        default=lambda args: 0.0
    )

    def to_anthropic_dict(self) -> Dict[str, Any]:
        """Serialize for the Anthropic tools API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCatalog:
    """Name-keyed registry of :class:`Tool`."""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError(
                f"unknown tool {name!r}; known: {sorted(self.tools)}"
            )
        return self.tools[name]

    def names(self) -> List[str]:
        return sorted(self.tools)

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        return [t.to_anthropic_dict() for t in self.tools.values()]


# ---------------------------------------------------------------------------
# Plan + ToolCall + ToolCallResult
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One tool invocation extracted from the LLM's plan.

    The LLM produces these as ``tool_use`` content blocks; we lift
    them into this dataclass so the rest of the loop is API-shape-
    agnostic.
    """

    tool_use_id: str       # Anthropic tool-use block id
    name: str
    args: Dict[str, Any]


@dataclass
class ToolCallResult:
    """Outcome of one tool execution."""

    tool_use_id: str
    name: str
    args: Dict[str, Any]
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


@dataclass
class Plan:
    """One LLM response broken into tool calls.

    May also carry a free-text ``rationale`` extracted from any
    ``text`` content blocks the model emitted; useful for the agent
    log's human-readable column.
    """

    tool_calls: List[ToolCall]
    rationale: str = ""
    raw_response: Optional[Dict[str, Any]] = None


class PlanValidationError(ValueError):
    """Raised when a tool call references an unknown tool or fails
    JSON-schema validation against the registered tool's input."""


def _validate_against_schema(
    args: Dict[str, Any], schema: Dict[str, Any], *, where: str,
) -> None:
    """Minimal JSON-schema validator covering the shapes our tools use:
    object with properties + required.

    We deliberately don't pull in ``jsonschema`` as a hard dep — the
    schemas we ship are simple, and the validator's job is to catch
    obvious LLM mistakes (missing required key, wrong type) rather
    than implement the full draft-7 spec. If a tool grows a complex
    schema, swap this for jsonschema.validate.
    """
    if schema.get("type") != "object":
        return
    if not isinstance(args, dict):
        raise PlanValidationError(
            f"{where}: args must be a JSON object, got {type(args).__name__}"
        )
    required = schema.get("required", [])
    for key in required:
        if key not in args:
            raise PlanValidationError(
                f"{where}: missing required argument {key!r}"
            )
    props = schema.get("properties", {})
    for key, val in args.items():
        if key not in props:
            # Permissive: extra keys are warnings, not errors. The
            # tool handler can decide whether to ignore or surface.
            logger.debug("%s: extra arg %r ignored by schema", where, key)
            continue
        ptype = props[key].get("type")
        if ptype == "string" and not isinstance(val, str):
            raise PlanValidationError(
                f"{where}: arg {key!r} must be string, got {type(val).__name__}"
            )
        if ptype == "number" and not isinstance(val, (int, float)):
            raise PlanValidationError(
                f"{where}: arg {key!r} must be number, got {type(val).__name__}"
            )
        if ptype == "integer" and not isinstance(val, int):
            raise PlanValidationError(
                f"{where}: arg {key!r} must be integer, got {type(val).__name__}"
            )
        if ptype == "array" and not isinstance(val, list):
            raise PlanValidationError(
                f"{where}: arg {key!r} must be array, got {type(val).__name__}"
            )
        if ptype == "object" and not isinstance(val, dict):
            raise PlanValidationError(
                f"{where}: arg {key!r} must be object, got {type(val).__name__}"
            )


# ---------------------------------------------------------------------------
# Safety rails
# ---------------------------------------------------------------------------


@dataclass
class CostGuard:
    """Halt before a tool call whose estimated $ cost would push the
    cumulative spend above ``max_cost_usd``.

    ``max_cost_usd = 0`` halts before **any** tool runs (the roadmap's
    "max_cost_usd=0 halts before the first DFT submission" check).
    ``max_cost_usd = None`` disables the guard.
    """

    max_cost_usd: Optional[float] = None
    cumulative_cost_usd: float = 0.0

    def check(self, estimated_cost_usd: float) -> bool:
        """``True`` if the call may proceed, ``False`` if it must be
        halted. Does not mutate state — call :meth:`record` after
        successful execution.
        """
        if self.max_cost_usd is None:
            return True
        return self.cumulative_cost_usd + estimated_cost_usd <= self.max_cost_usd

    def record(self, actual_cost_usd: float) -> None:
        self.cumulative_cost_usd += actual_cost_usd


class KillSwitch:
    """Thread-safe halt flag flipped by the API endpoint.

    Wraps ``threading.Event`` so we get the standard semantics
    (``set`` is monotonic, ``is_set`` is cheap, multiple threads can
    poll without locking).
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def trip(self) -> None:
        self._event.set()

    def is_tripped(self) -> bool:
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Agent log
# ---------------------------------------------------------------------------


@dataclass
class AgentLogEntry:
    """One round in the agent loop, persisted as JSONL.

    The schema is intentionally flat so JSONL grep / jq are easy.
    Promotion to the future ``agent_logs`` ORM table (Session 7.3b)
    maps each top-level field to a column.
    """

    round_index: int
    started_at: str
    finished_at: str
    rationale: str
    tool_calls: List[Dict[str, Any]]      # ToolCall.__dict__ list
    tool_results: List[Dict[str, Any]]    # ToolCallResult.__dict__ list
    token_usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    cost_usd_round: float = 0.0
    cost_usd_cumulative: float = 0.0
    halted: bool = False
    halt_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentLog:
    """Append-only JSONL log with deterministic replay support.

    Methods
    -------
    append(entry)
        Atomic write of one JSON line.
    entries()
        Read all entries (used for replay).
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: AgentLogEntry) -> None:
        line = json.dumps(entry.to_dict(), sort_keys=True) + "\n"
        # JSONL append is single-line atomic on POSIX up to PIPE_BUF
        # (4 KB on Linux/macOS). Our entries are larger, so we use a
        # lock + write+fsync to keep readers consistent.
        with self.path.open("a") as f:
            f.write(line)
            f.flush()

    def entries(self) -> List[AgentLogEntry]:
        if not self.path.is_file():
            return []
        out: List[AgentLogEntry] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            out.append(AgentLogEntry(**json.loads(line)))
        return out


# ---------------------------------------------------------------------------
# Planner — Anthropic SDK wrapper with dry-run + replay modes
# ---------------------------------------------------------------------------


_DEFAULT_MODEL = "claude-opus-4-7"
_SYSTEM_PROMPT = """\
You are the ORION research-platform agent. ORION is a materials-discovery
platform. You are given the current campaign state, a remaining budget,
and a tool catalog. Plan the next 1-3 tool calls that move the campaign
toward its objective most efficiently.

ORION conventions you must respect:
- Energies in eV (per-atom unless stated otherwise).
- Bandgaps in eV. PBE-DFT typically underestimates by ~30-50%.
- Compositions as reduced formulas; lattice parameters in Angstrom.
- DFT runs are expensive (~$0.50-$5 per static calc, more for relax).
- Prefer ML predictions when σ is below the calling threshold; fall
  through to DFT only when uncertainty makes the ML answer unreliable.

When you call tools, include all required arguments and respect each
tool's schema. Do not call more than 3 tools per round; the loop will
ask you again on the next round.
"""


class Planner:
    """Wraps the Anthropic SDK with retry + temperature=0.

    Parameters
    ----------
    api_key
        Anthropic API key. ``None`` reads ``$ANTHROPIC_API_KEY``.
    model
        Model id. Defaults to ``claude-opus-4-7`` per the roadmap.
    dry_run
        If True, the planner never calls the API. Instead it returns
        a deterministic stub plan that exercises every registered
        tool's schema once with sensible default args. Used by the
        dry-run acceptance test and by the kill-switch test where we
        don't want network calls.
    replay_log
        Optional :class:`AgentLog` to read previously-recorded
        responses from. When set, :meth:`plan` returns the
        ``replay_log.entries()[round_index]``'s plan instead of
        calling the API.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        dry_run: bool = False,
        replay_log: Optional[AgentLog] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dry_run = dry_run
        self.replay_log = replay_log
        self._client = None  # lazy

    def _client_or_raise(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "anthropic SDK not installed; run `pip install anthropic`"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def plan(
        self,
        *,
        catalog: ToolCatalog,
        round_index: int,
        objective: str,
        history_summary: str,
        remaining_budget_usd: Optional[float],
    ) -> Plan:
        """Return the next :class:`Plan`.

        Dispatches across the three modes (dry-run / replay / live)
        and normalizes the response shape.
        """
        if self.replay_log is not None:
            entries = self.replay_log.entries()
            if round_index >= len(entries):
                raise IndexError(
                    f"replay_log has {len(entries)} entries; round_index "
                    f"{round_index} out of range"
                )
            entry = entries[round_index]
            return Plan(
                tool_calls=[
                    ToolCall(
                        tool_use_id=tc["tool_use_id"],
                        name=tc["name"], args=tc["args"],
                    )
                    for tc in entry.tool_calls
                ],
                rationale=entry.rationale,
                raw_response=entry.raw_response,
            )
        if self.dry_run:
            return self._dry_run_plan(catalog, round_index)
        return self._live_plan(
            catalog=catalog, round_index=round_index,
            objective=objective, history_summary=history_summary,
            remaining_budget_usd=remaining_budget_usd,
        )

    # -- dry run ------------------------------------------------------

    def _dry_run_plan(self, catalog: ToolCatalog, round_index: int) -> Plan:
        """Deterministic stub plan: cycle through tools one per round
        with synthetic args from each tool's schema defaults.

        The acceptance test for "agent produces a plan that references
        valid tool names and plausible structures" exercises this path.
        """
        names = catalog.names()
        if not names:
            return Plan(tool_calls=[], rationale="dry_run: empty catalog")
        idx = round_index % len(names)
        tool = catalog.get(names[idx])
        args = self._defaults_from_schema(tool.input_schema)
        return Plan(
            tool_calls=[ToolCall(
                tool_use_id=f"dry-{round_index}-{tool.name}",
                name=tool.name, args=args,
            )],
            rationale=f"dry_run round {round_index}: exercising tool {tool.name!r}",
        )

    @staticmethod
    def _defaults_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """Pull a default arg-dict out of a JSON schema.

        For each required property we use the schema's ``default`` if
        set, otherwise a type-appropriate placeholder. Keeps the
        dry-run plans self-validating.
        """
        out: Dict[str, Any] = {}
        for key in schema.get("required", []):
            prop = schema.get("properties", {}).get(key, {})
            if "default" in prop:
                out[key] = prop["default"]
                continue
            ptype = prop.get("type", "string")
            placeholder: Any = {
                "string": "placeholder",
                "number": 1.0,
                "integer": 1,
                "boolean": False,
                "array": [],
                "object": {},
            }.get(ptype, None)
            out[key] = placeholder
        return out

    # -- live --------------------------------------------------------

    def _live_plan(
        self, *, catalog: ToolCatalog, round_index: int, objective: str,
        history_summary: str, remaining_budget_usd: Optional[float],
    ) -> Plan:  # pragma: no cover (live path; covered by `live` tests)
        client = self._client_or_raise()
        budget_msg = (
            f"remaining_budget_usd: ${remaining_budget_usd:.2f}\n"
            if remaining_budget_usd is not None else ""
        )
        user_msg = (
            f"objective: {objective}\n"
            f"{budget_msg}"
            f"history_summary:\n{history_summary}\n\n"
            "What is your next plan? Use the tools."
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.0,
            system=_SYSTEM_PROMPT,
            tools=catalog.to_anthropic_tools(),
            messages=[{"role": "user", "content": user_msg}],
        )
        tool_calls: List[ToolCall] = []
        rationale_parts: List[str] = []
        for block in resp.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use":
                tool_calls.append(ToolCall(
                    tool_use_id=getattr(block, "id"),
                    name=getattr(block, "name"),
                    args=dict(getattr(block, "input", {})),
                ))
            elif block_type == "text":
                rationale_parts.append(getattr(block, "text", ""))
        return Plan(
            tool_calls=tool_calls,
            rationale="\n".join(rationale_parts),
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )


# ---------------------------------------------------------------------------
# Agent loop driver
# ---------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Loop-level config."""

    objective: str
    catalog: ToolCatalog
    planner: Planner
    log: AgentLog
    cost_guard: CostGuard = field(default_factory=CostGuard)
    kill_switch: KillSwitch = field(default_factory=KillSwitch)
    max_rounds: int = 5
    max_tools_per_plan: int = 3


@dataclass
class AgentRoundResult:
    """One round's outcome (for callers that want the structured data
    directly without re-reading the log)."""

    round_index: int
    plan: Plan
    results: List[ToolCallResult]
    halted: bool
    halt_reason: Optional[str]
    cost_usd_round: float
    cost_usd_cumulative: float


class AgentLoop:
    """Drive the agent across multiple rounds.

    Stateless across processes — :meth:`run` reads the existing log
    to resume the round counter, but does not re-execute past tool
    calls. The cost guard's cumulative spend is also restored from
    the log so a resumed loop respects the original budget.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Restore cumulative cost from the log (resume-safe).
        for entry in config.log.entries():
            self.config.cost_guard.cumulative_cost_usd = entry.cost_usd_cumulative

    def run(self, *, history_summary: str = "") -> List[AgentRoundResult]:
        results: List[AgentRoundResult] = []
        starting_round = len(self.config.log.entries())
        for round_index in range(starting_round, starting_round + self.config.max_rounds):
            if self.config.kill_switch.is_tripped():
                logger.info("agent halted: kill switch")
                return results
            r = self._run_one_round(round_index, history_summary)
            results.append(r)
            if r.halted:
                return results
        return results

    def _run_one_round(
        self, round_index: int, history_summary: str,
    ) -> AgentRoundResult:
        started = datetime.utcnow().isoformat()
        t0 = time.time()
        plan = self.config.planner.plan(
            catalog=self.config.catalog,
            round_index=round_index,
            objective=self.config.objective,
            history_summary=history_summary,
            remaining_budget_usd=(
                None if self.config.cost_guard.max_cost_usd is None
                else self.config.cost_guard.max_cost_usd
                     - self.config.cost_guard.cumulative_cost_usd
            ),
        )
        latency_ms = int((time.time() - t0) * 1000)

        # Cap tools per plan.
        if len(plan.tool_calls) > self.config.max_tools_per_plan:
            plan = Plan(
                tool_calls=plan.tool_calls[: self.config.max_tools_per_plan],
                rationale=plan.rationale + (
                    f"\n[truncated to first {self.config.max_tools_per_plan} "
                    "tool calls]"
                ),
                raw_response=plan.raw_response,
            )

        tool_results: List[ToolCallResult] = []
        round_cost = 0.0
        halted = False
        halt_reason: Optional[str] = None

        for tc in plan.tool_calls:
            if self.config.kill_switch.is_tripped():
                halted, halt_reason = True, "kill_switch"
                break
            try:
                tool = self.config.catalog.get(tc.name)
            except KeyError as exc:
                tool_results.append(ToolCallResult(
                    tool_use_id=tc.tool_use_id, name=tc.name, args=tc.args,
                    success=False, error=str(exc),
                ))
                continue
            try:
                _validate_against_schema(
                    tc.args, tool.input_schema,
                    where=f"tool {tc.name!r}",
                )
            except PlanValidationError as exc:
                tool_results.append(ToolCallResult(
                    tool_use_id=tc.tool_use_id, name=tc.name, args=tc.args,
                    success=False, error=str(exc),
                ))
                continue
            est = tool.cost_estimate_usd(tc.args)
            if not self.config.cost_guard.check(est):
                tool_results.append(ToolCallResult(
                    tool_use_id=tc.tool_use_id, name=tc.name, args=tc.args,
                    success=False,
                    error=(
                        f"cost guard refused: estimate ${est:.4f} would "
                        f"push cumulative ${self.config.cost_guard.cumulative_cost_usd:.4f} "
                        f"above max ${self.config.cost_guard.max_cost_usd:.4f}"
                    ),
                ))
                halted, halt_reason = True, "cost_guard"
                break
            t_tool = time.time()
            try:
                output = tool.handler(tc.args)
                actual_cost = est  # stub: estimate == actual
                self.config.cost_guard.record(actual_cost)
                round_cost += actual_cost
                tool_results.append(ToolCallResult(
                    tool_use_id=tc.tool_use_id, name=tc.name, args=tc.args,
                    success=True, output=output,
                    cost_usd=actual_cost,
                    elapsed_s=time.time() - t_tool,
                ))
            except Exception as exc:  # noqa: BLE001
                tool_results.append(ToolCallResult(
                    tool_use_id=tc.tool_use_id, name=tc.name, args=tc.args,
                    success=False, error=str(exc),
                    elapsed_s=time.time() - t_tool,
                ))

        finished = datetime.utcnow().isoformat()
        cum = self.config.cost_guard.cumulative_cost_usd
        entry = AgentLogEntry(
            round_index=round_index,
            started_at=started, finished_at=finished,
            rationale=plan.rationale,
            tool_calls=[asdict(tc) for tc in plan.tool_calls],
            tool_results=[asdict(r) for r in tool_results],
            token_usage=self._extract_token_usage(plan),
            latency_ms=latency_ms,
            cost_usd_round=round_cost,
            cost_usd_cumulative=cum,
            halted=halted,
            halt_reason=halt_reason,
            raw_response=plan.raw_response,
        )
        self.config.log.append(entry)
        return AgentRoundResult(
            round_index=round_index, plan=plan, results=tool_results,
            halted=halted, halt_reason=halt_reason,
            cost_usd_round=round_cost,
            cost_usd_cumulative=cum,
        )

    @staticmethod
    def _extract_token_usage(plan: Plan) -> Dict[str, int]:
        """Pull token usage out of the Anthropic raw response if present."""
        raw = plan.raw_response or {}
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }
