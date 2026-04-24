"""Tests for Phase 7 / Session 7.3 — agent loop.

Covers
------

1. Tool catalog mechanics — registration, duplicate detection,
   schema-default extraction, Anthropic-shape serialization.
2. ``CostGuard`` — refuses calls that would push cumulative spend
   above ``max_cost_usd``; a guard at $0 refuses *all* costed calls.
3. ``KillSwitch`` — thread-safe trip + idempotent.
4. ``AgentLog`` — JSONL round-trip, append-only, replay reads back
   the entries in order.
5. Plan validation — unknown tool name and bad arg type both rejected
   *before* the handler runs.
6. **Acceptance**:
   - Dry-run plan for "find oxide with bandgap ~2 eV" references
     valid tool names and produces argument shapes plausible for
     the registered tools.
   - Cost guard at ``max_cost_usd=0`` halts before the first DFT
     submission.
   - Replay mode (loading a previously-recorded :class:`AgentLog`)
     reproduces the same tool-call sequence — bit-exact at
     temperature=0.

The live LLM acceptance (real Anthropic call against
``claude-opus-4-7``) is gated by ``pytest.mark.live`` and excluded
from the default fast-suite gate the user set for Phase 7.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Tool catalog
# ---------------------------------------------------------------------------


class TestToolCatalog:
    def test_register_and_get(self):
        from backend.common.agent_v2 import Tool, ToolCatalog

        cat = ToolCatalog()
        cat.register(Tool(
            name="t1", description="d", input_schema={"type": "object"},
            handler=lambda args: {"ok": True},
        ))
        assert cat.get("t1").name == "t1"
        assert cat.names() == ["t1"]

    def test_duplicate_register_raises(self):
        from backend.common.agent_v2 import Tool, ToolCatalog

        cat = ToolCatalog()
        cat.register(Tool(
            name="t1", description="d", input_schema={"type": "object"},
            handler=lambda args: {},
        ))
        with pytest.raises(ValueError, match="already registered"):
            cat.register(Tool(
                name="t1", description="d2", input_schema={"type": "object"},
                handler=lambda args: {},
            ))

    def test_unknown_get_raises_keyerror(self):
        from backend.common.agent_v2 import ToolCatalog

        with pytest.raises(KeyError, match="unknown tool"):
            ToolCatalog().get("nope")

    def test_default_catalog_has_four_roadmap_tools(self):
        from backend.common.agent_v2 import default_tool_catalog

        cat = default_tool_catalog()
        assert set(cat.names()) == {
            "structure_generator", "run_dft", "run_ml_predict", "suggest_bo",
        }

    def test_to_anthropic_dict_shape(self):
        from backend.common.agent_v2 import default_tool_catalog

        cat = default_tool_catalog()
        anth = cat.to_anthropic_tools()
        assert isinstance(anth, list) and len(anth) == 4
        for t in anth:
            assert set(t) >= {"name", "description", "input_schema"}
            assert t["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Cost guard
# ---------------------------------------------------------------------------


class TestCostGuard:
    def test_no_max_means_unlimited(self):
        from backend.common.agent_v2 import CostGuard

        g = CostGuard(max_cost_usd=None)
        assert g.check(1e9) is True

    def test_check_does_not_mutate_state(self):
        from backend.common.agent_v2 import CostGuard

        g = CostGuard(max_cost_usd=1.0)
        g.check(0.5)
        g.check(0.5)
        assert g.cumulative_cost_usd == 0.0

    def test_record_advances_cumulative(self):
        from backend.common.agent_v2 import CostGuard

        g = CostGuard(max_cost_usd=10.0)
        g.record(2.0)
        g.record(1.5)
        assert g.cumulative_cost_usd == pytest.approx(3.5)

    def test_zero_max_refuses_any_costed_call(self):
        from backend.common.agent_v2 import CostGuard

        g = CostGuard(max_cost_usd=0.0)
        assert g.check(0.0) is True
        assert g.check(0.001) is False


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch:
    def test_starts_untripped(self):
        from backend.common.agent_v2 import KillSwitch

        assert KillSwitch().is_tripped() is False

    def test_trip_then_check(self):
        from backend.common.agent_v2 import KillSwitch

        ks = KillSwitch()
        ks.trip()
        assert ks.is_tripped() is True
        # Idempotent: tripping again still True.
        ks.trip()
        assert ks.is_tripped() is True


# ---------------------------------------------------------------------------
# AgentLog
# ---------------------------------------------------------------------------


class TestAgentLog:
    def test_append_then_entries(self, tmp_path: Path):
        from backend.common.agent_v2 import AgentLog, AgentLogEntry

        log = AgentLog(tmp_path / "agent.jsonl")
        log.append(AgentLogEntry(
            round_index=0, started_at="t0", finished_at="t1",
            rationale="r", tool_calls=[], tool_results=[],
        ))
        log.append(AgentLogEntry(
            round_index=1, started_at="t2", finished_at="t3",
            rationale="r2", tool_calls=[], tool_results=[],
        ))
        entries = log.entries()
        assert [e.round_index for e in entries] == [0, 1]

    def test_entries_on_missing_file(self, tmp_path: Path):
        from backend.common.agent_v2 import AgentLog

        assert AgentLog(tmp_path / "missing.jsonl").entries() == []


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------


class TestPlanValidation:
    def test_unknown_tool_yields_error_result(self, tmp_path: Path):
        from backend.common.agent_v2 import (
            AgentConfig, AgentLog, AgentLoop, Plan, Planner, ToolCall,
            default_tool_catalog,
        )

        class _StubPlanner(Planner):
            def plan(self, **kwargs):
                return Plan(tool_calls=[ToolCall(
                    tool_use_id="t0", name="bogus_tool", args={},
                )])

        cfg = AgentConfig(
            objective="x", catalog=default_tool_catalog(),
            planner=_StubPlanner(dry_run=True),
            log=AgentLog(tmp_path / "a.jsonl"),
            max_rounds=1,
        )
        loop = AgentLoop(cfg)
        rounds = loop.run()
        assert len(rounds) == 1
        res = rounds[0].results
        assert len(res) == 1 and res[0].success is False
        assert "unknown tool" in res[0].error

    def test_bad_arg_type_rejected_before_handler(self, tmp_path: Path):
        from backend.common.agent_v2 import (
            AgentConfig, AgentLog, AgentLoop, Plan, Planner, ToolCall,
            default_tool_catalog,
        )

        called = {"yes": False}
        cat = default_tool_catalog()

        # Wrap the structure_generator handler to detect any call.
        def watching_handler(args):
            called["yes"] = True
            return {}
        cat.tools["structure_generator"].handler = watching_handler

        class _StubPlanner(Planner):
            def plan(self, **kwargs):
                # n_candidates expects integer; supply a dict to fail.
                return Plan(tool_calls=[ToolCall(
                    tool_use_id="t0", name="structure_generator",
                    args={"prototype": "ABO3", "elements": ["O"],
                          "n_candidates": {"oops": 1}},
                )])

        cfg = AgentConfig(
            objective="x", catalog=cat,
            planner=_StubPlanner(dry_run=True),
            log=AgentLog(tmp_path / "a.jsonl"),
            max_rounds=1,
        )
        loop = AgentLoop(cfg)
        loop.run()
        assert called["yes"] is False, "handler ran despite schema violation"


# ---------------------------------------------------------------------------
# Roadmap acceptance
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Roadmap acceptance for Session 7.3.

    All three checks run against the dry-run / replay paths — no live
    LLM call. The live ``claude-opus-4-7`` smoke is in a separate
    ``pytest.mark.live`` test (skipped by default per the user's
    Phase 7 pytest gate).
    """

    def test_dry_run_plan_references_valid_tools_and_plausible_args(
        self, tmp_path: Path,
    ):
        """Roadmap: agent produces a plan for "find oxide with bandgap
        ~2 eV" that references valid tool names and plausible args.
        """
        from backend.common.agent_v2 import (
            AgentConfig, AgentLog, AgentLoop, CostGuard, Planner,
            default_tool_catalog,
        )

        cat = default_tool_catalog()
        log = AgentLog(tmp_path / "dry.jsonl")
        cfg = AgentConfig(
            objective="find oxide with bandgap ~2 eV",
            catalog=cat,
            planner=Planner(dry_run=True),
            log=log,
            cost_guard=CostGuard(max_cost_usd=10.0),
            max_rounds=4, max_tools_per_plan=3,
        )
        rounds = AgentLoop(cfg).run(history_summary="no prior history")

        assert len(rounds) == 4
        all_tool_names = [
            tc.name for r in rounds for tc in r.plan.tool_calls
        ]
        # All names valid.
        for n in all_tool_names:
            assert n in cat.names(), (
                f"plan referenced unknown tool {n!r}; valid: {cat.names()}"
            )
        # Plan exercises every roadmap tool at least once across 4 rounds.
        assert set(all_tool_names) == set(cat.names())
        # Every tool call's args validate against its schema (no errors
        # in the result rows).
        for r in rounds:
            for res in r.results:
                assert res.success is True, (
                    f"tool {res.name!r} dry-run failed: {res.error}"
                )

    def test_cost_guard_zero_halts_before_first_dft(self, tmp_path: Path):
        """Roadmap: configuring max_cost_usd=0 halts before the first
        DFT submission.

        We force the dry-run plan to start with run_dft (cycle index
        controls this; round 0 → 'run_dft' alphabetically). The cost
        estimate for run_dft is $0.50, so a $0 guard refuses it; the
        agent halts immediately with halt_reason='cost_guard'.
        """
        from backend.common.agent_v2 import (
            AgentConfig, AgentLog, AgentLoop, CostGuard, Planner,
            default_tool_catalog,
        )

        cat = default_tool_catalog()
        log = AgentLog(tmp_path / "guard.jsonl")
        cfg = AgentConfig(
            objective="x", catalog=cat,
            planner=Planner(dry_run=True),
            log=log,
            cost_guard=CostGuard(max_cost_usd=0.0),
            max_rounds=4,
        )
        rounds = AgentLoop(cfg).run()

        # First round must halt; cumulative cost must remain $0.
        assert rounds[0].halted is True
        assert rounds[0].halt_reason == "cost_guard"
        assert rounds[0].cost_usd_cumulative == 0.0
        # The first attempted tool call was rejected with the guard
        # error (not executed).
        assert any(
            res.success is False and "cost guard" in (res.error or "")
            for res in rounds[0].results
        ), f"expected a cost-guard rejection in round 0; got: {rounds[0].results}"

    def test_replay_reproduces_tool_call_sequence(self, tmp_path: Path):
        """Roadmap: every plan is stored; replaying logs reproduces
        the same sequence (modulo LLM nondeterminism — temperature=0
        gives bit-exact replay against a stub planner).
        """
        from backend.common.agent_v2 import (
            AgentConfig, AgentLog, AgentLoop, CostGuard, Planner,
            default_tool_catalog,
        )

        # 1. Original run with dry-run planner → records the log.
        cat = default_tool_catalog()
        log_path = tmp_path / "rec.jsonl"
        cfg1 = AgentConfig(
            objective="x", catalog=cat,
            planner=Planner(dry_run=True),
            log=AgentLog(log_path),
            cost_guard=CostGuard(max_cost_usd=10.0),
            max_rounds=3,
        )
        original = AgentLoop(cfg1).run()
        original_tools = [
            [tc.name for tc in r.plan.tool_calls] for r in original
        ]

        # 2. Replay against the recorded log — fresh agent, fresh
        # cost guard, but planner reads from log.
        log_replay = AgentLog(log_path)
        cfg2 = AgentConfig(
            objective="x", catalog=default_tool_catalog(),
            # Planner.plan() with replay_log set reads back the
            # recorded plans round-by-round.
            planner=Planner(replay_log=log_replay),
            # Use a separate write path so the replay log isn't
            # mutated by the rerun.
            log=AgentLog(tmp_path / "replay-write.jsonl"),
            cost_guard=CostGuard(max_cost_usd=10.0),
            max_rounds=3,
        )
        replayed = AgentLoop(cfg2).run()
        replayed_tools = [
            [tc.name for tc in r.plan.tool_calls] for r in replayed
        ]

        assert replayed_tools == original_tools, (
            f"replay diverged from original:\n"
            f"original: {original_tools}\nreplayed: {replayed_tools}"
        )

        # And the replayed log has exactly 3 entries (same as original).
        replayed_log = AgentLog(tmp_path / "replay-write.jsonl").entries()
        assert len(replayed_log) == 3
