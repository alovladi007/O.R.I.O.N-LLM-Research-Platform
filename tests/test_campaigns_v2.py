"""Tests for Phase 7 / Session 7.2 — campaign orchestrator.

Covers
------

1. ``HaltingCriteria`` — budget, no-improvement-K, target-reached
   each fire when their condition is met.
2. ``JsonStateStore`` — save/load round-trip; atomicity of the
   ``os.replace`` write so a crash mid-write doesn't corrupt the
   target file.
3. ``Campaign`` mechanics — step persistence, ``cumulative_best``
   tracking, scorer-shape validation, init-design idempotency.
4. **Acceptance**:
   - Toy synthetic-function campaign completes within budget and
     logs every step.
   - Killing the worker mid-campaign (raise in scorer at step k)
     and restarting resumes without double-counting and without
     re-running the completed steps.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quad_scorer(point: dict):
    """Closed-form ``(x - 0.3)²``; min at x=0.3, value=0."""
    from backend.common.campaigns_v2 import ScorerResult

    return ScorerResult(values=[(point["x"] - 0.3) ** 2])


def _branin_scorer(point: dict):
    """Standard Branin; min ≈ 0.398 at three locations."""
    from backend.common.campaigns_v2 import ScorerResult

    a, b = 1.0, 5.1 / (4 * math.pi ** 2)
    c, r = 5.0 / math.pi, 6.0
    s, t = 10.0, 1.0 / (8 * math.pi)
    x1, x2 = point["x1"], point["x2"]
    val = (
        a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
        + s * (1.0 - t) * math.cos(x1) + s
    )
    return ScorerResult(values=[val], metadata={"via": "synthetic"})


def _quad_config(tmp_path: Path, *, max_steps: int = 8, n_initial: int = 3):
    from backend.common.campaigns_v2 import (
        CampaignConfig, HaltingCriteria,
    )
    from backend.common.ml.bo_v2 import ContinuousDim, Objective, Space

    return CampaignConfig(
        name="quad",
        space=Space(dims=[ContinuousDim("x", -1.0, 1.0)]),
        objectives=[Objective("y", minimize=True)],
        scorer=_quad_scorer,
        halting=HaltingCriteria(max_steps=max_steps),
        n_initial=n_initial, seed=0,
    )


# ---------------------------------------------------------------------------
# Halting
# ---------------------------------------------------------------------------


class TestHalting:
    def test_budget_exhausted_after_max_steps(self, tmp_path: Path):
        from backend.common.campaigns_v2 import (
            Campaign, HaltReason, JsonStateStore,
        )

        cfg = _quad_config(tmp_path, max_steps=4, n_initial=3)
        store = JsonStateStore(tmp_path / "c.json")
        c = Campaign(cfg, store)
        c.run()
        assert len(c.steps) == 4
        assert c.halt_reason == HaltReason.BUDGET_EXHAUSTED

    def test_target_reached_halts_for_minimization(self, tmp_path: Path):
        from backend.common.campaigns_v2 import (
            Campaign, CampaignConfig, HaltReason, HaltingCriteria,
            JsonStateStore,
        )
        from backend.common.ml.bo_v2 import ContinuousDim, Objective, Space

        cfg = CampaignConfig(
            name="quad",
            space=Space(dims=[ContinuousDim("x", -1.0, 1.0)]),
            objectives=[Objective("y", minimize=True)],
            scorer=_quad_scorer,
            # Loose target — early stop the moment best <= 0.05.
            halting=HaltingCriteria(max_steps=20, target_value=0.05),
            n_initial=3, seed=0,
        )
        store = JsonStateStore(tmp_path / "c.json")
        c = Campaign(cfg, store)
        c.run()
        assert c.halt_reason == HaltReason.TARGET_REACHED
        assert c._best_value() <= 0.05
        # Should have stopped well short of the budget.
        assert len(c.steps) < 20

    def test_no_improvement_k_fires(self, tmp_path: Path):
        from backend.common.campaigns_v2 import (
            Campaign, CampaignConfig, HaltReason, HaltingCriteria,
            JsonStateStore, ScorerResult,
        )
        from backend.common.ml.bo_v2 import ContinuousDim, Objective, Space

        # Constant scorer → cumulative_best never improves → halts as
        # soon as we have K+1 = 4 steps with no improvement.
        def constant(point):
            return ScorerResult(values=[1.0])

        cfg = CampaignConfig(
            name="constant",
            space=Space(dims=[ContinuousDim("x", -1.0, 1.0)]),
            objectives=[Objective("y", minimize=True)],
            scorer=constant,
            halting=HaltingCriteria(max_steps=20, no_improvement_k=3),
            n_initial=3, seed=0,
        )
        store = JsonStateStore(tmp_path / "c.json")
        c = Campaign(cfg, store)
        c.run()
        assert c.halt_reason == HaltReason.NO_IMPROVEMENT_K
        # K+1 = 4 step look-back window; we halt at step 4 (zero-
        # indexed), so steps == 4.
        assert len(c.steps) == 4


# ---------------------------------------------------------------------------
# State store
# ---------------------------------------------------------------------------


class TestJsonStateStore:
    def test_save_then_load_roundtrips(self, tmp_path: Path):
        from backend.common.campaigns_v2 import (
            CampaignSnapshot, JsonStateStore,
        )

        snap = CampaignSnapshot(
            id="abc", name="x", status="RUNNING",
            halt_reason="NOT_HALTED",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:01",
            config_json={"k": 1},
            steps=[{"step_index": 0, "point": {"x": 0.5},
                    "values": [1.0], "sigma": None,
                    "cumulative_best": 1.0, "metadata": {},
                    "started_at": "", "finished_at": ""}],
            best_value=1.0, best_step_index=0,
        )
        store = JsonStateStore(tmp_path / "s.json")
        store.save(snap)
        loaded = store.load()
        assert loaded is not None
        assert loaded.id == "abc"
        assert loaded.steps[0]["point"] == {"x": 0.5}

    def test_load_missing_returns_none(self, tmp_path: Path):
        from backend.common.campaigns_v2 import JsonStateStore

        store = JsonStateStore(tmp_path / "missing.json")
        assert store.load() is None

    def test_atomic_replace_does_not_leak_tempfiles_on_success(
        self, tmp_path: Path,
    ):
        from backend.common.campaigns_v2 import (
            CampaignSnapshot, JsonStateStore,
        )

        snap = CampaignSnapshot(
            id="x", name="x", status="RUNNING", halt_reason="NOT_HALTED",
            created_at="t0", updated_at="t1",
            config_json={}, steps=[],
        )
        store = JsonStateStore(tmp_path / "s.json")
        for _ in range(3):
            store.save(snap)
        # No leftover .tmp files in the dir.
        leftovers = list(tmp_path.glob(".s.json.tmp.*"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# Campaign mechanics
# ---------------------------------------------------------------------------


class TestCampaignMechanics:
    def test_initial_design_used_for_first_n_steps(self, tmp_path: Path):
        """The first ``n_initial`` steps must come from the Sobol
        design — independent of any GP. Verify by reseeding with the
        same campaign config and checking the first n_initial
        ``point.x`` values match.
        """
        from backend.common.campaigns_v2 import Campaign, JsonStateStore
        from backend.common.ml.bo_v2 import initial_design

        cfg = _quad_config(tmp_path, max_steps=3, n_initial=3)
        c = Campaign(cfg, JsonStateStore(tmp_path / "c.json"))
        c.run()
        sobol = initial_design(cfg.space, n=3, seed=0)
        for step, expected in zip(c.steps, sobol):
            assert step.point["x"] == pytest.approx(expected["x"])

    def test_cumulative_best_is_monotone_non_increasing_for_minimization(
        self, tmp_path: Path,
    ):
        from backend.common.campaigns_v2 import Campaign, JsonStateStore

        cfg = _quad_config(tmp_path, max_steps=8, n_initial=3)
        c = Campaign(cfg, JsonStateStore(tmp_path / "c.json"))
        c.run()
        bests = [s.cumulative_best for s in c.steps]
        # For minimization, best can only decrease or stay flat.
        for i in range(1, len(bests)):
            assert bests[i] <= bests[i - 1] + 1e-12, (
                f"cumulative_best regressed at step {i}: "
                f"{bests[i - 1]} → {bests[i]}"
            )

    def test_scorer_shape_mismatch_raises(self, tmp_path: Path):
        from backend.common.campaigns_v2 import (
            Campaign, CampaignConfig, HaltingCriteria,
            JsonStateStore, ScorerResult,
        )
        from backend.common.ml.bo_v2 import ContinuousDim, Objective, Space

        def bad_scorer(point):
            return ScorerResult(values=[1.0, 2.0])  # 2 values, 1 obj

        cfg = CampaignConfig(
            name="bad",
            space=Space(dims=[ContinuousDim("x", -1.0, 1.0)]),
            objectives=[Objective("y")],
            scorer=bad_scorer,
            halting=HaltingCriteria(max_steps=2),
            n_initial=1, seed=0,
        )
        store = JsonStateStore(tmp_path / "c.json")
        c = Campaign(cfg, store)
        with pytest.raises(ValueError, match="returned 2 values"):
            c.run()

    def test_already_completed_run_is_noop(self, tmp_path: Path):
        from backend.common.campaigns_v2 import Campaign, JsonStateStore

        cfg = _quad_config(tmp_path, max_steps=3, n_initial=3)
        store = JsonStateStore(tmp_path / "c.json")
        c1 = Campaign(cfg, store)
        c1.run()
        n_before = len(c1.steps)
        # Reload + run again — should not add steps.
        c2 = Campaign(cfg, store)
        c2.run()
        assert len(c2.steps) == n_before


# ---------------------------------------------------------------------------
# Acceptance — toy synthetic campaign + crash-resume
# ---------------------------------------------------------------------------


class TestAcceptance:
    """Roadmap acceptance for Session 7.2.

    Both checks deliberately use the closed-form Branin function as
    the scorer — it's cheap, deterministic, and sensitive enough to
    detect double-counting bugs in the resume path.
    """

    def test_toy_campaign_completes_and_logs_every_step(self, tmp_path: Path):
        """Roadmap: a toy campaign optimizing a closed-form synthetic
        function completes within budget and logs every step.
        """
        from backend.common.campaigns_v2 import (
            Campaign, CampaignConfig, HaltReason, HaltingCriteria,
            JsonStateStore,
        )
        from backend.common.ml.bo_v2 import ContinuousDim, Objective, Space

        cfg = CampaignConfig(
            name="branin",
            space=Space(dims=[
                ContinuousDim("x1", -5.0, 10.0),
                ContinuousDim("x2", 0.0, 15.0),
            ]),
            objectives=[Objective("branin", minimize=True)],
            scorer=_branin_scorer,
            halting=HaltingCriteria(max_steps=12),
            n_initial=5, seed=0,
        )
        store = JsonStateStore(tmp_path / "branin.json")
        c = Campaign(cfg, store)
        steps = c.run()

        # Budget completed.
        assert len(steps) == 12
        assert c.halt_reason == HaltReason.BUDGET_EXHAUSTED
        # Every step logged with non-empty point + values + a finite
        # cumulative_best.
        for s in steps:
            assert set(s.point) == {"x1", "x2"}
            assert len(s.values) == 1 and math.isfinite(s.values[0])
            assert math.isfinite(s.cumulative_best)
            assert s.started_at and s.finished_at
            assert s.metadata.get("via") == "synthetic"
        # Persisted snapshot matches the in-memory state.
        on_disk = store.load()
        assert on_disk is not None
        assert len(on_disk.steps) == 12
        assert on_disk.best_value == pytest.approx(c._best_value())

    def test_crash_mid_step_resumes_without_double_counting(
        self, tmp_path: Path,
    ):
        """Roadmap: killing the worker mid-campaign and restarting
        resumes without double-counting.

        We simulate the crash by a scorer that raises ``RuntimeError``
        the first time it's called with step_index == 4 (zero-based —
        the 5th step). The persistence boundary is *after* the scorer
        returns, so the on-disk state has exactly 4 steps after the
        crash. Reloading and re-running with a non-raising scorer
        must complete to budget == 12 with no duplicate ``step_index``
        and no re-suggested-but-not-persisted points lost.
        """
        from backend.common.campaigns_v2 import (
            Campaign, CampaignConfig, HaltReason, HaltingCriteria,
            JsonStateStore, ScorerResult,
        )
        from backend.common.ml.bo_v2 import ContinuousDim, Objective, Space

        crash_at = 4
        crashed = {"value": False}

        def crashy_scorer(point):
            # The scorer doesn't know its step_index; we use a counter
            # plus the on-disk step count to decide whether to crash.
            store = JsonStateStore(tmp_path / "c.json")
            snap = store.load()
            current_step = 0 if snap is None else len(snap.steps)
            if current_step == crash_at and not crashed["value"]:
                crashed["value"] = True
                raise RuntimeError("simulated worker crash")
            return _branin_scorer(point)

        space = Space(dims=[
            ContinuousDim("x1", -5.0, 10.0),
            ContinuousDim("x2", 0.0, 15.0),
        ])
        cfg = CampaignConfig(
            name="branin-crash", space=space,
            objectives=[Objective("branin", minimize=True)],
            scorer=crashy_scorer,
            halting=HaltingCriteria(max_steps=12),
            n_initial=5, seed=0,
        )
        store = JsonStateStore(tmp_path / "c.json")
        c1 = Campaign(cfg, store)
        with pytest.raises(RuntimeError, match="simulated worker crash"):
            c1.run()

        # On-disk state should have exactly `crash_at` steps — the
        # crash happened *inside* the scorer, before persistence.
        on_disk = store.load()
        assert on_disk is not None
        assert len(on_disk.steps) == crash_at, (
            f"expected {crash_at} steps persisted, got {len(on_disk.steps)}"
        )
        assert on_disk.status == "RUNNING"

        # Resume with a non-crashing scorer.
        cfg_resume = CampaignConfig(
            name="branin-crash", space=space,
            objectives=[Objective("branin", minimize=True)],
            scorer=_branin_scorer,
            halting=HaltingCriteria(max_steps=12),
            n_initial=5, seed=0,
        )
        c2 = Campaign(cfg_resume, store)
        # Resume must pick up exactly where we left off.
        assert len(c2.steps) == crash_at
        c2.run()
        assert c2.halt_reason == HaltReason.BUDGET_EXHAUSTED
        assert len(c2.steps) == 12

        # No duplicate step_index — the canonical check.
        idxs = [s.step_index for s in c2.steps]
        assert idxs == list(range(12)), (
            f"step_index sequence broken: {idxs}"
        )
        # And the snapshot agrees with the in-memory state.
        on_disk = store.load()
        assert on_disk is not None
        assert len(on_disk.steps) == 12
        assert [s["step_index"] for s in on_disk.steps] == list(range(12))
