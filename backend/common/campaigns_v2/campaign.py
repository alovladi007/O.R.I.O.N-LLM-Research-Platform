"""Campaign engine — BO-driven optimization loop with persistent steps.

A :class:`Campaign` runs the loop:

    while not halted:
        1. ask BO engine for ``q`` next points (Session 7.1 ``suggest``)
        2. for each point: scorer(point) → ScorerResult(value, sigma, …)
        3. atomically persist an :class:`AgentStep` row
        4. update history, check halting

The persistence boundary is :class:`StateStore` — :class:`JsonStateStore`
gives a real on-disk implementation that the acceptance test exercises
for the crash-resume check. A future :class:`DbStateStore` (Session
7.2b) wraps the existing ``DesignCampaign`` / ``DesignIteration`` ORM
models and keeps the engine logic identical.

Design notes
------------

- **Scorer composition.** The roadmap calls for "ML prediction → DFT
  if σ > threshold → score from DFT". That's caller code, not engine
  code — a scorer is just ``(point) → ScorerResult``. The
  composition examples in the docstring of :class:`Scorer` show the
  ML-then-DFT pattern; the engine's job is only to call the scorer
  and persist the result.

- **Resume without double-counting.** The state file is written
  *after* the scorer returns, so a crash mid-scorer leaves the file
  in its pre-step state. On resume the engine re-runs the BO ask
  with the same seed (deterministic) and the campaign continues
  exactly where it left off. The acceptance test verifies this by
  raising in the scorer for a deliberately-targeted step, then
  reloading the campaign and confirming step counts haven't drifted
  + the next call produces the same suggestion as the original.

- **Halting.** Three reasons (mirroring the roadmap):
  ``BUDGET_EXHAUSTED``, ``NO_IMPROVEMENT_K``, ``TARGET_REACHED``.
  The engine evaluates them after every step in that priority order
  so the first applicable one wins.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
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
    Tuple,
)

import numpy as np

from ..ml.bo_v2 import (
    History,
    LinearInequality,
    Objective,
    Space,
    initial_design,
    suggest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status / halting
# ---------------------------------------------------------------------------


class CampaignStatus(str, Enum):
    """Campaign lifecycle. Mirrors the existing ``DesignCampaign`` enum
    so the future DB-promotion is a label-for-label match."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class HaltReason(str, Enum):
    """Why a campaign reached a terminal state."""

    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    NO_IMPROVEMENT_K = "NO_IMPROVEMENT_K"
    TARGET_REACHED = "TARGET_REACHED"
    CANCELLED = "CANCELLED"
    NOT_HALTED = "NOT_HALTED"


@dataclass
class HaltingCriteria:
    """All three halting knobs from the roadmap.

    Attributes
    ----------
    max_steps
        Hard upper bound on completed steps. Always present.
    no_improvement_k
        Halt after K consecutive steps with no improvement in best-so-
        far. ``None`` disables this check.
    target_value
        Halt when best-so-far reaches this value. For minimization,
        the comparison is ``best <= target_value``; for maximization
        it's ``best >= target_value``. ``None`` disables.
    """

    max_steps: int = 30
    no_improvement_k: Optional[int] = None
    target_value: Optional[float] = None


# ---------------------------------------------------------------------------
# Scorer protocol
# ---------------------------------------------------------------------------


@dataclass
class ScorerResult:
    """One point's evaluation.

    Attributes
    ----------
    values
        Objective values **in the order of** ``CampaignConfig.objectives``.
        For single-objective campaigns this is ``[value]``.
    sigma
        Per-objective uncertainty. ``None`` if the scorer doesn't
        report one (e.g., closed-form synthetic functions).
    metadata
        Free-form dict for the scorer to attach provenance — typically
        ``{"job_id": "...", "via": "ml_then_dft", "elapsed_s": ...}``.
        Persisted with the step row.
    """

    values: List[float]
    sigma: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Scorer(Protocol):
    """Callable mapping a decoded point dict to a :class:`ScorerResult`.

    Examples
    --------

    .. code:: python

        # Closed-form synthetic (acceptance test).
        def scorer(point: dict) -> ScorerResult:
            return ScorerResult(values=[branin(point["x1"], point["x2"])])

        # Materials: ML predict, fall through to DFT if uncertain.
        def scorer(point: dict) -> ScorerResult:
            mu, sigma = ml_model.predict_with_uncertainty(point)
            if sigma > threshold:
                value = run_dft_blocking(point).bandgap_ev
                return ScorerResult(values=[value], sigma=[0.0],
                                    metadata={"via": "dft", "ml_mu": mu})
            return ScorerResult(values=[mu], sigma=[sigma],
                                metadata={"via": "ml"})
    """

    def __call__(self, point: Dict[str, Any]) -> ScorerResult: ...


# ---------------------------------------------------------------------------
# Step + snapshot data structures
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One executed step.

    Persisted as a row in the state store; serializable to JSON.
    Mirrors the ``DesignIteration`` ORM model closely enough that the
    7.2b promotion can map field-by-field without a translator.
    """

    step_index: int
    point: Dict[str, Any]
    values: List[float]
    sigma: Optional[List[float]]
    cumulative_best: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentStep":
        return cls(**d)


@dataclass
class CampaignSnapshot:
    """The state store payload — everything needed to resume a campaign."""

    id: str
    name: str
    status: str
    halt_reason: str
    created_at: str
    updated_at: str
    config_json: Dict[str, Any]
    steps: List[Dict[str, Any]]  # AgentStep.to_dict() rows
    best_value: Optional[float] = None
    best_step_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CampaignSnapshot":
        return cls(**d)


# ---------------------------------------------------------------------------
# State store
# ---------------------------------------------------------------------------


class StateStore(Protocol):
    """Persistence boundary. Two operations: load + save (atomic)."""

    def load(self) -> Optional[CampaignSnapshot]: ...

    def save(self, snapshot: CampaignSnapshot) -> None: ...


@dataclass
class JsonStateStore:
    """Atomic JSON-file state store.

    Writes go through a ``tempfile`` + ``os.replace`` so a crash
    between ``open`` and ``flush`` can't leave a half-file. The
    acceptance test for "kill mid-step" relies on this — the engine
    only calls :meth:`save` *after* the scorer returns, so any crash
    inside the scorer leaves the on-disk state at the prior step.
    """

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[CampaignSnapshot]:
        if not self.path.is_file():
            return None
        with self._lock:
            text = self.path.read_text()
        return CampaignSnapshot.from_dict(json.loads(text))

    def save(self, snapshot: CampaignSnapshot) -> None:
        payload = json.dumps(snapshot.to_dict(), indent=2, sort_keys=True)
        with self._lock:
            # NamedTemporaryFile then os.replace — POSIX-atomic on the
            # same filesystem (which is guaranteed because we put the
            # tempfile in the same dir as the target).
            tmp = tempfile.NamedTemporaryFile(
                mode="w", dir=self.path.parent,
                prefix=f".{self.path.name}.tmp.",
                delete=False,
            )
            try:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp.close()
                os.replace(tmp.name, self.path)
            except Exception:
                # If we crash before the rename, clean the tmp file so
                # the directory doesn't accumulate cruft.
                try:
                    os.unlink(tmp.name)
                except FileNotFoundError:
                    pass
                raise


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CampaignConfig:
    """All inputs to a :class:`Campaign` run.

    The engine snapshots this dict-of-primitives into the state store
    so a resumed campaign reconstructs the same objectives and budget
    without trusting the live Python object.
    """

    name: str
    space: Space
    objectives: List[Objective]
    scorer: Scorer
    halting: HaltingCriteria = field(default_factory=HaltingCriteria)
    n_initial: int = 5
    q_per_step: int = 1
    inequalities: List[LinearInequality] = field(default_factory=list)
    seed: int = 0


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


class Campaign:
    """End-to-end campaign driver.

    Lifecycle:

    .. code:: python

        cfg = CampaignConfig(name="branin", space=..., objectives=...,
                             scorer=branin_scorer, halting=HaltingCriteria(max_steps=20))
        store = JsonStateStore(Path("/tmp/branin.json"))
        camp = Campaign(cfg, store)
        camp.run()  # blocks until halted; safe to call again to resume

    Attributes are read-only after :meth:`__init__`; the engine mutates
    its in-memory ``steps`` and ``status`` and persists after each step.
    """

    def __init__(self, config: CampaignConfig, store: StateStore) -> None:
        self.config = config
        self.store = store
        self.steps: List[AgentStep] = []
        self.status = CampaignStatus.CREATED
        self.halt_reason = HaltReason.NOT_HALTED
        self.id = str(uuid.uuid4())
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        # Resume from store if present.
        snap = store.load()
        if snap is not None:
            self.id = snap.id
            self.created_at = snap.created_at
            self.updated_at = snap.updated_at
            self.status = CampaignStatus(snap.status)
            self.halt_reason = HaltReason(snap.halt_reason)
            self.steps = [
                AgentStep.from_dict(s) for s in snap.steps
            ]
            logger.info(
                "campaign %s resumed from %s with %d steps "
                "(status=%s, halt_reason=%s)",
                config.name, getattr(store, "path", "store"),
                len(self.steps), self.status.value, self.halt_reason.value,
            )

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def run(self) -> List[AgentStep]:
        """Drive the loop until halted. Idempotent w.r.t. resume."""
        if self.status == CampaignStatus.COMPLETED:
            logger.info("campaign %s already COMPLETED", self.config.name)
            return self.steps
        self.status = CampaignStatus.RUNNING
        self._persist()

        try:
            while True:
                # Check halting *before* doing more work; resume-safe.
                reason = self._evaluate_halting()
                if reason != HaltReason.NOT_HALTED:
                    self.halt_reason = reason
                    self.status = CampaignStatus.COMPLETED
                    self._persist()
                    logger.info(
                        "campaign %s halted: %s after %d steps",
                        self.config.name, reason.value, len(self.steps),
                    )
                    return self.steps
                self._take_step()
        except KeyboardInterrupt:
            # User requested halt — promote the in-memory state to a
            # persisted CANCELLED so resume sees the right status.
            self.status = CampaignStatus.CANCELLED
            self.halt_reason = HaltReason.CANCELLED
            self._persist()
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_halting(self) -> HaltReason:
        c = self.config.halting
        n = len(self.steps)
        if n >= c.max_steps:
            return HaltReason.BUDGET_EXHAUSTED
        if c.target_value is not None and self.steps:
            best = self._best_value()
            assert best is not None
            primary_minimize = self.config.objectives[0].minimize
            hit = (
                best <= c.target_value if primary_minimize
                else best >= c.target_value
            )
            if hit:
                return HaltReason.TARGET_REACHED
        if c.no_improvement_k is not None and n >= c.no_improvement_k + 1:
            # Look at the last K+1 cumulative_best values; if the
            # earliest equals the latest, no improvement happened.
            window = [s.cumulative_best for s in self.steps[-(c.no_improvement_k + 1):]]
            if window[0] == window[-1]:
                return HaltReason.NO_IMPROVEMENT_K
        return HaltReason.NOT_HALTED

    def _best_value(self) -> Optional[float]:
        if not self.steps:
            return None
        # cumulative_best is updated per step; the last row's value
        # is the running best.
        return self.steps[-1].cumulative_best

    def _take_step(self) -> None:
        """One full step: ask BO → score → persist."""
        step_index = len(self.steps)
        # Build the History the BO engine wants. For the first
        # ``n_initial`` steps we use the Sobol initial design instead
        # of a GP fit (BoTorch needs ≥ 2 observations to fit).
        if step_index < self.config.n_initial:
            point = self._sobol_point(step_index)
        else:
            point = self._bo_point(step_index)
        started = datetime.utcnow().isoformat()
        result = self.config.scorer(point)
        finished = datetime.utcnow().isoformat()
        if len(result.values) != len(self.config.objectives):
            raise ValueError(
                f"scorer returned {len(result.values)} values; "
                f"campaign has {len(self.config.objectives)} objectives"
            )
        cumulative_best = self._update_best(result.values[0])
        step = AgentStep(
            step_index=step_index,
            point=point,
            values=list(result.values),
            sigma=list(result.sigma) if result.sigma is not None else None,
            cumulative_best=cumulative_best,
            metadata=dict(result.metadata),
            started_at=started,
            finished_at=finished,
        )
        self.steps.append(step)
        # Persist *after* the step is fully built — a crash inside the
        # scorer leaves the on-disk state at step_index - 1.
        self._persist()
        logger.info(
            "campaign %s step %d: point=%s value=%.4f best=%.4f",
            self.config.name, step_index,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in point.items()},
            result.values[0], cumulative_best,
        )

    def _sobol_point(self, index: int) -> Dict[str, Any]:
        """Pull the ``index``-th point of a deterministic Sobol design.

        Re-seeded each call with the campaign seed + n_initial so the
        same index always yields the same point — that's what
        guarantees resume-after-crash idempotency for the init phase.
        """
        design = initial_design(
            self.config.space, n=self.config.n_initial, seed=self.config.seed,
        )
        return design[index]

    def _bo_point(self, step_index: int) -> Dict[str, Any]:
        """Suggest the next point from the BO engine.

        ``q_per_step`` > 1 is supported but only the first candidate
        is consumed per step — multi-point batches lengthen the
        cycle without changing the per-step persistence shape.
        """
        hist = History(objectives=list(self.config.objectives))
        for s in self.steps:
            hist.add(self.config.space.encode(s.point), s.values)
        # Per-step seed = base seed + step_index → reproducible AND
        # different per step (Sobol noise on consecutive steps is
        # important for exploration).
        cands = suggest(
            space=self.config.space,
            objectives=list(self.config.objectives),
            history=hist,
            q=self.config.q_per_step,
            inequalities=list(self.config.inequalities) or None,
            seed=self.config.seed + step_index,
        )
        return cands[0]

    def _update_best(self, latest: float) -> float:
        """Update best-so-far against the **first** objective only.

        For multi-objective campaigns the per-step ``cumulative_best``
        tracks the primary objective; the full Pareto front lives in
        the steps themselves and is computed on demand by callers
        (``backend.common.ml.bo_v2.pareto_front``).
        """
        primary_minimize = self.config.objectives[0].minimize
        if not self.steps:
            return latest
        prev = self._best_value()
        assert prev is not None
        if primary_minimize:
            return min(prev, latest)
        return max(prev, latest)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        self.updated_at = datetime.utcnow().isoformat()
        self.store.save(self._snapshot())

    def _snapshot(self) -> CampaignSnapshot:
        best = self._best_value()
        best_idx = (len(self.steps) - 1) if self.steps else None
        return CampaignSnapshot(
            id=self.id,
            name=self.config.name,
            status=self.status.value,
            halt_reason=self.halt_reason.value,
            created_at=self.created_at,
            updated_at=self.updated_at,
            config_json=self._config_to_jsonable(),
            steps=[s.to_dict() for s in self.steps],
            best_value=best,
            best_step_index=best_idx,
        )

    def _config_to_jsonable(self) -> Dict[str, Any]:
        """Serialize config primitives for the snapshot.

        We do not serialize the ``scorer`` (it's a Python callable);
        a resumed campaign must be reconstructed with the same scorer
        callable. The state store records *enough* to re-derive the
        BO suggest sequence given a matching scorer — which is all the
        roadmap acceptance check requires.
        """
        from dataclasses import is_dataclass

        def _serialise_dim(d):
            if is_dataclass(d):
                return {"kind": type(d).__name__, **asdict(d)}
            return d

        return {
            "objectives": [{"name": o.name, "minimize": o.minimize}
                           for o in self.config.objectives],
            "halting": asdict(self.config.halting),
            "n_initial": self.config.n_initial,
            "q_per_step": self.config.q_per_step,
            "inequalities": [asdict(c) for c in self.config.inequalities],
            "seed": self.config.seed,
            "space_dims": [_serialise_dim(d) for d in self.config.space.dims],
        }
