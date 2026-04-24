"""Phase 7 / Session 7.2 — Campaign orchestrator.

A :class:`Campaign` is ``{space, budget, objectives, scorer,
halting_criteria}`` per the roadmap, driven by the Session 7.1 BO
engine. Each step takes a BO suggestion, scores it (the scorer can
itself dispatch to ML / DFT / a closed-form synthetic function), and
appends an :class:`AgentStep` row to the campaign's persistent log.
Halting on budget exhaustion, no-improvement-for-K, or target-reached.

Persistence is **filesystem-JSON** for Session 7.2 — the campaign
state lives in a single JSON file per campaign that's atomically
re-written after each step. That gives us the resume-after-crash
property the roadmap acceptance requires without dragging in the DB
+ Alembic + Celery wiring (which lands in Session 7.2b alongside the
existing ``DesignCampaign`` ORM model). The DB-backed promotion is a
thin shim — :class:`Campaign` already takes a generic
:class:`StateStore` so swapping JSON for SQL is a drop-in.

The legacy :mod:`backend.common.campaigns` module
(``DesignLoopService``) is a different campaign idea (structure
generation + ML scoring against target properties); we leave it
unchanged. Anything new in Phase 7 lives in :mod:`campaigns_v2`.
"""

from .campaign import (
    AgentStep,
    Campaign,
    CampaignConfig,
    CampaignSnapshot,
    CampaignStatus,
    HaltReason,
    HaltingCriteria,
    JsonStateStore,
    Scorer,
    ScorerResult,
    StateStore,
)

__all__ = [
    "AgentStep",
    "Campaign",
    "CampaignConfig",
    "CampaignSnapshot",
    "CampaignStatus",
    "HaltReason",
    "HaltingCriteria",
    "JsonStateStore",
    "Scorer",
    "ScorerResult",
    "StateStore",
]
