"""
Tests for the SimulationJob state machine (Session 1.4).

Most of the work lives in pure functions
(``src.api.models.simulation.check_transition`` and ``is_terminal_status``)
so we can exercise the state machine without touching SQLAlchemy ORM
state. The ``transition_to`` method on the model delegates to
``check_transition`` and adds timestamp / error_message side effects.

Per the roadmap:

    PENDING  → QUEUED | CANCELLED | FAILED
    QUEUED   → RUNNING | CANCELLED | FAILED
    RUNNING  → COMPLETED | FAILED | CANCELLED | TIMEOUT
    COMPLETED, FAILED, CANCELLED, TIMEOUT → (terminal, no outgoing)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Legal transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,target",
    [
        # PENDING
        ("PENDING", "QUEUED"),
        ("PENDING", "CANCELLED"),
        ("PENDING", "FAILED"),
        # QUEUED
        ("QUEUED", "RUNNING"),
        ("QUEUED", "CANCELLED"),
        ("QUEUED", "FAILED"),
        # RUNNING
        ("RUNNING", "COMPLETED"),
        ("RUNNING", "FAILED"),
        ("RUNNING", "CANCELLED"),
        ("RUNNING", "TIMEOUT"),
    ],
)
def test_legal_transition_accepted(source: str, target: str):
    from src.api.models.simulation import JobStatus, check_transition

    got = check_transition(source, target)
    assert got == JobStatus(target)


# ---------------------------------------------------------------------------
# Illegal transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,target",
    [
        # Can't skip states.
        ("PENDING", "RUNNING"),
        ("PENDING", "COMPLETED"),
        ("QUEUED", "COMPLETED"),
        # Can't go backwards.
        ("QUEUED", "PENDING"),
        ("RUNNING", "PENDING"),
        ("RUNNING", "QUEUED"),
        # Terminal states are terminal.
        ("COMPLETED", "RUNNING"),
        ("COMPLETED", "QUEUED"),
        ("FAILED", "COMPLETED"),
        ("FAILED", "RUNNING"),
        ("CANCELLED", "RUNNING"),
        ("CANCELLED", "PENDING"),
        ("TIMEOUT", "RUNNING"),
    ],
)
def test_illegal_transition_rejected(source: str, target: str):
    from src.api.models.simulation import IllegalJobTransitionError, check_transition

    with pytest.raises(IllegalJobTransitionError) as excinfo:
        check_transition(source, target)
    msg = str(excinfo.value)
    assert source in msg or source.lower() in msg.lower()
    assert target in msg or target.lower() in msg.lower()


def test_error_message_enumerates_legal_targets():
    from src.api.models.simulation import IllegalJobTransitionError, check_transition

    with pytest.raises(IllegalJobTransitionError) as excinfo:
        check_transition("RUNNING", "QUEUED")
    # Message must name what RUNNING *can* go to — users depend on this.
    msg = str(excinfo.value)
    for legal in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
        assert legal in msg, f"expected {legal} to appear in error message"


# ---------------------------------------------------------------------------
# Terminal-state predicates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"])
def test_terminal_statuses_are_terminal(status: str):
    from src.api.models.simulation import is_terminal_status

    assert is_terminal_status(status) is True


@pytest.mark.parametrize("status", ["PENDING", "QUEUED", "RUNNING"])
def test_active_statuses_not_terminal(status: str):
    from src.api.models.simulation import is_terminal_status

    assert is_terminal_status(status) is False


# ---------------------------------------------------------------------------
# JobKind enum — lock the spelling against the roadmap list.
# ---------------------------------------------------------------------------


def test_job_kind_values_match_roadmap():
    from src.api.models.simulation import JobKind

    expected = {
        "mock_static", "dft_relax", "dft_static", "dft_bands", "dft_dos",
        "md_nvt", "md_npt", "md_nve",
        "continuum_elastic", "continuum_thermal", "mesoscale_kmc",
        "ml_train", "ml_infer", "bo_suggest", "al_query",
        "import", "export", "agent_step",
    }
    got = {k.value for k in JobKind}
    missing = expected - got
    assert not missing, f"JobKind missing: {missing}"


# ---------------------------------------------------------------------------
# Sanity check on `transition_to` (model instance)
# ---------------------------------------------------------------------------


class _FakeJob:
    """Minimal stand-in mimicking SimulationJob's transition_to contract.

    Verifies the side-effect behavior (timestamps, error_message) without
    needing SQLAlchemy's InstrumentedAttribute machinery. Binds the real
    transition_to function for method-call semantics.
    """

    def __init__(self, status):
        self.status = status
        self.started_at = None
        self.finished_at = None
        self.updated_at = None
        self.error_message = None

    # Bind the real implementation so the side effects we're testing are
    # exactly the ones that run in production.
    from src.api.models.simulation import SimulationJob  # type: ignore

    transition_to = SimulationJob.transition_to  # type: ignore[assignment]


def test_transition_to_sets_started_at():
    from src.api.models.simulation import JobStatus

    job = _FakeJob(JobStatus.QUEUED)
    job.transition_to(JobStatus.RUNNING, set_started=True)
    assert job.status == JobStatus.RUNNING
    assert job.started_at is not None
    assert job.finished_at is None


def test_transition_to_sets_finished_at_and_error_message():
    from src.api.models.simulation import JobStatus

    job = _FakeJob(JobStatus.RUNNING)
    job.transition_to(
        JobStatus.FAILED,
        set_finished=True,
        error_message="engine exit code 1",
    )
    assert job.status == JobStatus.FAILED
    assert job.finished_at is not None
    assert job.error_message == "engine exit code 1"


def test_transition_to_raises_on_illegal():
    from src.api.models.simulation import IllegalJobTransitionError, JobStatus

    job = _FakeJob(JobStatus.COMPLETED)
    with pytest.raises(IllegalJobTransitionError):
        job.transition_to(JobStatus.RUNNING)
    assert job.status == JobStatus.COMPLETED  # unchanged
