"""Pydantic schema for a workflow DAG spec + topological sort.

Workflow shape (roadmap Session 2.4)
------------------------------------

A ``WorkflowSpec`` is a list of steps. Each step declares:

- ``id`` — unique within the workflow.
- ``kind`` — what the step runs; maps to ``JobKind`` values from
  Session 1.4. The runtime dispatches these through the same
  ``_DISPATCH_TASKS`` table the ``POST /api/v1/jobs/dispatch`` endpoint
  uses, so any engine registered there is workflow-callable.
- ``inputs`` — free-form dict. Values may be literals or references:
  ``{"uses": "step_b.outputs.energy_ev"}``. References are resolved
  lazily when the step becomes ready (see ``resolver.py``).
- ``foreach`` — optional fan-out directive. ``{"temperature":
  [100, 200, 300]}`` expands to three children, each with
  ``inputs.temperature`` bound to one value.
- ``depends_on`` — optional explicit deps (sometimes the ``uses``
  references don't cover them, e.g. ordering by side effect).
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkflowSpecError(ValueError):
    """Malformed workflow spec (duplicate IDs, unknown deps, etc.)."""


class WorkflowCycleError(WorkflowSpecError):
    """Topological sort discovered a cycle."""


class StepStatus(str, enum.Enum):
    """Per-step runtime status.

    Kept distinct from :class:`~src.api.models.simulation.JobStatus` so
    workflow-level concerns (e.g. ``BLOCKED_BY_UPSTREAM``) don't bleed
    into the job state machine.
    """

    PENDING = "pending"          # created but predecessors not done
    DISPATCHABLE = "dispatchable"  # deps satisfied, awaiting dispatch
    RUNNING = "running"          # underlying SimulationJob RUNNING
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"          # upstream FAILED

    @property
    def is_terminal(self) -> bool:
        return self in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.CANCELLED,
            StepStatus.SKIPPED,
        )


class StepSpec(BaseModel):
    """One step in a workflow DAG."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64)
    kind: str = Field(..., description="JobKind value (mock_static, dft_relax, ...)")
    structure_id: Optional[str] = Field(
        None, description="Optional override — some steps carry their own structure."
    )
    inputs: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    foreach: Optional[Dict[str, List[Any]]] = Field(
        default=None,
        description=(
            "Fan-out spec: {var_name: [values]}. Expansion happens at "
            "workflow submission; the expanded children inherit "
            "inputs + this step's id as a prefix."
        ),
    )

    @field_validator("id")
    @classmethod
    def _id_is_slug_like(cls, v: str) -> str:
        if not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError("step id must be alphanumeric + [_-]")
        return v

    @field_validator("foreach")
    @classmethod
    def _foreach_nonempty(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError("foreach must have at least one variable")
        for key, values in v.items():
            if not values:
                raise ValueError(f"foreach[{key!r}] has no values")
        return v


class WorkflowSpec(BaseModel):
    """A submittable DAG of steps."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = Field(None, max_length=1024)
    steps: List[StepSpec]
    default_structure_id: Optional[str] = Field(
        None, description="Used by steps that don't set structure_id explicitly."
    )

    @field_validator("steps")
    @classmethod
    def _steps_have_unique_ids(cls, steps: List[StepSpec]) -> List[StepSpec]:
        seen: Set[str] = set()
        for s in steps:
            if s.id in seen:
                raise ValueError(f"duplicate step id: {s.id!r}")
            seen.add(s.id)
        return steps

    def step_ids(self) -> List[str]:
        return [s.id for s in self.steps]

    def validate_dependencies(self) -> None:
        """Raises :class:`WorkflowSpecError` if ``depends_on`` or
        ``uses`` references name a step that doesn't exist."""
        known = set(self.step_ids())
        for s in self.steps:
            for dep in s.depends_on:
                if dep not in known:
                    raise WorkflowSpecError(
                        f"step {s.id!r} depends on unknown step {dep!r}"
                    )
            # Collect `uses` references — values that look like
            # {"uses": "step_id.outputs.X"}.
            for target in _collect_uses_targets(s.inputs):
                ref_step = target.split(".", 1)[0]
                if ref_step not in known:
                    raise WorkflowSpecError(
                        f"step {s.id!r} uses unknown step {ref_step!r} "
                        f"(full ref: {target!r})"
                    )


def _collect_uses_targets(obj: Any) -> List[str]:
    """Walk *obj* and return every ``{"uses": "..."}`` target string.

    Shallow-ish recursion is fine — nested inputs are rare but the
    recursion keeps it easy to support a future ``nested.map`` case.
    """
    out: List[str] = []
    if isinstance(obj, dict):
        if set(obj.keys()) == {"uses"} and isinstance(obj["uses"], str):
            out.append(obj["uses"])
            return out
        for v in obj.values():
            out.extend(_collect_uses_targets(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_uses_targets(v))
    return out


def _implicit_deps_from_uses(spec: WorkflowSpec) -> Dict[str, Set[str]]:
    """For each step id, the set of step ids referenced via ``uses``."""
    deps: Dict[str, Set[str]] = {}
    for s in spec.steps:
        deps[s.id] = set(s.depends_on)
        for target in _collect_uses_targets(s.inputs):
            deps[s.id].add(target.split(".", 1)[0])
    return deps


def toposort_steps(spec: WorkflowSpec) -> List[str]:
    """Return step ids in dispatch order.

    Kahn's algorithm. Raises :class:`WorkflowCycleError` if the graph
    has a cycle and :class:`WorkflowSpecError` if deps reference
    unknown steps.
    """
    spec.validate_dependencies()
    deps = _implicit_deps_from_uses(spec)

    # Remaining incoming-edge count per node.
    in_degree = {sid: len(ds) for sid, ds in deps.items()}
    # Reverse edges: predecessor → successors, so we can decrement
    # successor in-degrees when a node is emitted.
    reverse: Dict[str, List[str]] = {sid: [] for sid in deps}
    for sid, ds in deps.items():
        for d in ds:
            reverse[d].append(sid)

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    ordered: List[str] = []
    while queue:
        # Stable order: sort ready set so topologically-equivalent
        # nodes come out in a predictable order for tests/reproducibility.
        queue.sort()
        node = queue.pop(0)
        ordered.append(node)
        for succ in reverse[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(ordered) != len(deps):
        remaining = [sid for sid in deps if sid not in ordered]
        raise WorkflowCycleError(f"cycle detected involving steps: {remaining}")
    return ordered
