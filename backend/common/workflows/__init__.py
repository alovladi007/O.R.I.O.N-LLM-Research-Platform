"""Session 2.4 workflow DAG runtime.

Spec-side pieces (parse, topological sort, foreach expansion,
`{"uses": ...}` reference resolution) live here. The executor that
drives Celery dispatch lives in ``src/worker/tasks.py`` because it
needs a DB session and the SimulationJob state machine.

What this package is not
------------------------

- Not a general-purpose workflow engine (Airflow/Prefect/etc.). Steps
  map 1:1 to ORION ``SimulationJob`` rows via the ``kind`` dispatcher
  added in Session 2.2. There's no support for branching, retries at
  the workflow level, or sub-workflows — all deferred until a concrete
  need shows up.
- Not tied to a specific serialization format. A :class:`WorkflowSpec`
  can come from inline JSON on an API request, a YAML file on disk, or
  a campaign generator — they all build the same pydantic model.
"""

from .executor import Dispatcher, TickResult, tick
from .foreach import ForeachError, expand_foreach
from .manifest import build_workflow_manifest
from .resolver import ResolutionError, resolve_references
from .spec import (
    StepSpec,
    StepStatus,
    WorkflowCycleError,
    WorkflowSpec,
    WorkflowSpecError,
    toposort_steps,
)

__all__ = [
    "Dispatcher",
    "ForeachError",
    "ResolutionError",
    "StepSpec",
    "StepStatus",
    "TickResult",
    "WorkflowCycleError",
    "WorkflowSpec",
    "WorkflowSpecError",
    "build_workflow_manifest",
    "expand_foreach",
    "resolve_references",
    "tick",
    "toposort_steps",
]
