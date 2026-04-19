"""Build the ``workflow.json`` manifest for a completed run.

Shape (roadmap):

    {
      "workflow_run_id": "...",
      "name": "...",
      "steps": {
        "step_id": {
          "job_id": "...",
          "status": "completed",
          "outputs": {...},
          "artifact": {"bucket": "orion-artifacts", "key": "jobs/.../run.tgz"}
        },
        ...
      }
    }
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


def build_workflow_manifest(
    *,
    workflow_run_id: str,
    name: str,
    step_records: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-step records into the canonical manifest dict.

    Each record in *step_records* should carry at least::

        {
          "step_id": ...,
          "status": ...,
          "job_id": ... or None,
          "outputs": {...} or None,
          "artifact": {...} or None,
        }
    """
    steps: Dict[str, Any] = {}
    for rec in step_records:
        sid = rec["step_id"]
        steps[sid] = {
            "job_id": rec.get("job_id"),
            "status": rec.get("status"),
            "outputs": rec.get("outputs"),
            "artifact": rec.get("artifact"),
        }
    return {
        "workflow_run_id": workflow_run_id,
        "name": name,
        "manifest_schema": "workflow_run.v1",
        "steps": steps,
    }
