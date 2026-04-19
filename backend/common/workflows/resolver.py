"""Resolve ``{"uses": "step_id.outputs.path"}`` references.

When a step becomes dispatchable, its ``inputs`` dict is walked and
every ``{"uses": ...}`` node is replaced with the resolved value from
the outputs of the referenced step.

Reference grammar
-----------------

    step_id '.' 'outputs' '.' dotted.path

Examples:
    step_a.outputs.energy_ev
    step_b.outputs.forces.0.fx    # list index OK
    step_c.outputs.stress.xx      # nested dict

If the step hasn't produced outputs yet (or produces ``None`` at the
referenced path), raises :class:`ResolutionError`. The executor treats
a resolution failure as "the step isn't ready" rather than a hard fail.
"""

from __future__ import annotations

import re
from typing import Any, Dict


class ResolutionError(Exception):
    """Reference could not be resolved against the current run state."""


_REF_RE = re.compile(r"^(?P<step>[A-Za-z0-9_-]+)\.outputs(?:\.(?P<path>.+))?$")


def _lookup(container: Any, path_parts: list[str]) -> Any:
    """Follow dotted path through dicts / lists."""
    cur = container
    for part in path_parts:
        if isinstance(cur, dict):
            if part not in cur:
                raise ResolutionError(f"key {part!r} not found; available: {list(cur)[:10]}")
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError as exc:
                raise ResolutionError(
                    f"list index must be integer, got {part!r}"
                ) from exc
            if idx < 0 or idx >= len(cur):
                raise ResolutionError(f"list index {idx} out of range (len={len(cur)})")
            cur = cur[idx]
        else:
            raise ResolutionError(
                f"cannot descend into {type(cur).__name__} at segment {part!r}"
            )
    return cur


def _resolve_one(ref: str, outputs_by_step: Dict[str, Dict[str, Any]]) -> Any:
    """Resolve one 'step.outputs.path' reference."""
    m = _REF_RE.match(ref)
    if not m:
        raise ResolutionError(f"bad reference shape: {ref!r}")
    step = m.group("step")
    path = m.group("path")
    if step not in outputs_by_step:
        raise ResolutionError(
            f"no outputs for step {step!r} (available: {sorted(outputs_by_step)[:10]})"
        )
    step_outputs = outputs_by_step[step]
    if step_outputs is None:
        raise ResolutionError(f"step {step!r} has no outputs yet")
    parts = [] if not path else path.split(".")
    return _lookup(step_outputs, parts)


def resolve_references(
    value: Any,
    outputs_by_step: Dict[str, Dict[str, Any]],
) -> Any:
    """Recursively replace every ``{"uses": ref}`` node with the resolved value."""
    if isinstance(value, dict):
        if set(value.keys()) == {"uses"} and isinstance(value["uses"], str):
            return _resolve_one(value["uses"], outputs_by_step)
        return {k: resolve_references(v, outputs_by_step) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_references(v, outputs_by_step) for v in value]
    return value
