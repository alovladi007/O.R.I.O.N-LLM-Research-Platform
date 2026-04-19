"""Fan-out expansion for steps that declare ``foreach``.

Roadmap example::

    steps:
    - id: sweep
      kind: mock_static
      foreach:
        temperature: [100, 200, 300]

expands to three concrete steps ``sweep__t0``, ``sweep__t1``,
``sweep__t2`` (or more descriptive names keyed on values), each with
``inputs.temperature`` bound to one of the values.

Multi-variable foreach is a Cartesian product:
``{T: [100, 200], P: [1, 2]}`` ⇒ 4 children.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Dict, List

from .spec import StepSpec, WorkflowSpec


class ForeachError(ValueError):
    """Raised when foreach expansion would collide with existing step ids."""


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify_value(value: Any) -> str:
    """``100`` → ``'100'``; ``'low temperature'`` → ``'low-temperature'``."""
    s = str(value)
    s = _SLUG_RE.sub("-", s)
    return s or "x"


def _expand_one(step: StepSpec) -> List[StepSpec]:
    """Replace a foreach step with its Cartesian product of children."""
    if not step.foreach:
        return [step]
    keys = list(step.foreach.keys())
    value_lists = [step.foreach[k] for k in keys]
    children: List[StepSpec] = []
    for combo in itertools.product(*value_lists):
        bindings = dict(zip(keys, combo))
        # New id: step.id + "__" + slug(values joined with "-")
        slug = "-".join(_slugify_value(combo[i]) for i in range(len(combo)))
        child_id = f"{step.id}__{slug}"
        merged_inputs = {**step.inputs, **bindings}
        children.append(
            StepSpec(
                id=child_id,
                kind=step.kind,
                structure_id=step.structure_id,
                inputs=merged_inputs,
                depends_on=list(step.depends_on),
                foreach=None,
            )
        )
    return children


def expand_foreach(spec: WorkflowSpec) -> WorkflowSpec:
    """Return a new :class:`WorkflowSpec` with all foreach steps expanded.

    Expanded child ids are appended to each downstream step's
    ``depends_on`` so a step that depended on the un-expanded parent
    now depends on every expanded child (i.e. a synchronization
    barrier). This is the correct semantics for a fan-out followed by
    aggregation.
    """
    expansions: Dict[str, List[StepSpec]] = {}
    for step in spec.steps:
        children = _expand_one(step)
        expansions[step.id] = children

    # Check for id collisions — expansion could accidentally clash with
    # a user-written step id.
    new_ids: List[str] = []
    for children in expansions.values():
        new_ids.extend(c.id for c in children)
    if len(new_ids) != len(set(new_ids)):
        dupes = [i for i in new_ids if new_ids.count(i) > 1]
        raise ForeachError(f"foreach expansion produced duplicate ids: {sorted(set(dupes))}")

    # Build the expanded step list + rewrite depends_on of downstream steps.
    new_steps: List[StepSpec] = []
    for step in spec.steps:
        children = expansions[step.id]
        for child in children:
            # Rewrite this child's depends_on: if it referenced another
            # step that expanded, replace the reference with *all* of
            # that step's children.
            rewritten_deps: List[str] = []
            for dep in child.depends_on:
                if dep in expansions and expansions[dep] != [step]:
                    rewritten_deps.extend(c.id for c in expansions[dep])
                else:
                    rewritten_deps.append(dep)
            new_steps.append(
                StepSpec(
                    id=child.id,
                    kind=child.kind,
                    structure_id=child.structure_id,
                    inputs=child.inputs,
                    depends_on=rewritten_deps,
                    foreach=None,
                )
            )

    return WorkflowSpec(
        name=spec.name,
        description=spec.description,
        steps=new_steps,
        default_structure_id=spec.default_structure_id,
    )
