"""Run ``pw.x`` on a rendered input and parse the result.

Workflow
--------

1. Caller provides a :class:`RenderedInput` (from Session 3.1) and a
   run directory.
2. :func:`run_pw` writes the input file and stages the needed UPFs
   (or symlinks them — configurable).
3. Calls the execution backend (Session 2.3) to run
   ``pw.x -in <input>`` with the caller's resource budget.
4. Parses ``stdout.txt`` (the file the backend writes) into a
   :class:`PWOutput` via :func:`parse_pw_output`.
5. Attaches per-atom species labels onto the force entries (the
   parser can't know species from the output alone).

This is a synchronous function — the Celery task in
``src.worker.tasks`` calls it under :class:`JobLifecycle`.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.common.engines.qe_input import RenderedInput

from .output import PWOutput, parse_pw_output

logger = logging.getLogger(__name__)

DEFAULT_QE_EXECUTABLE = "pw.x"
DEFAULT_RUN_TIMEOUT_MIN = 60


class PWRunError(RuntimeError):
    """pw.x terminated with a non-zero exit and we couldn't parse useful output."""


@dataclass
class PWRunResult:
    """What :func:`run_pw` returns."""

    run_dir: Path
    input_path: Path
    stdout_path: Path
    stderr_path: Path
    output: PWOutput
    returncode: int
    success: bool
    stage: str = "ok"             # "ok" | "parse_failed" | "nonzero_exit" | ...
    error_message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def run_pw(
    rendered: RenderedInput,
    run_dir: Path | str,
    *,
    qe_executable: str = DEFAULT_QE_EXECUTABLE,
    pseudo_src_dir: Optional[Path | str] = None,
    stage_pseudos: bool = True,
    execution_kind: str = "local",
    cpus: int = 1,
    walltime_minutes: int = DEFAULT_RUN_TIMEOUT_MIN,
    species_hint: Optional[List[str]] = None,
) -> PWRunResult:
    """Execute ``pw.x`` on *rendered* and return a parsed result.

    Parameters
    ----------
    rendered
        Output of :func:`backend.common.engines.qe_input.generate_pw_input`.
    run_dir
        Directory the job runs in. Created if missing.
    qe_executable
        Name of the binary. Usually ``pw.x`` (on PATH) or an absolute
        path (conda/Homebrew/custom build).
    pseudo_src_dir
        Where the UPFs live on disk. Defaults to the ``pseudo_dir``
        string embedded in the rendered input's text. Only matters
        when ``stage_pseudos`` is true.
    stage_pseudos
        When ``True``, copy the UPFs referenced by the input into
        ``run_dir/pseudos/`` and rewrite the input's ``pseudo_dir``
        to that local path. Robust against the execution backend
        running in a container with no access to the original path.
        When ``False``, the rendered input's original ``pseudo_dir``
        must be readable from the worker.
    execution_kind
        Passed through to :func:`backend.common.execution.get_execution_backend`.
    cpus, walltime_minutes
        Resource budget.
    species_hint
        Per-atom species list, same order as the structure. Used to
        label the parsed force vectors. If ``None``, forces come back
        with species ``"?"``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage pseudos if requested -----------------------------------------
    input_text = rendered.text
    if stage_pseudos:
        pseudos_local = run_dir / "pseudos"
        pseudos_local.mkdir(exist_ok=True)
        src = Path(pseudo_src_dir).expanduser() if pseudo_src_dir else _extract_pseudo_dir(
            input_text
        )
        if src is None or not src.is_dir():
            raise PWRunError(
                f"pseudo source dir not found: {pseudo_src_dir!r} "
                f"(rendered input was {_extract_pseudo_dir(input_text)!r})"
            )
        for upf in rendered.pseudo_files:
            src_path = src / upf
            dst_path = pseudos_local / upf
            if not src_path.exists():
                raise PWRunError(f"UPF {upf!r} not found in {src}")
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
        input_text = _rewrite_pseudo_dir(input_text, str(pseudos_local.resolve()))

    # --- Write input file ---------------------------------------------------
    input_path = run_dir / rendered.input_filename
    input_path.write_text(input_text)

    # --- Invoke the execution backend --------------------------------------
    from backend.common.execution import (
        JobState,
        Resources,
        get_execution_backend,
        sync_execute,
    )

    backend = get_execution_backend(execution_kind)
    resources = Resources(cpus=cpus, walltime_minutes=walltime_minutes)
    state = sync_execute(
        backend,
        [qe_executable, "-in", str(input_path)],
        run_dir,
        resources,
        poll_interval_seconds=0.5,
    )

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"

    # --- Parse output regardless of exit code; pw.x can write useful partial
    # output even when it crashes, and the parser is defensive.
    try:
        output = parse_pw_output(stdout_path) if stdout_path.exists() else None
    except Exception as exc:  # noqa: BLE001
        output = None
        parse_err = f"parse_failed: {type(exc).__name__}: {exc}"
    else:
        parse_err = None

    if output is not None and species_hint is not None:
        _label_forces(output, species_hint)

    returncode = _rc_from_state(state)
    success = state == JobState.COMPLETED and (
        output is not None and output.convergence.value == "converged"
    )

    if output is None:
        stage = "parse_failed"
        error_message = parse_err or "pw.x produced no parseable stdout"
    elif state != JobState.COMPLETED:
        stage = "nonzero_exit"
        error_message = (
            f"pw.x exited non-zero ({state.value}); "
            f"convergence={output.convergence.value}"
        )
    elif output.convergence.value != "converged":
        stage = "unconverged"
        error_message = (
            f"pw.x exit=0 but convergence={output.convergence.value}"
        )
    else:
        stage = "ok"
        error_message = None

    return PWRunResult(
        run_dir=run_dir,
        input_path=input_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output=output,     # may be None; callers check .success before .output.*
        returncode=returncode,
        success=success,
        stage=stage,
        error_message=error_message,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_pseudo_dir(input_text: str) -> Optional[Path]:
    import re

    m = re.search(r"pseudo_dir\s*=\s*'([^']+)'", input_text)
    return Path(m.group(1)).expanduser() if m else None


def _rewrite_pseudo_dir(input_text: str, new_dir: str) -> str:
    import re

    return re.sub(
        r"(pseudo_dir\s*=\s*)'[^']*'",
        lambda m: f"{m.group(1)}'{new_dir}'",
        input_text,
        count=1,
    )


def _rc_from_state(state) -> int:
    from backend.common.execution import JobState

    if state == JobState.COMPLETED:
        return 0
    if state == JobState.CANCELLED:
        return -15  # SIGTERM convention
    return 1


def _label_forces(output: PWOutput, species: List[str]) -> None:
    """Fill in ``ParsedForce.species`` using the caller's per-atom list."""
    for force in output.forces:
        if 0 <= force.atom_index < len(species):
            force.species = species[force.atom_index]
