"""Run ``lmp`` on a rendered input and parse the results.

Mirrors :func:`backend.common.engines.qe_run.run_pw`:

1. Caller supplies a :class:`RenderedLAMMPSInput` and a run dir.
2. :func:`run_lammps` writes ``in.lammps`` + ``structure.data`` +
   potential file via :func:`write_lammps_inputs`.
3. Invokes ``lmp_serial -in in.lammps`` (or caller-chosen binary)
   via the Session 2.3 execution backend.
4. Parses ``log.lammps`` for thermo + errors, and yields a
   :class:`LAMMPSRunResult` with everything the Celery task needs.

Trajectory file is NOT parsed eagerly — it can be large. Callers use
:func:`backend.common.engines.lammps_run.parse_lammps_dump` when they
want the frames.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.common.engines.lammps_input import (
    RenderedLAMMPSInput,
    write_lammps_inputs,
)

from .log import LAMMPSLog, parse_lammps_log

logger = logging.getLogger(__name__)

DEFAULT_LMP_EXECUTABLE = "lmp_serial"
DEFAULT_RUN_TIMEOUT_MIN = 60


class LAMMPSRunError(RuntimeError):
    """``lmp`` exited non-zero and nothing useful was produced."""


@dataclass
class LAMMPSRunResult:
    """What :func:`run_lammps` returns."""

    run_dir: Path
    input_path: Path
    data_path: Path
    stdout_path: Path
    stderr_path: Path
    log_path: Path
    dump_paths: List[Path]
    log: Optional[LAMMPSLog]
    returncode: int
    success: bool
    stage: str = "ok"  # "ok" | "nonzero_exit" | "parse_failed"
    error_message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def run_lammps(
    rendered: RenderedLAMMPSInput,
    run_dir: Path | str,
    *,
    lmp_executable: str = DEFAULT_LMP_EXECUTABLE,
    potential_source_dir: Optional[Path | str] = None,
    execution_kind: str = "local",
    cpus: int = 1,
    walltime_minutes: int = DEFAULT_RUN_TIMEOUT_MIN,
) -> LAMMPSRunResult:
    """Execute LAMMPS on *rendered* and return a parsed result.

    *potential_source_dir* is where the forcefield potential file lives;
    it'll be copied next to the input deck. If ``None``, uses the
    directory of the bundled / registry potential.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Stage input + data + potential files.
    src_path = (
        Path(potential_source_dir).expanduser()
        if potential_source_dir is not None
        else None
    )
    written = write_lammps_inputs(
        rendered, run_dir, potential_source_dir=src_path,
    )
    input_path = written["input"]
    data_path = written["data"]

    # Invoke lmp via the execution backend.
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
        [lmp_executable, "-in", input_path.name],
        run_dir,
        resources,
        poll_interval_seconds=0.5,
    )

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    # LAMMPS writes log.lammps in cwd by default. Some binaries don't;
    # fall back to stdout in that case.
    log_path = run_dir / "log.lammps"

    # Discover dump files produced by the run. Session 4.1's default
    # emits e.g. "traj.dump" or one per dump command.
    dump_paths = sorted(run_dir.glob("*.dump")) + sorted(run_dir.glob("*.lammpstrj"))

    log: Optional[LAMMPSLog] = None
    parse_err: Optional[str] = None
    try:
        if log_path.is_file():
            log = parse_lammps_log(log_path)
        elif stdout_path.is_file():
            log = parse_lammps_log(stdout_path)
    except Exception as exc:  # noqa: BLE001
        parse_err = f"log parse failed: {type(exc).__name__}: {exc}"

    returncode = 0 if state == JobState.COMPLETED else 1
    success = state == JobState.COMPLETED and log is not None and not log.errors

    if log is None:
        stage = "parse_failed"
        error_message = parse_err or "no log.lammps and no stdout.txt"
    elif state != JobState.COMPLETED:
        stage = "nonzero_exit"
        error_message = f"lmp exited {state.value}; {len(log.errors)} errors"
    elif log.errors:
        stage = "nonzero_exit"
        error_message = f"LAMMPS reported {len(log.errors)} errors: {log.errors[0]}"
    else:
        stage = "ok"
        error_message = None

    return LAMMPSRunResult(
        run_dir=run_dir,
        input_path=input_path,
        data_path=data_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        log_path=log_path,
        dump_paths=dump_paths,
        log=log,
        returncode=returncode,
        success=success,
        stage=stage,
        error_message=error_message,
    )
