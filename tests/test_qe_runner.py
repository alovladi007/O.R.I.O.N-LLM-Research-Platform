"""Tests for Session 3.2 — ``run_pw`` orchestration.

We don't invoke real ``pw.x`` here (that's the live smoke test,
skipped when the binary isn't installed). Instead we:

- Use a fake ``pw.x`` shell stub that emits a golden fixture.
- Assert the runner stages UPFs, rewrites ``pseudo_dir``, invokes
  the backend, and parses the output correctly.

When the caller has a real ``pw.x`` in PATH (env var
``ORION_PWX_PATH`` set), a ``requires_pw`` test runs a true
end-to-end Si SCF — that's the acceptance test from the roadmap.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


FIXTURES = Path(__file__).parent / "fixtures" / "pw_outputs"


# ---------------------------------------------------------------------------
# Fake pw.x — a shell script that copies a fixture to stdout and exits 0.
# ---------------------------------------------------------------------------


def _make_fake_pw(tmp_path: Path, fixture_name: str, exit_code: int = 0) -> Path:
    """Create an executable ``pw.x`` stub in *tmp_path* that emits *fixture*.

    Shell-quotes the fixture path — the repo path contains spaces and
    an ampersand, both of which break an unquoted ``cat``.
    """
    import shlex

    fixture_path = FIXTURES / fixture_name
    assert fixture_path.exists(), fixture_path

    script = tmp_path / "pw.x"
    script.write_text(
        f"""#!/bin/bash
cat {shlex.quote(str(fixture_path))}
exit {exit_code}
"""
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


# ---------------------------------------------------------------------------
# Helpers to build a minimal RenderedInput without touching real SSSP
# ---------------------------------------------------------------------------


def _make_fake_pseudos(dir_: Path) -> None:
    for name in ("Si.pbe-n-rrkjus_psl.1.0.0.UPF",):
        (dir_ / name).write_text("<UPF fake />\n")


def _build_rendered_input(tmp_path: Path):
    from backend.common.engines.qe_input import RenderedInput
    from backend.common.engines.qe_input.renderer import KPointGrid

    pseudos = tmp_path / "pseudos"
    pseudos.mkdir(parents=True, exist_ok=True)
    _make_fake_pseudos(pseudos)

    text = f"""&CONTROL
  calculation = 'scf'
  prefix = 'si'
  outdir = './tmp'
  pseudo_dir = '{pseudos}'
/
&SYSTEM
  ibrav = 0
  nat = 2
  ntyp = 1
  ecutwfc = 44.0
  ecutrho = 352.0
/
&ELECTRONS
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Si  28.086  Si.pbe-n-rrkjus_psl.1.0.0.UPF
CELL_PARAMETERS angstrom
  5.43 0 0
  0 5.43 0
  0 0 5.43
ATOMIC_POSITIONS crystal
  Si  0 0 0
  Si  0.25 0.25 0.25
K_POINTS automatic
  4 4 4  0 0 0
"""
    return (
        RenderedInput(
            text=text,
            input_filename="si.in",
            pseudo_files=["Si.pbe-n-rrkjus_psl.1.0.0.UPF"],
            ecutwfc_ry=44.0,
            ecutrho_ry=352.0,
            kpoints=KPointGrid(4, 4, 4),
        ),
        pseudos,
    )


# ---------------------------------------------------------------------------
# Staging + rewrite
# ---------------------------------------------------------------------------


class TestStagePseudos:
    def test_upfs_copied_into_run_dir(self, tmp_path: Path):
        from backend.common.engines.qe_run import run_pw

        rendered, pseudos_src = _build_rendered_input(tmp_path / "in")
        fake_pw = _make_fake_pw(tmp_path, "si_scf_converged.out")

        run_dir = tmp_path / "run"
        result = run_pw(
            rendered,
            run_dir,
            qe_executable=str(fake_pw),
            pseudo_src_dir=pseudos_src,
            cpus=1,
            walltime_minutes=5,
        )
        staged = run_dir / "pseudos" / "Si.pbe-n-rrkjus_psl.1.0.0.UPF"
        assert staged.exists()
        assert result.success is True

    def test_input_has_rewritten_pseudo_dir(self, tmp_path: Path):
        from backend.common.engines.qe_run import run_pw

        rendered, pseudos_src = _build_rendered_input(tmp_path / "in")
        fake_pw = _make_fake_pw(tmp_path, "si_scf_converged.out")

        run_dir = tmp_path / "run"
        run_pw(
            rendered,
            run_dir,
            qe_executable=str(fake_pw),
            pseudo_src_dir=pseudos_src,
        )
        text = (run_dir / "si.in").read_text()
        assert f"pseudo_dir = '{(run_dir / 'pseudos').resolve()}'" in text

    def test_missing_upf_raises_pw_run_error(self, tmp_path: Path):
        from backend.common.engines.qe_run import PWRunError, run_pw

        rendered, _ = _build_rendered_input(tmp_path / "in")
        fake_pw = _make_fake_pw(tmp_path, "si_scf_converged.out")

        empty_pseudos = tmp_path / "empty"
        empty_pseudos.mkdir()
        with pytest.raises(PWRunError):
            run_pw(
                rendered,
                tmp_path / "run",
                qe_executable=str(fake_pw),
                pseudo_src_dir=empty_pseudos,
            )


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_converged_run_returns_success(self, tmp_path: Path):
        from backend.common.engines.qe_run import run_pw

        rendered, pseudos_src = _build_rendered_input(tmp_path / "in")
        fake_pw = _make_fake_pw(tmp_path, "si_scf_converged.out")

        result = run_pw(
            rendered,
            tmp_path / "run",
            qe_executable=str(fake_pw),
            pseudo_src_dir=pseudos_src,
        )
        assert result.success is True
        assert result.stage == "ok"
        assert result.returncode == 0
        assert result.output is not None
        assert result.output.convergence.value == "converged"
        # Energy from the golden fixture.
        assert result.output.energy.total_ry == pytest.approx(-15.83, abs=1e-4)

    def test_unconverged_run_is_not_success(self, tmp_path: Path):
        from backend.common.engines.qe_run import run_pw

        rendered, pseudos_src = _build_rendered_input(tmp_path / "in")
        # pw.x exit 0 but SCF NOT achieved → stage=unconverged, success=False.
        fake_pw = _make_fake_pw(tmp_path, "si_scf_unconverged.out", exit_code=0)

        result = run_pw(
            rendered,
            tmp_path / "run",
            qe_executable=str(fake_pw),
            pseudo_src_dir=pseudos_src,
        )
        assert result.success is False
        assert result.stage == "unconverged"
        assert "convergence" in (result.error_message or "")

    def test_nonzero_exit_marks_failed(self, tmp_path: Path):
        from backend.common.engines.qe_run import run_pw

        rendered, pseudos_src = _build_rendered_input(tmp_path / "in")
        fake_pw = _make_fake_pw(tmp_path, "error_wrong_pseudo.out", exit_code=1)

        result = run_pw(
            rendered,
            tmp_path / "run",
            qe_executable=str(fake_pw),
            pseudo_src_dir=pseudos_src,
        )
        assert result.success is False
        # returncode != 0 and output reports ERRORED.
        assert result.stage == "nonzero_exit"
        assert result.output is not None
        assert result.output.convergence.value == "errored"

    def test_species_hint_labels_forces(self, tmp_path: Path):
        from backend.common.engines.qe_run import run_pw

        rendered, pseudos_src = _build_rendered_input(tmp_path / "in")
        fake_pw = _make_fake_pw(tmp_path, "si_scf_converged.out")

        result = run_pw(
            rendered,
            tmp_path / "run",
            qe_executable=str(fake_pw),
            pseudo_src_dir=pseudos_src,
            species_hint=["Si", "Si"],
        )
        assert all(f.species == "Si" for f in result.output.forces)


# ---------------------------------------------------------------------------
# Live smoke test — runs only when a real pw.x is available.
# ---------------------------------------------------------------------------


_REAL_PW = os.environ.get("ORION_PWX_PATH")
_SSSP_DIR = Path.home() / "orion" / "pseudos" / "SSSP_1.3.0_PBE_efficiency"


@pytest.mark.skipif(
    _REAL_PW is None or not Path(_REAL_PW).exists(),
    reason="set ORION_PWX_PATH to a real pw.x to run the live smoke test",
)
@pytest.mark.skipif(
    not _SSSP_DIR.is_dir(),
    reason="real SSSP library not installed",
)
class TestLivePwSmoke:
    """End-to-end: generate Si input, run real pw.x, parse output.

    This is the roadmap Session 3.2 acceptance test. Only runs when the
    machine actually has ``pw.x`` installed.
    """

    def test_si_scf_end_to_end(self, tmp_path: Path):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )
        from backend.common.engines.qe_run import run_pw

        reg = PseudopotentialRegistry(_SSSP_DIR)
        structure = {
            "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
            "species": ["Si", "Si"],
            "frac_coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
        }
        rendered = generate_pw_input(
            structure,
            QEInputParams(prefix="si", calculation="scf", kpoints=(2, 2, 2)),
            reg,
        )
        result = run_pw(
            rendered,
            tmp_path / "run",
            qe_executable=_REAL_PW,
            pseudo_src_dir=_SSSP_DIR,
            cpus=1,
            walltime_minutes=10,
            species_hint=["Si", "Si"],
        )
        assert result.success is True, result.error_message
        # Si total energy should be around -15 Ry (≈ -210 eV) at these cutoffs.
        assert -300 < result.output.energy.total_ev < -100
        assert result.output.convergence.value == "converged"
