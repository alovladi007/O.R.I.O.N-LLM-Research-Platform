"""Tests for Phase 3 / Session 3.1 — QE input generation.

Scope
-----

1. ``PseudopotentialRegistry`` — element detection across filename
   conventions (pslibrary, GBRV/USPP, ONCV, atompaw), cutoff parsing
   from UPF headers, unknown-element + empty-dir errors.
2. ``kgrid_from_structure`` — 1/Å spacing heuristic against cubic +
   tetragonal lattices.
3. ``generate_pw_input`` — renders a valid pw.x input for Si, NaCl,
   and a multi-species system; verifies ATOMIC_SPECIES ordering,
   CELL_PARAMETERS / ATOMIC_POSITIONS blocks, override wiring, and
   ``extra_control`` escape hatch.

The registry tests use a synthetic on-disk pseudos dir (built in
``tmp_path``) so they don't depend on the user having SSSP installed.
A separate test uses the real SSSP dir at ``~/orion/pseudos/...``
when present, marked ``sssp_installed``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Tiny fake UPF factory — matches just enough of the real format for cutoff
# parsing.
# ---------------------------------------------------------------------------


_FAKE_UPF_V2 = """<UPF version="2.0.1">
  <PP_INFO>
    Element: {element}
    Pseudopotential type: USPP
    Functional: PBE

    Suggested minimum cutoff for wavefunctions: {wfc}. Ry
    Suggested minimum cutoff for charge density: {rho}. Ry
  </PP_INFO>
  <PP_HEADER
    element="{element}"
    wfc_cutoff="{wfc_sci}"
    rho_cutoff="{rho_sci}"
  />
</UPF>
"""


def _make_upf(dir_: Path, filename: str, element: str, wfc: float, rho: float) -> Path:
    p = dir_ / filename
    p.write_text(
        _FAKE_UPF_V2.format(
            element=element,
            wfc=wfc,
            rho=rho,
            wfc_sci=f"{wfc:.5e}",
            rho_sci=f"{rho:.5e}",
        )
    )
    return p


@pytest.fixture
def synthetic_pseudos(tmp_path):
    """A mini-library mimicking SSSP's filename diversity."""
    _make_upf(tmp_path, "Si.pbe-n-rrkjus_psl.1.0.0.UPF", "Si", 44, 175)
    _make_upf(tmp_path, "na_pbe_v1.5.uspp.F.UPF", "Na", 40, 320)
    _make_upf(tmp_path, "cl_pbe_v1.4.uspp.F.UPF", "Cl", 45, 225)
    _make_upf(tmp_path, "O.pbe-n-kjpaw_psl.0.1.UPF", "O", 47, 330)
    _make_upf(tmp_path, "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF", "Fe", 90, 1080)
    # A file with no matching element token → should be skipped.
    (tmp_path / "README.txt").write_text("ignore me")
    return tmp_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_detects_elements_across_filename_styles(self, synthetic_pseudos):
        from backend.common.engines.qe_input import PseudopotentialRegistry

        reg = PseudopotentialRegistry(synthetic_pseudos)
        # pslibrary + USPP lowercase + kjpaw + spn
        assert {"Si", "Na", "Cl", "O", "Fe"} <= set(reg.elements())

    def test_filename_mapping(self, synthetic_pseudos):
        from backend.common.engines.qe_input import PseudopotentialRegistry

        reg = PseudopotentialRegistry(synthetic_pseudos)
        assert reg.get("Si").filename == "Si.pbe-n-rrkjus_psl.1.0.0.UPF"
        assert reg.get("Na").filename == "na_pbe_v1.5.uspp.F.UPF"
        assert reg.get("Cl").filename == "cl_pbe_v1.4.uspp.F.UPF"

    def test_cutoffs_parsed_from_header(self, synthetic_pseudos):
        from backend.common.engines.qe_input import PseudopotentialRegistry

        reg = PseudopotentialRegistry(synthetic_pseudos)
        assert reg.get("Si").ecutwfc_ry == pytest.approx(44.0, abs=0.1)
        assert reg.get("Si").ecutrho_ry == pytest.approx(175.0, abs=0.1)
        assert reg.get("Fe").ecutwfc_ry == pytest.approx(90.0, abs=0.1)

    def test_recommended_cutoffs_take_max(self, synthetic_pseudos):
        from backend.common.engines.qe_input import PseudopotentialRegistry

        reg = PseudopotentialRegistry(synthetic_pseudos)
        # Fe dominates both.
        assert reg.recommended_ecutwfc(["Si", "O", "Fe"]) == pytest.approx(90.0, abs=0.1)
        assert reg.recommended_ecutrho(["Si", "O", "Fe"]) == pytest.approx(1080.0, abs=1)

    def test_unknown_element_raises(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            UnknownElementError,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        with pytest.raises(UnknownElementError):
            reg.get("Xe")

    def test_empty_dir_raises(self, tmp_path):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            UPFFileNotFoundError,
        )

        (tmp_path / "README.txt").write_text("just docs")
        with pytest.raises(UPFFileNotFoundError):
            PseudopotentialRegistry(tmp_path)

    def test_missing_dir_raises(self, tmp_path):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            UPFFileNotFoundError,
        )

        with pytest.raises(UPFFileNotFoundError):
            PseudopotentialRegistry(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# K-point grid
# ---------------------------------------------------------------------------


class TestKGridFromStructure:
    def test_cubic_5ang(self):
        from backend.common.engines.qe_input import kgrid_from_structure

        # Cubic box, a=5 Å → |b| = 2π/5 ≈ 1.257; kspacing=0.25 → ceil(5.03)=6
        lat = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        g = kgrid_from_structure(lat, kspacing=0.25)
        assert g.nk1 == g.nk2 == g.nk3 == 6

    def test_anisotropic_grid(self):
        from backend.common.engines.qe_input import kgrid_from_structure

        # a=5, b=10, c=20 → denser grid along a
        lat = [[5.0, 0, 0], [0, 10.0, 0], [0, 0, 20.0]]
        g = kgrid_from_structure(lat, kspacing=0.3)
        assert g.nk1 > g.nk2 > g.nk3

    def test_tighter_spacing_gives_denser_grid(self):
        from backend.common.engines.qe_input import kgrid_from_structure

        lat = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        coarse = kgrid_from_structure(lat, kspacing=0.5)
        fine = kgrid_from_structure(lat, kspacing=0.1)
        assert fine.nk1 > coarse.nk1

    def test_singular_lattice_raises(self):
        from backend.common.engines.qe_input import kgrid_from_structure

        lat = [[0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
        with pytest.raises(ValueError, match="singular"):
            kgrid_from_structure(lat)


# ---------------------------------------------------------------------------
# Renderer — end-to-end
# ---------------------------------------------------------------------------


_SI_STRUCT = {
    "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
    "species": ["Si", "Si"],
    "frac_coords": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
}

_NACL_STRUCT = {
    "lattice": [[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
    "species": ["Na", "Cl"],
    "frac_coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
}


class TestRenderer:
    def test_si_scf_basic_blocks(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(_SI_STRUCT, QEInputParams(prefix="si"), reg)
        t = r.text
        # Required sections
        assert "&CONTROL" in t and t.count("/") >= 3  # &CONTROL, &SYSTEM, &ELECTRONS closers
        assert "calculation = 'scf'" in t
        assert "prefix = 'si'" in t
        assert "ibrav = 0" in t
        assert "nat = 2" in t
        assert "ntyp = 1" in t
        assert "ATOMIC_SPECIES" in t
        assert "Si.pbe-n-rrkjus_psl.1.0.0.UPF" in t
        assert "CELL_PARAMETERS angstrom" in t
        assert "ATOMIC_POSITIONS crystal" in t
        assert "K_POINTS automatic" in t
        # Order: CELL_PARAMETERS must come before ATOMIC_POSITIONS (ibrav=0 rule).
        assert t.index("CELL_PARAMETERS") < t.index("ATOMIC_POSITIONS")

    def test_si_cutoffs_inferred_from_registry(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(_SI_STRUCT, QEInputParams(), reg)
        # Si cutoff is 44 Ry in the synthetic fixture.
        assert r.ecutwfc_ry == pytest.approx(44.0, abs=0.1)
        # ecutrho = max(175, 8*44) = 352
        assert r.ecutrho_ry == pytest.approx(352.0, abs=0.5)

    def test_user_override_wins(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        params = QEInputParams(ecutwfc=60.0, ecutrho=600.0)
        r = generate_pw_input(_SI_STRUCT, params, reg)
        assert r.ecutwfc_ry == 60.0
        assert r.ecutrho_ry == 600.0
        assert "ecutwfc = 60.0" in r.text

    def test_multi_species_nacl(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(_NACL_STRUCT, QEInputParams(prefix="nacl"), reg)
        # ntyp = 2, nat = 2
        assert "ntyp = 2" in r.text
        assert "nat = 2" in r.text
        # Both UPFs referenced
        assert "na_pbe_v1.5.uspp.F.UPF" in r.text
        assert "cl_pbe_v1.4.uspp.F.UPF" in r.text
        # pseudo_files list matches what's referenced
        assert set(r.pseudo_files) == {
            "na_pbe_v1.5.uspp.F.UPF", "cl_pbe_v1.4.uspp.F.UPF",
        }

    def test_kpoints_override(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(
            _SI_STRUCT,
            QEInputParams(kpoints=(8, 8, 8), kpoints_shift=(1, 1, 1)),
            reg,
        )
        assert r.kpoints.nk1 == 8
        assert "8 8 8  1 1 1" in r.text

    def test_relax_writes_ions_namelist(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(_SI_STRUCT, QEInputParams(calculation="relax"), reg)
        assert "&IONS" in r.text
        assert "ion_dynamics = 'bfgs'" in r.text
        assert "forc_conv_thr" in r.text
        assert "&CELL" not in r.text  # only vc-relax gets CELL

    def test_vc_relax_writes_cell_namelist(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(_SI_STRUCT, QEInputParams(calculation="vc-relax"), reg)
        assert "&CELL" in r.text
        assert "cell_dynamics = 'bfgs'" in r.text
        assert "press_conv_thr" in r.text

    def test_spin_polarized(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(
            _SI_STRUCT,
            QEInputParams(spin_polarized=True, tot_magnetization=2.0),
            reg,
        )
        assert "nspin = 2" in r.text
        assert "tot_magnetization = 2.0" in r.text

    def test_extra_control_escape_hatch(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(
            _SI_STRUCT,
            QEInputParams(extra_control={"wf_collect": True, "restart_mode": "restart"}),
            reg,
        )
        assert "wf_collect = .true." in r.text
        assert "restart_mode = 'restart'" in r.text

    def test_unknown_element_raises(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            UnknownElementError,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        bad = {
            "lattice": [[5, 0, 0], [0, 5, 0], [0, 0, 5]],
            "species": ["Xe"],  # not in synthetic library
            "frac_coords": [[0, 0, 0]],
        }
        with pytest.raises(UnknownElementError):
            generate_pw_input(bad, QEInputParams(), reg)

    def test_bad_structure_shape_raises(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        with pytest.raises(ValueError):
            generate_pw_input({"atoms": []}, QEInputParams(), reg)

    def test_pseudo_dir_override(self, synthetic_pseudos):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(synthetic_pseudos)
        r = generate_pw_input(
            _SI_STRUCT, QEInputParams(), reg, pseudo_dir="/cluster/shared/qe",
        )
        assert "pseudo_dir = '/cluster/shared/qe'" in r.text


# ---------------------------------------------------------------------------
# Real SSSP — only runs when the install exists
# ---------------------------------------------------------------------------


SSSP_DIR = Path.home() / "orion" / "pseudos" / "SSSP_1.3.0_PBE_efficiency"


@pytest.mark.skipif(
    not SSSP_DIR.is_dir(), reason="real SSSP library not installed",
)
class TestRealSSSP:
    def test_registry_loads_sssp(self):
        from backend.common.engines.qe_input import PseudopotentialRegistry

        reg = PseudopotentialRegistry(SSSP_DIR)
        assert len(reg.elements()) >= 80
        assert "Si" in reg
        assert "Na" in reg
        assert "Cl" in reg

    def test_si_input_against_real_sssp(self):
        from backend.common.engines.qe_input import (
            PseudopotentialRegistry,
            QEInputParams,
            generate_pw_input,
        )

        reg = PseudopotentialRegistry(SSSP_DIR)
        r = generate_pw_input(_SI_STRUCT, QEInputParams(prefix="si"), reg)
        assert "Si.pbe-n-rrkjus_psl" in r.text
        # SSSP Si cutoff is 44 Ry (not the SSSP JSON value of 30 — the UPF
        # header carries the pseudo's intrinsic minimum).
        assert 30 <= r.ecutwfc_ry <= 100
