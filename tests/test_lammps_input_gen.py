"""Tests for Phase 4 / Session 4.1 — LAMMPS input generation.

Scope
-----

1. Forcefield registry — auto-select, explicit lookup, failure when no
   FF covers an element mix, availability flag semantics.
2. :class:`LAMMPSInputParams` — defaults, damping inference, step
   counting from ``duration_ps / timestep_fs``.
3. Renderer — produces a valid ``in.lammps`` for Cu (EAM, 108 atoms,
   NVT 300 K, 100 ps, dt=1 fs) and a valid ``structure.data`` via
   pymatgen's ``LammpsData``.
4. Unit audits — fs → ps conversion for ``metal`` units; fs →
   fs for ``real`` units; reduced-time passthrough for ``lj``.

Acceptance criteria in the roadmap:

- Cu (108 atoms, EAM, NVT 300K, 100 ps, dt=1 fs) — registry auto-select
  picks EAM, renderer emits the right ``pair_style eam/alloy`` line
  and the ``timestep 0.001`` (ps) line, ``run 100000``.
- Cu + H composition fails cleanly with ``NoCompatibleForcefieldError``
  once LJ has been removed from a custom registry (otherwise LJ
  covers everything — that's the documented fallback).
- All timesteps in fs internally; converted to LAMMPS units at
  generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures — synthetic potentials in tmp_path, and simple Cu/Si structures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_potentials(tmp_path):
    """Write fake EAM + Tersoff files into a tmp data dir.

    We don't care about the file contents — the renderer only
    references them by filename. LAMMPS would reject them, but
    the unit tests never actually invoke LAMMPS; that's Session 4.2.
    """
    for name in (
        "Cu_u3.eam",
        "Ni99_v2.eam.alloy",
        "Al99_v2.eam.alloy",
        "Si.tersoff",
        "C.tersoff",
    ):
        (tmp_path / name).write_text(f"# fake {name} for unit tests\n")
    return tmp_path


@pytest.fixture
def patched_default_registry(synthetic_potentials, monkeypatch):
    """Rebuild ``default_registry`` against the synthetic potentials dir.

    We patch ``POTENTIAL_DATA_DIR`` to the tmp path, rebuild the
    registry, and restore both at teardown (monkeypatch handles the
    restore). This yields a registry where Cu/Ni/Al/Si/C are all
    ``available=True`` — what acceptance tests need.
    """
    from backend.common.engines.lammps_input.forcefields import registry as reg_mod

    monkeypatch.setattr(reg_mod, "POTENTIAL_DATA_DIR", synthetic_potentials)
    new_registry = reg_mod.ForcefieldRegistry(reg_mod._build_default_specs())
    monkeypatch.setattr(reg_mod, "default_registry", new_registry)
    # Also patch the re-exports so test imports see the fresh one.
    from backend.common.engines import lammps_input

    monkeypatch.setattr(lammps_input, "default_registry", new_registry)
    monkeypatch.setattr(
        lammps_input.forcefields, "default_registry", new_registry
    )
    return new_registry


def _cubic(a: float, species: list, frac_coords: list) -> dict:
    return {
        "lattice": [[a, 0, 0], [0, a, 0], [0, 0, a]],
        "species": species,
        "frac_coords": frac_coords,
    }


def _cu_108_atoms() -> dict:
    """3x3x3 supercell of FCC Cu primitive (4 atoms/cell) → 108 atoms."""
    a0 = 3.615
    super_a = 3 * a0
    base = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]
    coords = []
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                for b in base:
                    coords.append(
                        [
                            (b[0] + ix) / 3,
                            (b[1] + iy) / 3,
                            (b[2] + iz) / 3,
                        ]
                    )
    species = ["Cu"] * len(coords)
    return {
        "lattice": [[super_a, 0, 0], [0, super_a, 0], [0, 0, super_a]],
        "species": species,
        "frac_coords": coords,
    }


# ---------------------------------------------------------------------------
# Forcefield registry
# ---------------------------------------------------------------------------


class TestForcefieldRegistry:
    def test_default_registry_names_include_all_canonical_ffs(self):
        from backend.common.engines.lammps_input import default_registry

        names = set(default_registry.names())
        for expected in (
            "eam_cu", "eam_ni", "eam_al",
            "tersoff_si", "tersoff_c",
            "reaxff", "mace", "lj",
        ):
            assert expected in names

    def test_auto_select_cu_picks_eam_when_available(self, patched_default_registry):
        ff = patched_default_registry.auto_select({"Cu"})
        assert ff.name == "eam_cu"
        assert ff.kind == "eam"
        assert ff.units == "metal"

    def test_auto_select_si_picks_tersoff_not_lj(self, patched_default_registry):
        ff = patched_default_registry.auto_select({"Si"})
        assert ff.name == "tersoff_si"
        # Tersoff is declared before LJ so it wins even though LJ
        # also covers Si via ``matches_all``.

    def test_auto_select_unregistered_element_falls_back_to_lj(
        self, patched_default_registry,
    ):
        # Ar is not parameterized by any non-LJ spec.
        ff = patched_default_registry.auto_select({"Ar"})
        assert ff.name == "lj"
        assert ff.units == "lj"

    def test_custom_registry_without_lj_raises_for_uncovered_mix(self):
        from backend.common.engines.lammps_input.forcefields import (
            ForcefieldRegistry, NoCompatibleForcefieldError, _build_default_specs,
        )

        # Build a registry without LJ so the fallback is gone.
        specs = [s for s in _build_default_specs() if s.name != "lj"]
        reg = ForcefieldRegistry(specs)
        with pytest.raises(NoCompatibleForcefieldError, match=r"Cu.*H"):
            reg.auto_select({"Cu", "H"})

    def test_unavailable_ff_is_skipped_by_autoselect(self):
        # Fresh default registry (no tmp potentials) — EAM files aren't
        # shipped, so eam_cu is declared but not available. Auto-select
        # for Cu must therefore fall back to LJ rather than raising.
        from backend.common.engines.lammps_input import default_registry

        # Recover a fresh registry independent of monkeypatching
        from backend.common.engines.lammps_input.forcefields import registry as reg_mod

        fresh = reg_mod.ForcefieldRegistry(reg_mod._build_default_specs())
        # In CI without fetched EAM files, eam_cu.available is False.
        ff = fresh.auto_select({"Cu"})
        if not fresh._specs["eam_cu"].available:
            assert ff.name == "lj"
        else:
            assert ff.name == "eam_cu"

    def test_explicit_get_unavailable_raises(self):
        from backend.common.engines.lammps_input import (
            default_registry, NoCompatibleForcefieldError,
        )

        # ReaxFF is always unavailable in the default build until wired
        with pytest.raises(NoCompatibleForcefieldError):
            default_registry.get("reaxff")

    def test_covers_respects_matches_all(self):
        from backend.common.engines.lammps_input import default_registry

        lj = default_registry._specs["lj"]
        assert lj.covers({"Xe"}) is True  # matches_all
        eam_cu = default_registry._specs["eam_cu"]
        assert eam_cu.covers({"Cu"}) is True
        assert eam_cu.covers({"Cu", "Ni"}) is False


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


class TestLAMMPSInputParams:
    def test_defaults_are_reasonable(self):
        from backend.common.engines.lammps_input import LAMMPSInputParams

        p = LAMMPSInputParams()
        assert p.temperature_k == 300.0
        assert p.timestep_fs == 1.0
        assert p.duration_ps == 100.0
        assert p.ensemble == "nvt_nose_hoover"

    def test_damping_inference(self):
        from backend.common.engines.lammps_input import LAMMPSInputParams

        p = LAMMPSInputParams(timestep_fs=2.0)
        assert p.t_damp_fs_resolved == 200.0   # 100 * dt
        assert p.p_damp_fs_resolved == 2000.0  # 1000 * dt

    def test_damping_override(self):
        from backend.common.engines.lammps_input import LAMMPSInputParams

        p = LAMMPSInputParams(t_damp_fs=50.0, p_damp_fs=500.0)
        assert p.t_damp_fs_resolved == 50.0
        assert p.p_damp_fs_resolved == 500.0

    def test_n_steps_from_duration(self):
        from backend.common.engines.lammps_input import LAMMPSInputParams

        # 100 ps @ 1 fs dt → 100000 steps
        assert LAMMPSInputParams(duration_ps=100.0, timestep_fs=1.0).n_steps == 100_000
        # 100 ps @ 0.5 fs dt → 200000 steps
        assert LAMMPSInputParams(duration_ps=100.0, timestep_fs=0.5).n_steps == 200_000
        # 1 ps @ 0.25 fs → 4000
        assert LAMMPSInputParams(duration_ps=1.0, timestep_fs=0.25).n_steps == 4_000

    def test_extra_forbid(self):
        from backend.common.engines.lammps_input import LAMMPSInputParams

        with pytest.raises(Exception):
            LAMMPSInputParams(not_a_field=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Renderer — Cu EAM acceptance case
# ---------------------------------------------------------------------------


class TestRendererCuEAM:
    def test_cu_108_atoms_eam_nvt_300k_100ps_dt1fs(self, patched_default_registry):
        """Full roadmap acceptance scenario."""
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        struct = _cu_108_atoms()
        params = LAMMPSInputParams(
            ensemble="nvt_nose_hoover",
            temperature_k=300.0,
            timestep_fs=1.0,
            duration_ps=100.0,
            dump_every=1000,
            thermo_every=100,
        )
        r = generate_lammps_input(
            struct, params, registry=patched_default_registry,
        )

        # Forcefield choice
        assert r.forcefield is not None and r.forcefield.name == "eam_cu"
        assert r.potential_file == "Cu_u3.eam"

        # Data file
        assert "108  atoms" in r.data_text
        assert "1  atom types" in r.data_text
        assert "Masses" in r.data_text
        # Cu mass ~63.546
        assert "63.546" in r.data_text

        t = r.input_text

        # Units + atom_style
        assert "units           metal" in t
        assert "atom_style      atomic" in t
        assert "boundary        p p p" in t

        # Pair style + coeff
        assert "pair_style eam" in t
        # Cu_u3.eam is the single-element ``pair_style eam`` format;
        # pair_coeff takes no element labels. Session 4.3b registry change.
        assert "pair_coeff * * Cu_u3.eam" in t
        assert "pair_coeff * * Cu_u3.eam Cu" not in t

        # Initial velocities
        assert "velocity        all create 300.0000" in t

        # Timestep conversion — 1 fs → 0.001 ps for metal units
        assert "timestep        0.00100000" in t

        # NVT fix, damping = 100 * dt = 100 fs = 0.1 ps in metal units
        assert "fix             1 all nvt temp 300.0000 300.0000 0.100000" in t

        # Run length = 100000 steps
        assert "run             100000" in t

        # Thermo / dump
        assert "thermo          100" in t
        assert "dump            1 all custom 1000 dump.lammpstrj" in t

    def test_cu_h_mix_fails_cleanly_without_lj(self, patched_default_registry):
        """Roadmap: 'Cu,H' should fail cleanly when LJ isn't a fallback."""
        from backend.common.engines.lammps_input import (
            ForcefieldRegistry, LAMMPSInputParams, NoCompatibleForcefieldError,
            generate_lammps_input,
        )
        from backend.common.engines.lammps_input.forcefields import _build_default_specs

        specs = [s for s in _build_default_specs() if s.name != "lj"]
        reg_no_lj = ForcefieldRegistry(specs)

        struct = {
            "lattice": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
            "species": ["Cu", "H"],
            "frac_coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
        }
        with pytest.raises(NoCompatibleForcefieldError, match=r"Cu.*H|H.*Cu"):
            generate_lammps_input(
                struct, LAMMPSInputParams(), registry=reg_no_lj,
            )

    def test_explicit_ff_that_does_not_cover_elements_raises(
        self, patched_default_registry,
    ):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, NoCompatibleForcefieldError, generate_lammps_input,
        )

        struct = _cubic(3.5, ["Ni"], [[0, 0, 0]])
        with pytest.raises(NoCompatibleForcefieldError, match=r"eam_cu"):
            generate_lammps_input(
                struct,
                LAMMPSInputParams(forcefield_name="eam_cu"),
                registry=patched_default_registry,
            )


# ---------------------------------------------------------------------------
# Unit audits
# ---------------------------------------------------------------------------


class TestUnitAudits:
    def test_metal_units_convert_fs_to_ps(self, patched_default_registry):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        struct = _cu_108_atoms()
        r = generate_lammps_input(
            struct,
            LAMMPSInputParams(timestep_fs=2.5, duration_ps=10.0, t_damp_fs=250.0),
            registry=patched_default_registry,
        )
        # 2.5 fs → 0.0025 ps
        assert r.timestep_in_units == pytest.approx(0.0025, abs=1e-9)
        assert "timestep        0.00250000" in r.input_text
        # 250 fs → 0.25 ps — rendered with 6 decimals
        assert " 0.250000" in r.input_text
        # n_steps = 10*1000/2.5 = 4000
        assert r.n_steps == 4000
        assert "run             4000" in r.input_text

    def test_lj_units_passthrough(self, patched_default_registry):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        # Pick an element no solid-state FF covers → LJ path.
        struct = _cubic(10.0, ["Ar"], [[0, 0, 0]])
        r = generate_lammps_input(
            struct,
            # Session 4.3b: LJ runs use dedicated reduced-unit fields.
            # Leaving them None falls back to the LAMMPS-community
            # default dt* = 0.005, duration* = 500.
            LAMMPSInputParams(
                timestep_lj_reduced=0.005, duration_lj_reduced=1.0,
            ),
            registry=patched_default_registry,
        )
        assert r.forcefield.name == "lj"
        assert r.forcefield.units == "lj"
        assert r.timestep_in_units == pytest.approx(0.005, abs=1e-12)
        assert r.n_steps == 200  # duration 1.0 / dt 0.005
        assert "units           lj" in r.input_text

    def test_lj_defaults_when_reduced_fields_unset(self, patched_default_registry):
        """Session 4.3b regression for the Session 4.1 LJ-units bug.

        Before the fix, a naive ``LAMMPSInputParams()`` on a LJ system
        produced ``timestep 1.0`` in reduced units — catastrophic.
        The fix: LJ defaults to ``dt* = 0.005, duration* = 500``
        whenever the dedicated reduced fields aren't set.
        """
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        struct = _cubic(10.0, ["Ar"], [[0, 0, 0]])
        r = generate_lammps_input(
            struct, LAMMPSInputParams(), registry=patched_default_registry,
        )
        assert r.forcefield.units == "lj"
        assert r.timestep_in_units == pytest.approx(0.005, abs=1e-12)
        # 500 / 0.005 = 100000
        assert r.n_steps == 100_000
        assert "timestep        0.00500000" in r.input_text

    def test_real_units_convert_fs_to_fs(self, patched_default_registry):
        """ReaxFF uses ``real`` units; timestep stays in fs.

        Locks in the behavior of ``_timestep_in_units`` for the "real"
        branch. LJ is not exercised here — see
        ``test_timestep_in_units_lj_raises``.
        """
        from backend.common.engines.lammps_input.renderer import _timestep_in_units

        assert _timestep_in_units(1.0, "real") == 1.0
        assert _timestep_in_units(0.25, "real") == 0.25
        assert _timestep_in_units(1.0, "metal") == pytest.approx(0.001)
        with pytest.raises(ValueError, match="unsupported"):
            _timestep_in_units(1.0, "cgs")

    def test_timestep_in_units_lj_raises(self):
        """LJ runs must not go through ``_timestep_in_units``.

        Session 4.1 silently passed fs→reduced, producing dt* = 1.0
        on naive inputs. The 4.3b fix removes that footgun: calling
        the fn with ``units='lj'`` raises with a pointer to the
        correct reduced-unit fields.
        """
        from backend.common.engines.lammps_input.renderer import _timestep_in_units

        with pytest.raises(ValueError, match="timestep_lj_reduced"):
            _timestep_in_units(1.0, "lj")


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------


class TestEnsembles:
    @pytest.fixture
    def si_struct(self):
        return _cubic(5.43, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    def test_nve_emits_fix_nve(self, si_struct):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        r = generate_lammps_input(
            si_struct, LAMMPSInputParams(ensemble="nve"),
        )
        assert "fix             1 all nve" in r.input_text
        # No thermostat or barostat lines
        assert "nvt" not in r.input_text
        assert "npt" not in r.input_text
        assert "langevin" not in r.input_text

    def test_nvt_langevin_emits_nve_plus_langevin(self, si_struct):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        r = generate_lammps_input(
            si_struct, LAMMPSInputParams(ensemble="nvt_langevin", temperature_k=500.0),
        )
        assert "fix             1 all nve" in r.input_text
        assert "fix             2 all langevin 500.0000 500.0000" in r.input_text

    def test_npt_emits_parrinello_rahman_fix(self, si_struct):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input,
        )

        r = generate_lammps_input(
            si_struct,
            LAMMPSInputParams(
                ensemble="npt", temperature_k=400.0, pressure_bar=1.0,
                timestep_fs=0.5,
            ),
        )
        # NPT line mentions temp + pressure
        assert "all npt temp 400.0000" in r.input_text
        assert "iso 1.0000 1.0000" in r.input_text


# ---------------------------------------------------------------------------
# write_lammps_inputs
# ---------------------------------------------------------------------------


class TestWriteInputs:
    def test_writes_input_data_and_copies_potential(
        self, patched_default_registry, tmp_path,
    ):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input, write_lammps_inputs,
        )

        struct = _cu_108_atoms()
        r = generate_lammps_input(
            struct, LAMMPSInputParams(duration_ps=1.0),
            registry=patched_default_registry,
        )
        run_dir = tmp_path / "run"
        paths = write_lammps_inputs(r, run_dir)

        assert paths["input"].is_file()
        assert paths["data"].is_file()
        # Potential was copied alongside under its bare filename.
        assert paths["potential"].name == "Cu_u3.eam"
        assert paths["potential"].parent == run_dir
        # Input file references the bare filename so LAMMPS resolves
        # it relative to the run dir.
        assert "pair_coeff * * Cu_u3.eam" in paths["input"].read_text()

    def test_lj_requires_no_potential_copy(self, tmp_path):
        from backend.common.engines.lammps_input import (
            LAMMPSInputParams, generate_lammps_input, write_lammps_inputs,
        )

        struct = _cubic(10.0, ["Ar"], [[0, 0, 0]])
        r = generate_lammps_input(
            struct, LAMMPSInputParams(duration_ps=0.001),
        )
        assert r.forcefield.name == "lj"
        paths = write_lammps_inputs(r, tmp_path / "lj_run")
        assert "input" in paths and "data" in paths
        assert "potential" not in paths
