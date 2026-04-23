"""Reusable LAMMPS MD workflow DAG specs.

Session 4.3 scope: four templates from the roadmap.

- ``equilibrate_nvt_then_nve`` — NVT warm-up → NVE production with
  RDF/MSD on the NVE leg. **Fully shipped.**
- ``melting_curve`` — temperature sweep via ``foreach``, detect
  melting via MSD jump + enthalpy discontinuity. **DAG shipped;
  post-analyzer (melting detection) deferred to Session 4.3b.**
- ``diffusivity_vs_T`` — temperature sweep, Arrhenius fit.
  **DAG shipped; Arrhenius fitter deferred to 4.3b.**
- ``elastic_constants_via_strain`` — apply ±ε strains, fit C_ij.
  **DAG shipped; C_ij solver deferred to 4.3b.**

The three deferred post-analyzers live in
:mod:`backend.common.reports.md`; they currently raise
:class:`NotImplementedError` rather than returning silently-wrong
numbers. The DAGs still dispatch correctly through the Session 2.4
tick executor — users who want the raw MD runs can use these
templates today; the aggregate-report endpoint errors loudly until
the analyzer lands.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..spec import StepSpec, WorkflowSpec


# ---------------------------------------------------------------------------
# Template 1: equilibrate_nvt_then_nve (ships end-to-end)
# ---------------------------------------------------------------------------


def equilibrate_nvt_then_nve_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    temperature_k: float = 300.0,
    equilibrate_ps: float = 20.0,
    production_ps: float = 20.0,
    timestep_fs: float = 1.0,
) -> WorkflowSpec:
    """NVT warm-up → NVE production.

    The NVE leg is where structural (RDF) and transport (MSD) analyses
    are meaningful — NVT Langevin adds friction that kills long-time
    diffusion, so D from NVE > D from NVT.

    Both legs run on the same structure; the NVE leg depends on NVT
    via a ``uses`` edge so the workflow executor orders them.
    """
    steps = [
        StepSpec(
            id="equilibrate",
            kind="md_nvt",
            structure_id=structure_id,
            inputs={
                "ensemble": "nvt_langevin",
                "temperature_k": temperature_k,
                "timestep_fs": timestep_fs,
                "duration_ps": equilibrate_ps,
                "thermo_every": 100,
                "dump_every": 1000,  # sparse dumps during warm-up
            },
        ),
        StepSpec(
            id="production",
            kind="md_nve",
            structure_id=structure_id,
            inputs={
                "ensemble": "nve",
                "temperature_k": temperature_k,  # used for initial velocity
                "timestep_fs": timestep_fs,
                "duration_ps": production_ps,
                "thermo_every": 100,
                "dump_every": 100,  # denser dumps for analyzer quality
                # Depend on equilibrate completing (reference forces ordering).
                "_equilibrate_energy": {
                    "uses": "equilibrate.outputs.final_thermo",
                },
            },
        ),
    ]
    return WorkflowSpec(
        name=name or "equilibrate_nvt_then_nve",
        description="MD NVT warm-up followed by NVE production (RDF/MSD on NVE).",
        steps=steps,
        default_structure_id=structure_id,
    )


# ---------------------------------------------------------------------------
# Template 2: melting_curve (DAG shipped, post-analyzer in 4.3b)
# ---------------------------------------------------------------------------


def melting_curve_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    temperatures_k: Sequence[float] = (800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600),
    duration_ps_per_point: float = 20.0,
    timestep_fs: float = 1.0,
    ensemble: str = "npt",
) -> WorkflowSpec:
    """Temperature sweep to locate melting via MSD jump.

    Ships the fan-out workflow DAG: N independent MD runs at N
    different temperatures. The melting-detection post-analyzer
    (``backend.common.reports.md.detect_melting_point``) is not yet
    implemented — a 4.3b followup will add the MSD-jump +
    enthalpy-discontinuity detection.

    Users who want the raw runs today can submit this template; to
    interpret the result they'd either wait for 4.3b or inspect the
    per-step outputs manually.
    """
    sweep_step = StepSpec(
        id="sweep",
        kind=f"md_{ensemble}" if ensemble != "nvt_langevin" else "md_nvt",
        structure_id=structure_id,
        inputs={
            "ensemble": ensemble,
            "timestep_fs": timestep_fs,
            "duration_ps": duration_ps_per_point,
            "thermo_every": 200,
            "dump_every": 1000,
            "pressure_bar": 1.0 if ensemble == "npt" else 1.0,
        },
        foreach={"temperature_k": list(temperatures_k)},
    )
    return WorkflowSpec(
        name=name or "melting_curve",
        description=(
            f"MD temperature sweep: {len(temperatures_k)} points from "
            f"{min(temperatures_k)} to {max(temperatures_k)} K. "
            "Melting-detection post-analyzer lands in Session 4.3b."
        ),
        steps=[sweep_step],
        default_structure_id=structure_id,
    )


# ---------------------------------------------------------------------------
# Template 3: diffusivity_vs_T (DAG shipped, Arrhenius fit in 4.3b)
# ---------------------------------------------------------------------------


def diffusivity_vs_t_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    temperatures_k: Sequence[float] = (400, 500, 600, 700, 800),
    duration_ps_per_point: float = 50.0,
    timestep_fs: float = 1.0,
) -> WorkflowSpec:
    """Temperature sweep for D(T) → Arrhenius fit.

    Fan-out produces one NVE production per temperature (preceded by
    an implicit NVT warm-up is NOT done here — each sweep point is a
    single NVE run with `init_velocity=True` at the target T). For a
    clean production run per point, users should compose with
    ``equilibrate_nvt_then_nve`` — a future 4.3b could wire that up.

    The Arrhenius fit (``backend.common.reports.md.arrhenius_fit``)
    is deferred to 4.3b.
    """
    sweep_step = StepSpec(
        id="sweep",
        kind="md_nve",
        structure_id=structure_id,
        inputs={
            "ensemble": "nve",
            "timestep_fs": timestep_fs,
            "duration_ps": duration_ps_per_point,
            "thermo_every": 200,
            "dump_every": 100,  # dense for MSD
        },
        foreach={"temperature_k": list(temperatures_k)},
    )
    return WorkflowSpec(
        name=name or "diffusivity_vs_T",
        description=(
            f"MD diffusion sweep: {len(temperatures_k)} points. "
            "Arrhenius fitter lands in Session 4.3b."
        ),
        steps=[sweep_step],
        default_structure_id=structure_id,
    )


# ---------------------------------------------------------------------------
# Template 4: elastic_constants_via_strain (DAG shipped, C_ij solver in 4.3b)
# ---------------------------------------------------------------------------


# Six independent strain directions for a 3D elastic tensor in Voigt
# notation. Each entry is (id_suffix, voigt_index, signed_strain).
# Session 4.3b's C_ij solver assembles the stress-vs-strain slopes.
_ELASTIC_STRAINS: List[tuple[str, int, float]] = [
    ("e11_plus", 0, +0.005),
    ("e11_minus", 0, -0.005),
    ("e22_plus", 1, +0.005),
    ("e22_minus", 1, -0.005),
    ("e33_plus", 2, +0.005),
    ("e33_minus", 2, -0.005),
    # Shear strains — skipped for MVP; cubic-system C_ij can be
    # obtained from the three diagonal pairs + bulk modulus, so the
    # shears aren't strictly required for the roadmap's "C11 within
    # 15% of 108 GPa for Al" target.
]


def elastic_constants_via_strain_spec(
    structure_id: str,
    *,
    name: Optional[str] = None,
    temperature_k: float = 300.0,
    duration_ps: float = 10.0,
    timestep_fs: float = 1.0,
) -> WorkflowSpec:
    """Apply ±ε strains, measure stress → fit C_ij.

    Ships six NVT runs, one per ±0.5% strain on each diagonal axis.
    The C_ij post-fitter
    (``backend.common.reports.md.fit_elastic_constants``) uses
    Hooke's law σ = C ε to extract the diagonal elastic constants.
    Currently raises NotImplementedError; lands in 4.3b.

    Caveats documented in 4.3b:
    - Strain application currently piggybacks on LAMMPS' ``change_box``
      at the start of each MD run. That path isn't wired through
      ``LAMMPSInputParams`` yet — users would need a custom
      ``extra_commands`` block. Another 4.3b item.
    - Shear (off-diagonal) strains aren't generated here; C_44 etc.
      require full deformation workflow.
    """
    steps: List[StepSpec] = []
    for suffix, _voigt_idx, strain in _ELASTIC_STRAINS:
        steps.append(
            StepSpec(
                id=f"strain_{suffix}",
                kind="md_nvt",
                structure_id=structure_id,
                inputs={
                    "ensemble": "nvt_langevin",
                    "temperature_k": temperature_k,
                    "timestep_fs": timestep_fs,
                    "duration_ps": duration_ps,
                    "thermo_every": 100,
                    "dump_every": 1000,
                    # 4.3b will prepend a change_box command here. For
                    # now this metadata flows into outputs as a marker.
                    "_elastic_strain_voigt_index": _voigt_idx,
                    "_elastic_strain_value": strain,
                },
            )
        )
    return WorkflowSpec(
        name=name or "elastic_constants_via_strain",
        description=(
            "Six ±ε diagonal-strain MD runs. C_ij solver + change_box "
            "prepend land in Session 4.3b."
        ),
        steps=steps,
        default_structure_id=structure_id,
    )


SPEC_BUILDERS: Dict[str, Any] = {
    "equilibrate_nvt_then_nve": equilibrate_nvt_then_nve_spec,
    "melting_curve": melting_curve_spec,
    "diffusivity_vs_T": diffusivity_vs_t_spec,
    "elastic_constants_via_strain": elastic_constants_via_strain_spec,
}
