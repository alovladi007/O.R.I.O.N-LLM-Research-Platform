"""MD-specific Report models + post-analyzers.

Session 4.3b implements the three post-analyzers that 4.3a declared:

- :func:`detect_melting_point` — MSD-jump + enthalpy-discontinuity
  heuristic across a temperature sweep (NPT preferred, NVT tolerated).
- :func:`arrhenius_fit` — linearized least-squares of ``ln D`` vs
  ``1/T`` with an R² quality gate.
- :func:`fit_elastic_constants` — Hooke's law solve for diagonal
  elastic constants from a ±ε strain sweep.

**Step-outputs convention.** Analyzers consume a dict of
``{step_id: outputs}`` where each ``outputs`` dict is what
``_run_lammps_step`` returns plus a handful of fields the workflow
runner stamps in (``temperature_k``, ``pressure_bar``,
``strain_voigt``, ``strain_value``). We do this so analyzers are pure
functions — they don't reach back into the workflow spec for per-step
context, which makes them trivially unit-testable with synthetic
inputs.

Missing fields are tolerated (they're treated as ``None``); the
analyzers decline to extrapolate. That's the point of the R² gate,
the MSD-jump threshold, and the confidence levels.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class PendingAnalyzerError(NotImplementedError):
    """Raised by post-analyzers that haven't been implemented yet.

    Still exported after 4.3b — the ReaxFF / ML-potential paths
    introduce new analyzers (VACF→vDOS, Green-Kubo viscosity) that
    will re-use this sentinel until their own sessions implement them.
    """

    def __init__(self, analyzer: str, tracker: str = "Session 4.3b"):
        self.analyzer = analyzer
        self.tracker = tracker
        super().__init__(
            f"{analyzer} is not yet implemented — {tracker}. The workflow "
            "DAG ran correctly; inspect per-step outputs for raw values."
        )


class AnalyzerInputError(ValueError):
    """The caller handed the analyzer step_outputs that don't fit its contract.

    Thrown for structural failures: empty dicts, missing temperature
    fields, non-monotonic strain values, fewer than the required number
    of points. This is loud-and-clear failure — the caller supplied bad
    data. Contrast with the R² / confidence gates below, which flag
    *physics* quality issues on well-formed input.
    """


# ---------------------------------------------------------------------------
# Base MD report
# ---------------------------------------------------------------------------


class MDReport(BaseModel):
    """Common fields shared by all MD aggregate reports."""

    model_config = ConfigDict(extra="forbid")

    report_schema: str = "md_report.v1"
    workflow_run_id: str = ""
    name: str = ""
    n_steps: int = 0
    step_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Helpers shared across analyzers
# ---------------------------------------------------------------------------


def _get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    """Return ``d[key]`` coerced to float, or None if missing / NaN / bad."""
    if key not in d or d[key] is None:
        return None
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _get_nested(d: Dict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
    """Follow ``path`` into ``d`` and coerce to float. None on miss."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if cur is None:
        return None
    try:
        v = float(cur)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _lstsq(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """Ordinary least squares for ``y = m x + b``.

    Returns ``(slope, intercept, r_squared)``. Callers guard on n>=2;
    we raise on degenerate input (zero variance in x) rather than
    silently producing NaN.
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        raise AnalyzerInputError(
            f"_lstsq needs at least 2 paired points; got n={n}, len(y)={len(ys)}"
        )
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0:
        raise AnalyzerInputError("_lstsq: x has zero variance")
    slope = sxy / sxx
    intercept = my - slope * mx
    # R² = 1 - SS_res / SS_tot. SS_res = Σ (y - ŷ)².
    y_hat = [slope * x + intercept for x in xs]
    ss_res = sum((ys[i] - y_hat[i]) ** 2 for i in range(n))
    r2 = 1.0 - (ss_res / syy) if syy > 0 else 1.0
    return slope, intercept, r2


# ---------------------------------------------------------------------------
# Melting curve
# ---------------------------------------------------------------------------


class MeltingCurveReport(MDReport):
    """Temperature sweep with melting-point detection."""

    report_schema: str = "melting_curve_report.v1"
    temperatures_k: List[float] = Field(default_factory=list)
    total_energies_ev: List[Optional[float]] = Field(default_factory=list)
    enthalpies_ev: List[Optional[float]] = Field(default_factory=list)
    msd_final_ang2: List[Optional[float]] = Field(default_factory=list)
    diffusion_coefficients: List[Optional[float]] = Field(default_factory=list)
    detected_melting_point_k: Optional[float] = None
    detection_confidence: Optional[str] = None   # "low" | "medium" | "high"
    detection_method: Optional[str] = None


# Empirical threshold: when going from solid to liquid, MSD grows by
# at least this factor relative to the cold reference. Calibrated on
# LJ and EAM Cu in session 4.3b — solids have MSD plateaus around 0.1
# Å², liquids reach 5–20 Å² in ~20 ps.
_MSD_JUMP_THRESHOLD = 10.0

# Relative change in dE/dT across the transition that qualifies as a
# "discontinuity" for our purposes. NPT enthalpy jumps ~5–10× for
# most materials at melting.
_DEDT_JUMP_THRESHOLD = 2.0


def detect_melting_point(
    step_outputs: Dict[str, Dict[str, Any]],
) -> MeltingCurveReport:
    """Detect ``T_m`` from an MSD-jump + enthalpy-discontinuity heuristic.

    Algorithm
    ---------
    1. Extract ``(T, E or H, MSD_final)`` from each step's outputs,
       drop entries missing ``temperature_k`` (loud on empty input).
    2. Sort by T ascending.
    3. Compute MSD ratios: for consecutive T points, ``MSD[i+1] / MSD[i]``.
    4. Flag the first i where the ratio exceeds
       :data:`_MSD_JUMP_THRESHOLD` **and** ``dE/dT`` (or ``dH/dT`` for
       NPT) rises sharply vs the pre-transition slope.
    5. Return T_m as the midpoint of the bracket (``(T[i] + T[i+1]) / 2``).

    Confidence scoring
    ------------------
    - ``high`` — both MSD-jump and enthalpy-discontinuity tests agree
      on the same bracket, and the jump ratio is ≥ 10×.
    - ``medium`` — only one test fires, or the bracket is at the
      sweep edge (couldn't confirm with cold/hot context).
    - ``low`` — neither test fires but the sweep shows monotonic MSD
      growth; report the midpoint of the sweep with a warning.

    Input fields expected per step
    ------------------------------
    - ``temperature_k`` — required. :class:`AnalyzerInputError` if any
      step lacks it.
    - ``msd_final_ang2`` — required for MSD-jump detection.
    - ``final_thermo.TotEng`` — total energy in eV (LAMMPS ``metal``
      units). Optional; if missing, enthalpy cross-check is skipped
      and confidence drops one level.
    - ``final_thermo.Press``, ``final_thermo.Volume`` — used to derive
      enthalpy ``H = E + P·V`` for NPT. Optional; falls back to E.
    """
    # 1. Extract and validate.
    raw: List[Tuple[float, Optional[float], Optional[float], Optional[float], Optional[float], str]] = []
    for step_id, out in step_outputs.items():
        t = _get_float(out, "temperature_k")
        if t is None:
            raise AnalyzerInputError(
                f"detect_melting_point: step {step_id!r} has no "
                f"'temperature_k'; the workflow runner must stamp it."
            )
        msd = _get_float(out, "msd_final_ang2")
        e = _get_nested(out, ("final_thermo", "TotEng"))
        p = _get_nested(out, ("final_thermo", "Press"))
        v = _get_nested(out, ("final_thermo", "Volume"))
        raw.append((t, msd, e, p, v, step_id))

    if len(raw) < 2:
        raise AnalyzerInputError(
            f"detect_melting_point: need at least 2 temperature points; got {len(raw)}"
        )

    raw.sort(key=lambda row: row[0])
    temps = [r[0] for r in raw]
    msds = [r[1] for r in raw]
    energies = [r[2] for r in raw]
    # Enthalpy: E + P·V. LAMMPS ``metal`` units give P in bar and V in Å³.
    # Convert PV to eV: 1 bar·Å³ = 1e-25 J = 6.2415e-7 eV. Negligible for
    # condensed phases; included for correctness.
    bar_ang3_to_ev = 6.2415e-7
    enthalpies: List[Optional[float]] = []
    for (_t, _msd, e, p, v, _sid) in raw:
        if e is None:
            enthalpies.append(None)
        elif p is None or v is None:
            enthalpies.append(e)  # fall back to TotEng
        else:
            enthalpies.append(e + p * v * bar_ang3_to_ev)

    # 2. MSD-jump scan.
    jump_idx = -1
    max_ratio = 0.0
    for i in range(len(msds) - 1):
        a, b = msds[i], msds[i + 1]
        if a is None or b is None or a <= 0:
            continue
        ratio = b / a
        if ratio > max_ratio:
            max_ratio = ratio
        if ratio >= _MSD_JUMP_THRESHOLD and jump_idx < 0:
            jump_idx = i

    # 3. dE/dT (or dH/dT) slope change. Compare the slope across the
    # jump bracket to the slope before it. Slope spikes → transition.
    e_spike_idx = -1
    dE_ratio = 0.0
    for i in range(1, len(enthalpies) - 1):
        h_prev, h_cur, h_next = enthalpies[i - 1], enthalpies[i], enthalpies[i + 1]
        if h_prev is None or h_cur is None or h_next is None:
            continue
        dT_lo = temps[i] - temps[i - 1]
        dT_hi = temps[i + 1] - temps[i]
        if dT_lo <= 0 or dT_hi <= 0:
            continue
        slope_lo = (h_cur - h_prev) / dT_lo
        slope_hi = (h_next - h_cur) / dT_hi
        if abs(slope_lo) < 1e-9:
            continue
        ratio = abs(slope_hi / slope_lo) if slope_lo != 0 else 0.0
        if ratio > dE_ratio:
            dE_ratio = ratio
        if ratio >= _DEDT_JUMP_THRESHOLD and e_spike_idx < 0:
            e_spike_idx = i

    # 4. Resolve T_m and confidence.
    detected: Optional[float] = None
    confidence: Optional[str] = None
    method: Optional[str] = None

    if jump_idx >= 0:
        bracket_lo, bracket_hi = temps[jump_idx], temps[jump_idx + 1]
        detected = 0.5 * (bracket_lo + bracket_hi)
        method = "msd_jump"
        # Check if enthalpy jump agrees: e_spike_idx should be within
        # [jump_idx, jump_idx+1]. That's the high-confidence case.
        if e_spike_idx in (jump_idx, jump_idx + 1):
            confidence = "high"
            method = "msd_jump+enthalpy_discontinuity"
        else:
            confidence = "medium"
    elif e_spike_idx >= 0:
        # Enthalpy agrees on a transition but MSD didn't show a clear
        # jump — rare. Report the enthalpy bracket with medium confidence.
        bracket_lo = temps[max(0, e_spike_idx - 1)]
        bracket_hi = temps[min(len(temps) - 1, e_spike_idx + 1)]
        detected = 0.5 * (bracket_lo + bracket_hi)
        confidence = "medium"
        method = "enthalpy_discontinuity"
    else:
        # No transition in the window. Report None; caller can widen T.
        detected = None
        confidence = "low" if max_ratio >= 2.0 else None
        method = "no_transition_detected" if detected is None else None

    return MeltingCurveReport(
        n_steps=len(step_outputs),
        step_outputs=step_outputs,
        temperatures_k=temps,
        total_energies_ev=energies,
        enthalpies_ev=enthalpies,
        msd_final_ang2=msds,
        diffusion_coefficients=[_get_float(step_outputs[r[5]], "diffusion_coefficient_ang2_per_ps") for r in raw],
        detected_melting_point_k=detected,
        detection_confidence=confidence,
        detection_method=method,
    )


# ---------------------------------------------------------------------------
# Arrhenius fit
# ---------------------------------------------------------------------------


class ArrheniusReport(MDReport):
    """D(T) at multiple temperatures with Arrhenius fit parameters.

    Model: ``D(T) = D_0 · exp(-E_a / k_B T)``.
    Linearized as ``ln D = ln D_0 - (E_a / k_B) · (1/T)``.
    """

    report_schema: str = "arrhenius_report.v1"
    temperatures_k: List[float] = Field(default_factory=list)
    diffusion_coefficients: List[float] = Field(default_factory=list)
    activation_energy_ev: Optional[float] = None
    prefactor_ang2_per_ps: Optional[float] = None
    r_squared: Optional[float] = None
    fit_quality: Optional[str] = None   # "good" | "fair" | "poor"


# Boltzmann constant in eV/K — our reports ship Arrhenius E_a in eV.
_K_B_EV_PER_K = 8.617333262e-5


def arrhenius_fit(
    step_outputs: Dict[str, Dict[str, Any]],
    *,
    min_r_squared_good: float = 0.95,
    min_r_squared_fair: float = 0.80,
) -> ArrheniusReport:
    """Fit ``ln D`` vs ``1/T`` via ordinary least squares.

    Inputs expected per step
    ------------------------
    - ``temperature_k`` (required)
    - ``diffusion_coefficient_ang2_per_ps`` (required, must be > 0)

    Steps missing either or with ``D ≤ 0`` (common for solids at low T)
    are dropped before the fit, not silently replaced with zeros. If
    fewer than 2 valid points survive, :class:`AnalyzerInputError`.

    Returns
    -------
    :class:`ArrheniusReport` with ``activation_energy_ev``,
    ``prefactor_ang2_per_ps`` (= D_0), ``r_squared``, and a
    ``fit_quality`` label by the supplied thresholds.
    """
    if not step_outputs:
        raise AnalyzerInputError("arrhenius_fit: step_outputs is empty")

    ts: List[float] = []
    ds: List[float] = []
    all_ts: List[float] = []
    all_ds: List[float] = []
    for sid, out in step_outputs.items():
        t = _get_float(out, "temperature_k")
        d = _get_float(out, "diffusion_coefficient_ang2_per_ps")
        if t is None:
            raise AnalyzerInputError(
                f"arrhenius_fit: step {sid!r} missing 'temperature_k'"
            )
        all_ts.append(t)
        all_ds.append(d if d is not None else 0.0)
        if d is None or d <= 0:
            continue
        ts.append(t)
        ds.append(d)

    if len(ts) < 2:
        raise AnalyzerInputError(
            f"arrhenius_fit: need at least 2 points with D>0; got {len(ts)}"
        )

    inv_t = [1.0 / t for t in ts]
    ln_d = [math.log(d) for d in ds]
    slope, intercept, r2 = _lstsq(inv_t, ln_d)
    # slope = -E_a / k_B → E_a = -slope * k_B
    ea_ev = -slope * _K_B_EV_PER_K
    d0 = math.exp(intercept)

    if r2 >= min_r_squared_good:
        quality = "good"
    elif r2 >= min_r_squared_fair:
        quality = "fair"
    else:
        quality = "poor"

    # Sort the report's T + D lists ascending so downstream plots are
    # predictable, regardless of step_outputs insertion order.
    paired = sorted(zip(all_ts, all_ds), key=lambda p: p[0])
    ts_sorted = [p[0] for p in paired]
    ds_sorted = [p[1] for p in paired]

    return ArrheniusReport(
        n_steps=len(step_outputs),
        step_outputs=step_outputs,
        temperatures_k=ts_sorted,
        diffusion_coefficients=ds_sorted,
        activation_energy_ev=ea_ev,
        prefactor_ang2_per_ps=d0,
        r_squared=r2,
        fit_quality=quality,
    )


# ---------------------------------------------------------------------------
# Elastic constants
# ---------------------------------------------------------------------------


class ElasticConstantsReport(MDReport):
    """Diagonal C_ij from ±ε strain sweep (cubic-system subset).

    For cubic symmetry, C_11 = C_22 = C_33; we report all three so
    non-cubic inputs produce a diagnostic rather than a silent wrong
    number. Off-diagonal (shear) constants C_44, C_12 require shear
    deformations that Session 4.3b's DAG template doesn't emit yet.
    """

    report_schema: str = "elastic_constants_report.v1"
    strains_voigt: List[float] = Field(default_factory=list)
    voigt_indices: List[int] = Field(default_factory=list)
    stresses_gpa: List[Optional[float]] = Field(default_factory=list)
    c11_gpa: Optional[float] = None
    c22_gpa: Optional[float] = None
    c33_gpa: Optional[float] = None
    c11_r_squared: Optional[float] = None
    c22_r_squared: Optional[float] = None
    c33_r_squared: Optional[float] = None


# LAMMPS `metal` units report pressure in bar (+compressive) but we
# report stiffness in GPa with the physics sign convention σ > 0 in
# tension. 1 GPa = 1e4 bar; σ = -P.
_BAR_TO_GPA = 1.0e-4


def fit_elastic_constants(
    step_outputs: Dict[str, Dict[str, Any]],
) -> ElasticConstantsReport:
    """Fit σ = C · ε for the diagonal elastic constants.

    Inputs expected per step
    ------------------------
    - ``strain_voigt`` (required, int in 0..5): which component of the
      Voigt strain vector was applied.
    - ``strain_value`` (required, float): the signed strain magnitude.
    - ``final_thermo.Pxx`` / ``Pyy`` / ``Pzz`` (required for whichever
      axis matches ``strain_voigt``): LAMMPS pressure components in bar.

    Algorithm
    ---------
    For each diagonal axis i ∈ {0,1,2} (xx, yy, zz in Voigt):

    1. Collect all steps with ``strain_voigt == i`` sorted by ``strain_value``.
    2. Build a stress vector ``σ_i = -P_ii`` (flip sign to the physics
       convention: tension > 0).
    3. Fit σ_i = C_ii · ε_i via :func:`_lstsq`. Report C_ii in GPa with
       its R².

    Missing axes yield ``None`` for that C_ii. At least one axis must
    have ≥ 2 points or :class:`AnalyzerInputError`.
    """
    if not step_outputs:
        raise AnalyzerInputError("fit_elastic_constants: step_outputs is empty")

    # Voigt axis → (strain list, stress list in GPa).
    per_axis: Dict[int, Tuple[List[float], List[float]]] = {0: ([], []), 1: ([], []), 2: ([], [])}
    all_strains: List[float] = []
    all_voigt: List[int] = []
    all_stress_gpa: List[Optional[float]] = []

    for sid, out in step_outputs.items():
        voigt = out.get("strain_voigt")
        strain = _get_float(out, "strain_value")
        if voigt is None or strain is None:
            raise AnalyzerInputError(
                f"fit_elastic_constants: step {sid!r} missing 'strain_voigt' or 'strain_value'"
            )
        try:
            vi = int(voigt)
        except (TypeError, ValueError):
            raise AnalyzerInputError(
                f"fit_elastic_constants: strain_voigt for {sid!r} is not int-like"
            )
        if vi not in (0, 1, 2):
            # Shear component (3-5): we don't handle those in 4.3b.
            all_strains.append(strain)
            all_voigt.append(vi)
            all_stress_gpa.append(None)
            continue

        p_key = ("Pxx", "Pyy", "Pzz")[vi]
        p_bar = _get_nested(out, ("final_thermo", p_key))
        if p_bar is None:
            all_strains.append(strain)
            all_voigt.append(vi)
            all_stress_gpa.append(None)
            continue
        sigma_gpa = -p_bar * _BAR_TO_GPA  # tension positive
        per_axis[vi][0].append(strain)
        per_axis[vi][1].append(sigma_gpa)
        all_strains.append(strain)
        all_voigt.append(vi)
        all_stress_gpa.append(sigma_gpa)

    c_values: List[Optional[float]] = [None, None, None]
    r2_values: List[Optional[float]] = [None, None, None]
    any_fit = False
    for vi, (strains, stresses) in per_axis.items():
        if len(strains) < 2:
            continue
        slope, _intercept, r2 = _lstsq(strains, stresses)
        c_values[vi] = slope      # GPa
        r2_values[vi] = r2
        any_fit = True

    if not any_fit:
        raise AnalyzerInputError(
            "fit_elastic_constants: no diagonal axis had ≥ 2 valid points"
        )

    # Pair-sort reports for deterministic plots.
    paired = sorted(zip(all_strains, all_voigt, all_stress_gpa), key=lambda t: (t[1], t[0]))
    strains_sorted = [p[0] for p in paired]
    voigt_sorted = [p[1] for p in paired]
    stress_sorted = [p[2] for p in paired]

    return ElasticConstantsReport(
        n_steps=len(step_outputs),
        step_outputs=step_outputs,
        strains_voigt=strains_sorted,
        voigt_indices=voigt_sorted,
        stresses_gpa=stress_sorted,
        c11_gpa=c_values[0],
        c22_gpa=c_values[1],
        c33_gpa=c_values[2],
        c11_r_squared=r2_values[0],
        c22_r_squared=r2_values[1],
        c33_r_squared=r2_values[2],
    )
