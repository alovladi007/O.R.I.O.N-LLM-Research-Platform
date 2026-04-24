"""Phase 7 / Session 7.1 — Bayesian-optimization engine (BoTorch wrapper).

Wraps `BoTorch <https://botorch.org/>`_ behind a small API surface that
the rest of ORION (campaigns, agent loop, ``POST /api/v1/bo/suggest``)
can drive without learning the BoTorch idiom. Single- and
multi-objective optimization both go through one entry point,
:func:`suggest`, which dispatches on ``len(objectives)``.

Design
------

* **Search spaces.** :class:`ContinuousSpace` (box-constrained box-uniform),
  :class:`SimplexSpace` (composition fractions on the unit simplex,
  i.e. ``sum(x_i) == 1`` with ``x_i >= 0``), :class:`IntegerSpace`
  (integer dimensions, e.g. supercell sizes), and :class:`CategoricalSpace`
  (one-of-k, used for restricted space-group subsets). Mixed spaces are
  combined via :class:`Space`, which produces a single ``[N, d]`` tensor
  the GP and acquisition can consume.

* **Constraints.** Two flavors —

  - **Linear inequality** (``a · x <= b``) for things like "formation
    energy ≤ 50 meV/atom". These get pushed into the acquisition's
    optimizer via ``inequality_constraints`` so candidates are
    rejection-sampled before evaluation.
  - **Categorical / structural** (charge neutrality, prototype
    blacklists) handled with a Python ``feasibility_fn`` that we apply
    to the raw-space candidate after the optimizer returns. Infeasible
    candidates are *not* returned to the caller; the optimizer is
    re-run with extra restarts until ``q`` feasible suggestions land.

* **Acquisition.** SO uses ``qLogExpectedImprovement`` (numerically
  more stable than the textbook EI; BoTorch's recommended default for
  q≥1 since v0.10). MO uses ``qExpectedHypervolumeImprovement`` —
  qEHVI per the roadmap — with a reference point either supplied or
  auto-derived from the worst observed point per objective with a
  10% pad (BoTorch convention).

* **Numerics.** ``torch.float64`` everywhere. GP hyperparameters refit
  per call from random restarts; standardized targets (mean-0 unit-
  std) so EI doesn't get swamped by an offset-rich loss surface.

The legacy ``backend/common/ml/bo.py`` is a from-scratch scipy stub
from a long-deleted "Session 19"; nothing in the canonical codebase
imports it, so v2 lives alongside without disturbing it (consistent
with the ``active_learning_v2`` / ``datasets_v2`` / ``cgcnn_v2``
pattern from Phase 6).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood

logger = logging.getLogger(__name__)

# Numerical defaults pinned for reproducibility. BoTorch's own examples
# use float64 + double-precision linalg; mixing with float32 trips
# Cholesky failures on the small init batches typical of materials BO.
_DTYPE = torch.float64
_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Search-space primitives
# ---------------------------------------------------------------------------


@dataclass
class ContinuousDim:
    """A scalar continuous dimension constrained to ``[low, high]``."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError(
                f"continuous dim {self.name!r} bounds must be finite"
            )
        if self.high <= self.low:
            raise ValueError(
                f"continuous dim {self.name!r}: high {self.high} must "
                f"exceed low {self.low}"
            )


@dataclass
class IntegerDim:
    """Inclusive integer range ``[low, high]`` (e.g. supercell sizes)."""

    name: str
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"integer dim {self.name!r}: high {self.high} < low {self.low}"
            )


@dataclass
class CategoricalDim:
    """Discrete one-of-k dimension. Choices are arbitrary hashables —
    they're encoded as integer indices internally."""

    name: str
    choices: Sequence[Any]

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(
                f"categorical dim {self.name!r} must have ≥ 2 choices; "
                f"got {len(self.choices)}"
            )


@dataclass
class SimplexSpace:
    """``k`` non-negative continuous dims that must sum to 1.

    Used for composition fractions (``Sn``-Pb``-I`` perovskites etc).
    Internally we parameterize the GP on the **first ``k-1``
    components** and recover the last as ``1 - sum(others)``; this
    keeps the GP search space full-dimensional while enforcing the
    equality constraint exactly.

    Caller doesn't need to know about that — :meth:`Space.encode` /
    :meth:`Space.decode` handle the round-trip.
    """

    name: str
    components: Sequence[str]
    minimum: float = 0.0  # per-component lower bound (e.g. 0.05 for "no zero entries")

    def __post_init__(self) -> None:
        if len(self.components) < 2:
            raise ValueError(
                f"simplex {self.name!r} needs ≥ 2 components; got "
                f"{len(self.components)}"
            )
        if self.minimum < 0.0 or self.minimum * len(self.components) >= 1.0:
            raise ValueError(
                f"simplex {self.name!r}: minimum={self.minimum} infeasible "
                f"with {len(self.components)} components"
            )


# A Dim is anything we know how to bound + encode.
Dim = Union[ContinuousDim, IntegerDim, CategoricalDim, SimplexSpace]


@dataclass
class Space:
    """A list of dims that together form the search space.

    Methods
    -------
    bounds()
        ``(2, d)`` tensor of low/high in **encoded** coordinates that
        BoTorch's optimizer wants. Categorical dims encode as
        ``[0, len(choices) - 1]``; simplex dims encode as their first
        ``k-1`` components, each in ``[minimum, 1 - (k-1)·minimum]``.
    encode(point)
        Forward map: human-readable point dict → encoded tensor row.
    decode(row)
        Inverse map: encoded tensor row → human-readable point dict.
        Categorical indices snap to nearest int; integer dims round.

    The d returned by :meth:`encoded_dim` is the GP's input
    dimensionality — it's ``len(continuous) + len(integer) +
    len(categorical) + sum(k-1 for k in simplex_components)``.
    """

    dims: List[Dim]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    @property
    def names(self) -> List[str]:
        return [d.name for d in self.dims]

    def encoded_dim(self) -> int:
        n = 0
        for d in self.dims:
            if isinstance(d, SimplexSpace):
                n += len(d.components) - 1
            else:
                n += 1
        return n

    # ------------------------------------------------------------------
    # Bounds + sampling
    # ------------------------------------------------------------------

    def bounds(self) -> torch.Tensor:
        lo: List[float] = []
        hi: List[float] = []
        for d in self.dims:
            if isinstance(d, ContinuousDim):
                lo.append(d.low)
                hi.append(d.high)
            elif isinstance(d, IntegerDim):
                # Optimize in continuous-relaxation; round on decode.
                lo.append(float(d.low))
                hi.append(float(d.high))
            elif isinstance(d, CategoricalDim):
                lo.append(0.0)
                hi.append(float(len(d.choices) - 1))
            elif isinstance(d, SimplexSpace):
                k = len(d.components)
                # Each free component lives in
                # [minimum, 1 - (k-1)·minimum]; the last component is
                # implicit and remains in the same range by symmetry.
                low = d.minimum
                high = 1.0 - (k - 1) * d.minimum
                for _ in range(k - 1):
                    lo.append(low)
                    hi.append(high)
            else:
                raise TypeError(f"unknown dim type: {type(d)}")
        return torch.tensor([lo, hi], dtype=_DTYPE, device=_DEVICE)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(self, point: Dict[str, Any]) -> torch.Tensor:
        """Forward map. Missing keys raise ``KeyError``."""
        out: List[float] = []
        for d in self.dims:
            v = point[d.name]
            if isinstance(d, ContinuousDim):
                out.append(float(v))
            elif isinstance(d, IntegerDim):
                out.append(float(int(v)))
            elif isinstance(d, CategoricalDim):
                if v not in d.choices:
                    raise ValueError(
                        f"{d.name!r}: value {v!r} not in choices {list(d.choices)}"
                    )
                out.append(float(d.choices.index(v)))
            elif isinstance(d, SimplexSpace):
                vec = list(v)
                if len(vec) != len(d.components):
                    raise ValueError(
                        f"simplex {d.name!r} expects {len(d.components)} "
                        f"components; got {len(vec)}"
                    )
                if not math.isclose(sum(vec), 1.0, abs_tol=1e-6):
                    raise ValueError(
                        f"simplex {d.name!r}: components must sum to 1; "
                        f"got {sum(vec)}"
                    )
                out.extend(float(x) for x in vec[:-1])
        return torch.tensor(out, dtype=_DTYPE, device=_DEVICE)

    def decode(self, row: torch.Tensor) -> Dict[str, Any]:
        """Inverse map for a single encoded row.

        Integer + categorical dims snap by ``round`` and ``clip``.
        """
        out: Dict[str, Any] = {}
        i = 0
        row = row.detach().cpu().numpy()
        for d in self.dims:
            if isinstance(d, ContinuousDim):
                out[d.name] = float(row[i])
                i += 1
            elif isinstance(d, IntegerDim):
                v = int(round(float(row[i])))
                v = max(d.low, min(d.high, v))
                out[d.name] = v
                i += 1
            elif isinstance(d, CategoricalDim):
                idx = int(round(float(row[i])))
                idx = max(0, min(len(d.choices) - 1, idx))
                out[d.name] = d.choices[idx]
                i += 1
            elif isinstance(d, SimplexSpace):
                k = len(d.components)
                free = np.array(row[i:i + (k - 1)], dtype=np.float64)
                free = np.clip(free, d.minimum, None)
                last = 1.0 - free.sum()
                if last < d.minimum - 1e-9:
                    # Renormalize to put the deficit back into the free
                    # components; rare on optimizer output but happens
                    # on user-supplied points.
                    excess = d.minimum - last
                    free = free - excess / (k - 1)
                    free = np.clip(free, d.minimum, None)
                    last = 1.0 - free.sum()
                vec = np.concatenate([free, [last]])
                # Numerical safety: re-project onto the simplex.
                vec = np.clip(vec, 0.0, 1.0)
                vec = vec / vec.sum()
                out[d.name] = {c: float(v) for c, v in zip(d.components, vec)}
                i += (k - 1)
        return out


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


# A linear inequality over the encoded space:
# ``coeffs @ x <= rhs``  →  rewritten as the BoTorch convention
# ``sum(coeffs[idx] * x[idx]) >= -rhs`` via sign flip in the helper.
@dataclass
class LinearInequality:
    """``coeff[name] * x[name] + ... <= rhs`` over the encoded space.

    Used for things like "formation_energy_eV_per_atom <= 0.05" when
    ``formation_energy_eV_per_atom`` is one of the encoded continuous
    dims (i.e. you're treating it as a constraint variable rather than
    an objective).
    """

    coeffs: Dict[str, float]
    rhs: float


# Python predicate evaluated on the **decoded** point. Returns True if
# feasible. Used for charge neutrality, prototype blacklists, anything
# that's hard to express as a linear inequality.
FeasibilityFn = Callable[[Dict[str, Any]], bool]


def _to_botorch_ineq(
    space: Space, ineqs: Sequence[LinearInequality],
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, float]]]:
    """Translate :class:`LinearInequality` to BoTorch's
    ``inequality_constraints`` triples ``(indices, coeffs, rhs)`` where
    ``sum(coeffs * x[indices]) >= rhs``.

    Returns ``None`` (rather than an empty list) when no constraints
    are supplied — saves BoTorch a code path.
    """
    if not ineqs:
        return None
    # Build a name → encoded-index map. Only continuous + integer +
    # categorical dims contribute single columns; SimplexSpaces contribute
    # k-1 columns whose names we don't expose for inequality use today
    # (open follow-up: per-component simplex inequality syntax).
    idx: Dict[str, int] = {}
    cursor = 0
    for d in space.dims:
        if isinstance(d, SimplexSpace):
            cursor += len(d.components) - 1
        else:
            idx[d.name] = cursor
            cursor += 1
    out: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
    for c in ineqs:
        try:
            cols = [idx[name] for name in c.coeffs]
        except KeyError as exc:
            raise ValueError(
                f"linear inequality references unknown dim {exc.args[0]!r}; "
                f"known: {sorted(idx)}"
            ) from exc
        coefs = [c.coeffs[name] for name in c.coeffs]
        # BoTorch: sum(coeffs * x[indices]) >= rhs.  We have:
        #   coeffs · x <= rhs    ↔    -coeffs · x >= -rhs
        out.append((
            torch.tensor(cols, dtype=torch.long, device=_DEVICE),
            torch.tensor([-c for c in coefs], dtype=_DTYPE, device=_DEVICE),
            float(-c.rhs),
        ))
    return out


# ---------------------------------------------------------------------------
# Objectives + history
# ---------------------------------------------------------------------------


@dataclass
class Objective:
    """One objective. ``minimize=True`` flips sign internally so the
    GP and acquisition can assume maximization."""

    name: str
    minimize: bool = False


@dataclass
class HistoryPoint:
    """One observation: encoded inputs + the realized objective values
    in the **same order** as ``Objective``s.

    Build via :meth:`Space.encode` for a human-readable point.
    """

    x_encoded: torch.Tensor  # (d,)
    y: List[float]           # one entry per Objective


@dataclass
class History:
    """Container for past observations.

    Convert to GP inputs via :meth:`tensors`, which returns
    ``(X, Y)`` with ``Y`` already sign-flipped to maximization-form
    according to ``objectives``.
    """

    objectives: List[Objective]
    points: List[HistoryPoint] = field(default_factory=list)

    def add(self, x_encoded: torch.Tensor, y: Sequence[float]) -> None:
        if len(y) != len(self.objectives):
            raise ValueError(
                f"history.add expected {len(self.objectives)} y values; "
                f"got {len(y)}"
            )
        self.points.append(HistoryPoint(
            x_encoded=x_encoded.to(dtype=_DTYPE, device=_DEVICE),
            y=[float(v) for v in y],
        ))

    def tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.points:
            raise ValueError("history is empty")
        X = torch.stack([p.x_encoded for p in self.points], dim=0)
        Y = torch.tensor(
            [p.y for p in self.points], dtype=_DTYPE, device=_DEVICE,
        )
        # Flip minimization objectives so GP sees maximization.
        signs = torch.tensor(
            [-1.0 if o.minimize else 1.0 for o in self.objectives],
            dtype=_DTYPE, device=_DEVICE,
        )
        return X, Y * signs


# ---------------------------------------------------------------------------
# Initial design (Sobol)
# ---------------------------------------------------------------------------


def initial_design(
    space: Space, n: int, *, seed: int = 0,
) -> List[Dict[str, Any]]:
    """Sobol-sequence initial design over the encoded bounds, then
    decoded back to human-readable points.

    Used to warm up a fresh BO run before any GP fits make sense.
    """
    bounds = space.bounds()
    # draw_sobol_samples returns shape (n, q=1, d); squeeze the q axis.
    rng = torch.manual_seed(seed)
    del rng  # silence "unused"
    raw = draw_sobol_samples(
        bounds=bounds, n=n, q=1, seed=seed,
    ).squeeze(1)
    return [space.decode(row) for row in raw]


# ---------------------------------------------------------------------------
# GP fitting
# ---------------------------------------------------------------------------


def _fit_gp(X: torch.Tensor, Y: torch.Tensor) -> Union[SingleTaskGP, ModelListGP]:
    """Fit a (model-list of) ``SingleTaskGP`` with input + output
    standardization.

    For ``Y.shape[-1] == 1``, a single SingleTaskGP. For multi-objective,
    one SingleTaskGP per objective wrapped in a ``ModelListGP`` —
    BoTorch's qEHVI requires this shape.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D; got shape {Y.shape}")
    n_obj = Y.shape[-1]
    bounds_in = torch.stack(
        [X.min(dim=0).values, X.max(dim=0).values], dim=0,
    )
    # Pad zero-variance columns so Normalize doesn't divide by zero on
    # constant integer / categorical dims. The +1 is arbitrary; only
    # the ratio matters for the normalize transform.
    span = bounds_in[1] - bounds_in[0]
    bounds_in[1] = torch.where(
        span == 0, bounds_in[0] + 1.0, bounds_in[1],
    )
    if n_obj == 1:
        gp = SingleTaskGP(
            train_X=X, train_Y=Y,
            input_transform=Normalize(d=X.shape[-1], bounds=bounds_in),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp
    models = []
    for j in range(n_obj):
        Yj = Y[:, j:j + 1]
        m = SingleTaskGP(
            train_X=X, train_Y=Yj,
            input_transform=Normalize(d=X.shape[-1], bounds=bounds_in),
            outcome_transform=Standardize(m=1),
        )
        models.append(m)
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ---------------------------------------------------------------------------
# Suggest
# ---------------------------------------------------------------------------


def suggest(
    *,
    space: Space,
    objectives: List[Objective],
    history: History,
    q: int = 1,
    inequalities: Optional[Sequence[LinearInequality]] = None,
    feasibility_fn: Optional[FeasibilityFn] = None,
    ref_point: Optional[Sequence[float]] = None,
    num_restarts: int = 10,
    raw_samples: int = 256,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Suggest ``q`` next points to evaluate.

    Dispatches on ``len(objectives)``:

    - 1 objective → ``qLogExpectedImprovement`` over a SingleTaskGP.
    - ≥ 2 objectives → ``qLogExpectedHypervolumeImprovement`` over a
      ModelListGP. ``ref_point`` is required (or auto-derived from
      observed worst per objective with a 10% pad below).

    Constraints
    -----------
    ``inequalities`` are pushed down to BoTorch's
    ``inequality_constraints`` and applied in the optimizer's L-BFGS
    inner loop. ``feasibility_fn`` is applied **after** decoding, in
    the user's natural coordinate system; infeasible candidates are
    dropped and the optimizer rerun (up to 5 retries with growing
    ``num_restarts`` and ``raw_samples``).

    Returns
    -------
    List of decoded candidate dicts of length ``q``.
    """
    if not objectives:
        raise ValueError("at least one Objective is required")
    if q < 1:
        raise ValueError("q must be ≥ 1")
    if seed is not None:
        torch.manual_seed(seed)

    X, Y_max = history.tensors()  # Y is already maximization-flipped.
    if X.shape[0] < 2:
        raise ValueError(
            f"history needs ≥ 2 observations to fit a GP; got {X.shape[0]}"
        )

    model = _fit_gp(X, Y_max)
    bounds = space.bounds()
    ineq = _to_botorch_ineq(space, inequalities or [])

    if len(objectives) == 1:
        acq = qLogExpectedImprovement(
            model=model, best_f=Y_max.max().item(),
        )
    else:
        if ref_point is None:
            # 10% below the worst observed point per objective, in the
            # *maximization* frame (so smaller is worse). Standard
            # BoTorch tutorial recipe — see
            # examples/multi_objective_bo.py.
            worst = Y_max.min(dim=0).values
            pad = 0.1 * (Y_max.max(dim=0).values - worst).abs().clamp_min(1e-6)
            rp = (worst - pad).tolist()
        else:
            # If user supplies a ref point in *natural* coords, flip
            # signs on minimization objectives so the comparison is
            # consistent with Y_max.
            rp = [
                (-r if o.minimize else r)
                for r, o in zip(ref_point, objectives)
            ]
        acq = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=torch.tensor(rp, dtype=_DTYPE, device=_DEVICE),
            partitioning=DominatedPartitioning(
                ref_point=torch.tensor(rp, dtype=_DTYPE, device=_DEVICE),
                Y=Y_max,
            ),
        )

    # Optimizer loop with feasibility retries.
    decoded: List[Dict[str, Any]] = []
    attempt = 0
    while len(decoded) < q and attempt < 5:
        attempt += 1
        cand_x, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts * attempt,
            raw_samples=raw_samples * attempt,
            inequality_constraints=ineq,
            options={"batch_limit": 5, "maxiter": 200},
        )
        for row in cand_x:
            point = space.decode(row)
            if feasibility_fn is None or feasibility_fn(point):
                decoded.append(point)
                if len(decoded) == q:
                    break
        if len(decoded) >= q:
            break
        logger.info(
            "BO attempt %d returned %d/%d feasible points; retrying with "
            "more restarts", attempt, len(decoded), q,
        )
    if len(decoded) < q:
        raise RuntimeError(
            f"could not produce {q} feasible BO candidates after {attempt} "
            "attempts — relax the feasibility_fn or expand the search space"
        )
    return decoded[:q]


# ---------------------------------------------------------------------------
# Diagnostics — Pareto front + IGD (used by acceptance + the API summary)
# ---------------------------------------------------------------------------


def pareto_front(Y: np.ndarray, *, minimize: Sequence[bool]) -> np.ndarray:
    """Boolean mask of non-dominated rows in a ``(n, m)`` objective array.

    Convention: a point dominates another if it's no worse in every
    objective and strictly better in at least one. ``minimize[j] ==
    True`` means smaller is better for objective ``j``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D; got shape {Y.shape}")
    n, m = Y.shape
    if len(minimize) != m:
        raise ValueError(
            f"minimize length {len(minimize)} doesn't match m={m}"
        )
    # Flip to maximization frame so the dominance check is uniform.
    sgn = np.where(np.asarray(minimize), -1.0, 1.0)
    Yp = Y * sgn
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # ``Yp[i]`` is dominated if any other ``Yp[k]`` is ≥ everywhere
        # and > somewhere.
        dominated = (Yp >= Yp[i]).all(axis=1) & (Yp > Yp[i]).any(axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def igd(approx_front: np.ndarray, true_front: np.ndarray) -> float:
    """Inverted Generational Distance — average min-distance from each
    point on the *true* Pareto front to its nearest neighbor on the
    *approximation*. Lower is better; 0 = perfect.

    Used by the ZDT2 acceptance to verify the MO loop's front
    converges over iterations.
    """
    A = np.asarray(approx_front, dtype=np.float64)
    P = np.asarray(true_front, dtype=np.float64)
    if A.ndim != 2 or P.ndim != 2:
        raise ValueError(
            f"both fronts must be 2-D; got shapes {A.shape}, {P.shape}"
        )
    if A.shape[1] != P.shape[1]:
        raise ValueError(
            f"front dim mismatch: approx m={A.shape[1]}, true m={P.shape[1]}"
        )
    if A.shape[0] == 0:
        return float("inf")
    # Pairwise Euclidean distances, then min over the approximation
    # axis for every true-front point.
    diffs = P[:, None, :] - A[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    return float(dist.min(axis=1).mean())


__all__ = [
    "CategoricalDim",
    "ContinuousDim",
    "FeasibilityFn",
    "History",
    "HistoryPoint",
    "IntegerDim",
    "LinearInequality",
    "Objective",
    "SimplexSpace",
    "Space",
    "igd",
    "initial_design",
    "pareto_front",
    "suggest",
]
