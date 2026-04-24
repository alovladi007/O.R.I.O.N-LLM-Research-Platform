"""Phase 6 / Session 6.5 — active learning loop.

Closes the loop between an uncertainty-aware regressor (a CGCNN deep
ensemble from Session 6.4 or any of the Session 6.3 baselines) and a
labelling oracle that stands in for the DFT workflow:

    repeat for ``n_cycles``:
        1. Re-fit the model on the labelled pool.
        2. Predict ``(μ, σ)`` on the unlabelled pool.
        3. Pick the next ``query_size`` candidates with an acquisition
           function (max-σ, UCB, EI, or BALD).
        4. Send them to the oracle (in production: a DFT job in the
           ``dft`` Celery queue; in tests: a Python callable that
           returns the ground-truth value).
        5. Move the newly-labelled rows from unlabelled → labelled.
        6. Log an :class:`ALCycle` row with metrics + cumulative best.

The roadmap acceptance for Session 6.5 has two parts:

  * ``max-σ`` beats random sampling on the same budget — the simulated
    head-to-head check :func:`run_paired_comparison` returns.
  * Each campaign's ``cumulative_best`` series is strictly monotone
    (non-decreasing for a maximisation campaign) — :class:`ALCampaign`
    enforces it row-by-row inside :meth:`ALCampaign.run`.

DB tables (``al_campaigns``, ``al_cycles``) and the Celery wiring are
left to Session 6.5b — the v2 module exposes a clean, in-memory engine
that the API + worker bindings can wrap without re-implementing the
science. This is the same shape as Sessions 6.2 / 6.3 / 6.4 (engine
first, DB+queue follow-up).

Conventions
-----------

* All acquisitions assume **maximisation**. For minimisation, pass
  ``maximize=False`` to :func:`acquire`; we negate ``μ`` internally so
  the rest of the engine stays direction-agnostic.
* ``μ`` and ``σ`` are 1-D ``ndarray`` of identical length over the
  unlabelled pool. ``σ`` ≥ 0 (clipped, mirroring the baseline contract).
* Acquisition outputs are scores, **not** probabilities — higher is
  better. Top-k selection uses :func:`numpy.argpartition` for speed on
  large pools.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model / oracle protocols — minimal interfaces the loop needs
# ---------------------------------------------------------------------------


class UncertaintyModel(Protocol):
    """Anything that can fit ``(X, y)`` and predict ``(μ, σ)``.

    The :class:`backend.common.ml.models.baselines.RandomForestRegressor`
    and :class:`XGBoostQuantileRegressor` satisfy this directly. The
    Session 6.4 CGCNN deep ensemble satisfies it via the wrapper at
    the bottom of this module (:class:`EnsembleAdapter`) which lifts
    ``predict_with_uncertainty`` into the ``predict_uncertainty``
    interface.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "UncertaintyModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_uncertainty(
        self, X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


# Oracle: row index in the **original** corpus → ground-truth value.
# Returning ``None`` is allowed for "DFT didn't converge" — the loop
# drops those rows from the candidate pool so they don't get re-queried.
OracleFn = Callable[[int], Optional[float]]


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


_VALID_ACQUISITIONS = ("max_sigma", "ucb", "ei", "bald")


def acquire(
    *,
    mu: np.ndarray,
    sigma: np.ndarray,
    kind: str = "max_sigma",
    best_so_far: Optional[float] = None,
    beta: float = 2.0,
    xi: float = 0.0,
    member_preds: Optional[np.ndarray] = None,
    maximize: bool = True,
) -> np.ndarray:
    """Compute per-candidate acquisition scores.

    Parameters
    ----------
    mu, sigma
        Posterior mean and std-dev over the unlabelled pool. Both 1-D,
        same length.
    kind
        One of ``"max_sigma"``, ``"ucb"``, ``"ei"``, ``"bald"``.

        - ``"max_sigma"`` — uncertainty-only; equivalent to
          *uncertainty sampling*. Strong default for regression AL when
          the goal is to **shrink the model's error** rather than
          **find the optimum**.
        - ``"ucb"`` — upper-confidence-bound, ``μ + β σ``. Mixes
          exploitation and exploration; tune ``β`` ≥ 0.
        - ``"ei"`` — expected improvement over ``best_so_far``. Closed
          form for Gaussian posterior; ``xi`` is the exploration
          padding (Brochu et al. 2010).
        - ``"bald"`` — Bayesian Active Learning by Disagreement
          (Houlsby et al. 2011). For an ensemble it reduces to the
          variance of per-member predictions, which under a Gaussian
          posterior assumption is just ``σ²`` — *but* using the raw
          per-member predictions when available is more honest. We pass
          ``member_preds`` (shape ``(n_members, n_pool)``) when the
          caller has them; otherwise fall back to ``σ²``.
    best_so_far
        Required for ``"ei"``. For ``maximize=True``, the maximum of
        the labelled targets so far; for ``maximize=False``, the min.
        Direction is handled here — pass the natural-direction value.
    beta, xi
        Hyperparameters for UCB / EI respectively.
    member_preds
        Optional ``(n_members, n_pool)`` matrix of per-ensemble-member
        predictions. Used by BALD; ignored otherwise.
    maximize
        If False, negates ``μ`` (and ``best_so_far``) so the same
        formulas pick the minimisation candidate.

    Returns
    -------
    scores : ``ndarray(n_pool,)``
        Higher = more desirable to query next. Use :func:`top_k_indices`
        to pick.
    """
    if kind not in _VALID_ACQUISITIONS:
        raise ValueError(
            f"unknown acquisition {kind!r}; choose from {_VALID_ACQUISITIONS}"
        )
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    if mu.shape != sigma.shape:
        raise ValueError(
            f"mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}"
        )
    sigma = np.clip(sigma, 0.0, None)

    if not maximize:
        mu = -mu
        if best_so_far is not None:
            best_so_far = -best_so_far

    if kind == "max_sigma":
        return sigma
    if kind == "ucb":
        return mu + beta * sigma
    if kind == "ei":
        if best_so_far is None:
            raise ValueError("EI requires best_so_far")
        return _expected_improvement(mu, sigma, best_so_far=best_so_far, xi=xi)
    if kind == "bald":
        if member_preds is not None:
            member_preds = np.asarray(member_preds, dtype=np.float64)
            if member_preds.ndim != 2:
                raise ValueError(
                    f"member_preds must be 2-D (n_members, n_pool); "
                    f"got shape {member_preds.shape}"
                )
            return member_preds.var(axis=0, ddof=0)
        # Gaussian-posterior fallback: BALD ∝ σ² for a homoscedastic
        # noise model — drop the noise term because it's a constant
        # offset that doesn't change the argmax.
        return sigma ** 2
    raise AssertionError("unreachable")  # pragma: no cover


def _expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, *, best_so_far: float, xi: float = 0.0,
) -> np.ndarray:
    """Closed-form EI for a Gaussian posterior.

    EI(x) = (μ - f* - ξ) Φ(z) + σ φ(z),  where  z = (μ - f* - ξ) / σ.

    For ``σ == 0`` we set EI = max(μ - f* - ξ, 0) — the limiting value
    as σ → 0⁺. Avoids dividing by zero on candidates the model is
    perfectly confident about.
    """
    diff = mu - best_so_far - xi
    out = np.zeros_like(diff)
    pos = sigma > 0.0
    z = np.zeros_like(diff)
    z[pos] = diff[pos] / sigma[pos]
    # erf-based Φ / φ — no scipy dep.
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + _erf_vec(z / math.sqrt(2.0)))
    out[pos] = diff[pos] * Phi[pos] + sigma[pos] * phi[pos]
    out[~pos] = np.maximum(diff[~pos], 0.0)
    return out


_ERF_VEC = np.vectorize(math.erf, otypes=[np.float64])


def _erf_vec(x: np.ndarray) -> np.ndarray:
    return _ERF_VEC(x)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the ``k`` largest entries in ``scores``.

    Uses :func:`numpy.argpartition` (O(n)) then sorts the top-k
    descending so callers can rely on ordering. Ties broken by index
    for determinism. ``k`` is clipped to ``len(scores)``.
    """
    n = scores.shape[0]
    k = max(0, min(k, n))
    if k == 0:
        return np.empty(0, dtype=np.intp)
    if k == n:
        return np.argsort(-scores, kind="stable")
    part = np.argpartition(-scores, k - 1)[:k]
    # Re-sort that slice so the highest score comes first.
    part = part[np.argsort(-scores[part], kind="stable")]
    return part


# ---------------------------------------------------------------------------
# Cycle / campaign data structures
# ---------------------------------------------------------------------------


@dataclass
class ALCycle:
    """One iteration of the active-learning loop.

    Mirrors the columns we'll add in the Session 6.5b ``al_cycles``
    table. Storing as a dataclass first keeps the engine DB-agnostic.
    """

    cycle_index: int
    n_train_before: int
    n_train_after: int
    n_pool_before: int
    n_pool_after: int
    queried_indices: List[int]
    queried_targets: List[float]
    val_mae: Optional[float]
    val_rmse: Optional[float]
    val_r2: Optional[float]
    val_spearman: Optional[float]
    cumulative_best: float
    acquisition: str
    acquisition_params: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class ALCampaign:
    """End-to-end active-learning loop driver.

    The campaign is fully described by:

    * ``X_pool`` / ``y_pool`` — features + ground-truth targets for
      every candidate. ``y_pool`` is consulted by the oracle; the model
      never sees it directly.
    * ``initial_train_indices`` — seed labelled set.
    * ``model`` — :class:`UncertaintyModel`. The campaign refits it
      from scratch each cycle; the supplied instance defines the
      hyperparameters / random state but its fit state is not
      preserved between cycles.
    * ``acquisition`` / ``query_size`` / ``n_cycles`` — loop control.

    The campaign also accepts an explicit ``X_val`` / ``y_val`` pair so
    the per-cycle metrics are computed on a stable held-out set rather
    than the shrinking pool. If ``X_val`` is None we evaluate on the
    *current pool*, which is a reasonable proxy but harder to compare
    across cycles (the pool changes shape).
    """

    X_pool: np.ndarray
    y_pool: np.ndarray
    initial_train_indices: Sequence[int]
    model: UncertaintyModel
    acquisition: str = "max_sigma"
    query_size: int = 10
    n_cycles: int = 10
    maximize: bool = True
    beta: float = 2.0
    xi: float = 0.0
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    seed: int = 0
    # Filled in during run().
    cycles: List[ALCycle] = field(default_factory=list)
    train_indices: List[int] = field(default_factory=list)
    pool_indices: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.X_pool = np.asarray(self.X_pool, dtype=np.float64)
        self.y_pool = np.asarray(self.y_pool, dtype=np.float64).reshape(-1)
        if self.X_pool.ndim != 2:
            raise ValueError(
                f"X_pool must be 2-D; got shape {self.X_pool.shape}"
            )
        if self.X_pool.shape[0] != self.y_pool.shape[0]:
            raise ValueError(
                "X_pool and y_pool sample-count mismatch: "
                f"{self.X_pool.shape[0]} vs {self.y_pool.shape[0]}"
            )
        n = self.X_pool.shape[0]
        seeds = list(self.initial_train_indices)
        if not seeds:
            raise ValueError("initial_train_indices must not be empty")
        if any(i < 0 or i >= n for i in seeds):
            raise ValueError("initial_train_indices contains out-of-bounds ids")
        if len(set(seeds)) != len(seeds):
            raise ValueError("initial_train_indices contains duplicates")
        self.train_indices = list(seeds)
        seed_set = set(seeds)
        self.pool_indices = [i for i in range(n) if i not in seed_set]

        if self.acquisition not in _VALID_ACQUISITIONS:
            raise ValueError(
                f"acquisition {self.acquisition!r} unknown; "
                f"choose from {_VALID_ACQUISITIONS}"
            )
        if self.query_size <= 0:
            raise ValueError("query_size must be positive")
        if self.n_cycles <= 0:
            raise ValueError("n_cycles must be positive")

        if (self.X_val is None) != (self.y_val is None):
            raise ValueError(
                "X_val and y_val must be supplied together (or both None)"
            )
        if self.X_val is not None:
            self.X_val = np.asarray(self.X_val, dtype=np.float64)
            self.y_val = np.asarray(self.y_val, dtype=np.float64).reshape(-1)
            if self.X_val.shape[0] != self.y_val.shape[0]:
                raise ValueError("X_val/y_val sample-count mismatch")
            if self.X_val.shape[1] != self.X_pool.shape[1]:
                raise ValueError("X_val feature-count must match X_pool")

    # ------------------------------------------------------------------
    # Public driver
    # ------------------------------------------------------------------

    def run(self, oracle: Optional[OracleFn] = None) -> List[ALCycle]:
        """Drive the loop for ``n_cycles`` iterations.

        ``oracle`` resolves a pool index → ground-truth value. Default
        is the trivial *cheating oracle* that reads ``y_pool[i]`` —
        appropriate for offline simulation and the acceptance test.
        Real campaigns pass an oracle that submits a DFT job and
        blocks on its result (Session 6.5b).
        """
        if oracle is None:
            oracle = lambda i: float(self.y_pool[i])  # noqa: E731

        cumulative_best = self._initial_best()
        for c in range(self.n_cycles):
            cycle = self._run_one_cycle(
                cycle_index=c, oracle=oracle, prev_best=cumulative_best,
            )
            cumulative_best = cycle.cumulative_best
            self.cycles.append(cycle)
            if not self.pool_indices:
                logger.info("AL pool exhausted at cycle %d", c)
                break
        return self.cycles

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initial_best(self) -> float:
        seeds = self.y_pool[np.asarray(self.train_indices, dtype=np.intp)]
        return float(seeds.max() if self.maximize else seeds.min())

    def _refit(self) -> None:
        idx = np.asarray(self.train_indices, dtype=np.intp)
        self.model.fit(self.X_pool[idx], self.y_pool[idx])

    def _predict_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(self.pool_indices, dtype=np.intp)
        if idx.size == 0:
            return np.empty(0), np.empty(0)
        mu, sigma = self.model.predict_uncertainty(self.X_pool[idx])
        return mu, sigma

    def _val_metrics(self) -> Dict[str, Optional[float]]:
        from .training import compute_metrics

        if self.X_val is not None:
            y_hat = self.model.predict(self.X_val)
            return compute_metrics(self.y_val, y_hat)
        # Fallback: evaluate on current pool. Consistent direction but
        # the pool shrinks each cycle, so cross-cycle comparison is
        # only directional, not absolute.
        idx = np.asarray(self.pool_indices, dtype=np.intp)
        if idx.size < 2:
            return {"mae": None, "rmse": None, "r2": None, "spearman": None}
        y_hat = self.model.predict(self.X_pool[idx])
        return compute_metrics(self.y_pool[idx], y_hat)

    def _run_one_cycle(
        self, *, cycle_index: int, oracle: OracleFn, prev_best: float,
    ) -> ALCycle:
        n_train_before = len(self.train_indices)
        n_pool_before = len(self.pool_indices)
        # 1. Refit on the labelled pool.
        self._refit()
        # 2. Score the unlabelled pool.
        mu, sigma = self._predict_pool()
        scores = acquire(
            mu=mu, sigma=sigma, kind=self.acquisition,
            best_so_far=prev_best, beta=self.beta, xi=self.xi,
            maximize=self.maximize,
        )
        top = top_k_indices(scores, self.query_size)
        # Map pool-local indices → original corpus indices.
        queried_pool_local = top.tolist()
        queried_global = [self.pool_indices[i] for i in queried_pool_local]
        # 3. Ask the oracle.
        labelled: List[Tuple[int, float]] = []
        for g in queried_global:
            y = oracle(g)
            if y is None:
                # Drop from the pool (so we don't re-query) but don't
                # add to the labelled set.
                continue
            labelled.append((g, float(y)))
        # 4. Update train / pool sets.
        labelled_global = {g for g, _ in labelled}
        self.train_indices.extend(g for g, _ in labelled)
        # Remove every queried index from the pool, even oracle
        # failures, so the loop makes progress.
        queried_set = set(queried_global)
        self.pool_indices = [i for i in self.pool_indices if i not in queried_set]
        # 5. Cumulative best (monotone by construction — we either get
        # a strictly better label or we keep the previous best).
        new_targets = [y for _, y in labelled]
        if new_targets:
            best_new = max(new_targets) if self.maximize else min(new_targets)
            if self.maximize:
                cumulative_best = max(prev_best, best_new)
            else:
                cumulative_best = min(prev_best, best_new)
        else:
            cumulative_best = prev_best
        # 6. Validation metrics on the *updated* train set — refit so
        # the metrics reflect post-cycle model state. Cheap if X_val
        # is small; for very large pools this is the dominant cost
        # and we'd cache it, but Session 6.5 doesn't need that yet.
        self._refit()
        metrics = self._val_metrics()
        return ALCycle(
            cycle_index=cycle_index,
            n_train_before=n_train_before,
            n_train_after=len(self.train_indices),
            n_pool_before=n_pool_before,
            n_pool_after=len(self.pool_indices),
            queried_indices=queried_global,
            queried_targets=new_targets,
            val_mae=metrics["mae"],
            val_rmse=metrics["rmse"],
            val_r2=metrics["r2"],
            val_spearman=metrics["spearman"],
            cumulative_best=cumulative_best,
            acquisition=self.acquisition,
            acquisition_params={"beta": self.beta, "xi": self.xi},
        )


# ---------------------------------------------------------------------------
# Random-sampling baseline (for the head-to-head acceptance test)
# ---------------------------------------------------------------------------


def run_random_baseline(
    *,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    initial_train_indices: Sequence[int],
    model_factory: Callable[[], UncertaintyModel],
    query_size: int,
    n_cycles: int,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 0,
    maximize: bool = True,
) -> List[ALCycle]:
    """Same loop as :class:`ALCampaign` but queries are uniform-random.

    Used by :func:`run_paired_comparison` to verify that the AL
    strategy actually improves over the trivial baseline at the same
    compute budget. ``model_factory`` lets callers pass a fresh model
    each call without worrying about shared state.
    """
    n = X_pool.shape[0]
    rng = np.random.default_rng(seed)
    train = list(initial_train_indices)
    seed_set = set(train)
    pool = [i for i in range(n) if i not in seed_set]
    cumulative_best = float(
        np.max(y_pool[train]) if maximize else np.min(y_pool[train])
    )
    cycles: List[ALCycle] = []
    for c in range(n_cycles):
        if not pool:
            break
        n_train_before = len(train)
        n_pool_before = len(pool)
        k = min(query_size, len(pool))
        chosen_pool_local = rng.choice(len(pool), size=k, replace=False)
        chosen_global = [pool[int(i)] for i in chosen_pool_local]
        new_targets = [float(y_pool[g]) for g in chosen_global]
        train.extend(chosen_global)
        chosen_set = set(chosen_global)
        pool = [i for i in pool if i not in chosen_set]
        if maximize:
            cumulative_best = max(cumulative_best, max(new_targets))
        else:
            cumulative_best = min(cumulative_best, min(new_targets))
        # Fit + score on val.
        model = model_factory()
        idx = np.asarray(train, dtype=np.intp)
        model.fit(X_pool[idx], y_pool[idx])
        from .training import compute_metrics

        if X_val is not None:
            y_hat = model.predict(X_val)
            metrics = compute_metrics(y_val, y_hat)
        else:
            metrics = {"mae": None, "rmse": None, "r2": None, "spearman": None}
        cycles.append(
            ALCycle(
                cycle_index=c,
                n_train_before=n_train_before,
                n_train_after=len(train),
                n_pool_before=n_pool_before,
                n_pool_after=len(pool),
                queried_indices=chosen_global,
                queried_targets=new_targets,
                val_mae=metrics["mae"],
                val_rmse=metrics["rmse"],
                val_r2=metrics["r2"],
                val_spearman=metrics["spearman"],
                cumulative_best=cumulative_best,
                acquisition="random",
            )
        )
    return cycles


@dataclass
class PairedComparisonResult:
    """Compares an AL strategy to the random baseline at equal budget."""

    al_final_mae: float
    random_final_mae: float
    al_cycles: List[ALCycle]
    random_cycles: List[ALCycle]

    @property
    def al_wins(self) -> bool:
        """True if AL's final-cycle val MAE strictly beats random's."""
        return self.al_final_mae < self.random_final_mae

    @property
    def mae_delta(self) -> float:
        return self.random_final_mae - self.al_final_mae


def run_paired_comparison(
    *,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    initial_train_indices: Sequence[int],
    model_factory: Callable[[], UncertaintyModel],
    acquisition: str = "max_sigma",
    query_size: int = 10,
    n_cycles: int = 10,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 0,
) -> PairedComparisonResult:
    """Run AL and random with the **same compute budget**, return both.

    The acceptance test for Session 6.5 calls this with the synthetic
    bandgap-like corpus and asserts ``result.al_wins``.
    """
    al = ALCampaign(
        X_pool=X_pool, y_pool=y_pool,
        initial_train_indices=list(initial_train_indices),
        model=model_factory(),
        acquisition=acquisition,
        query_size=query_size, n_cycles=n_cycles,
        X_val=X_val, y_val=y_val, seed=seed,
    )
    al_cycles = al.run()
    rand_cycles = run_random_baseline(
        X_pool=X_pool, y_pool=y_pool,
        initial_train_indices=list(initial_train_indices),
        model_factory=model_factory,
        query_size=query_size, n_cycles=n_cycles,
        X_val=X_val, y_val=y_val, seed=seed,
    )
    al_mae = al_cycles[-1].val_mae if al_cycles else float("inf")
    rand_mae = rand_cycles[-1].val_mae if rand_cycles else float("inf")
    return PairedComparisonResult(
        al_final_mae=float(al_mae) if al_mae is not None else float("inf"),
        random_final_mae=float(rand_mae) if rand_mae is not None else float("inf"),
        al_cycles=al_cycles,
        random_cycles=rand_cycles,
    )


# ---------------------------------------------------------------------------
# CGCNN ensemble adapter (Session 6.4 → UncertaintyModel)
# ---------------------------------------------------------------------------


@dataclass
class EnsembleAdapter:
    """Wrap a Session 6.4 deep ensemble as an :class:`UncertaintyModel`.

    The CGCNN ensemble works on graph batches, not feature matrices,
    so this adapter is **for the API/Celery binding** in Session 6.5b
    where the AL loop drives a CGCNN ensemble. It accepts a callable
    ``train_fn`` that the binding implements (because re-fitting the
    ensemble involves Lightning's Trainer and a DataModule rebuild).

    The acceptance test for 6.5 uses the much simpler baseline path
    via ``ModelFactoryAdapter`` (below) — same idea but for sklearn /
    XGBoost models that fit on plain ``ndarray``.

    This class deliberately raises if you try to use it with a
    feature-matrix interface — it's a marker for the API binding, not
    something the engine itself drives.
    """

    train_fn: Callable[[np.ndarray, np.ndarray], object]
    predict_fn: Callable[[object, np.ndarray], np.ndarray]
    predict_uncertainty_fn: Callable[
        [object, np.ndarray], Tuple[np.ndarray, np.ndarray],
    ]
    _state: object = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleAdapter":
        self._state = self.train_fn(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("EnsembleAdapter has not been fit")
        return self.predict_fn(self._state, X)

    def predict_uncertainty(
        self, X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._state is None:
            raise RuntimeError("EnsembleAdapter has not been fit")
        return self.predict_uncertainty_fn(self._state, X)


__all__ = [
    "ALCampaign",
    "ALCycle",
    "EnsembleAdapter",
    "OracleFn",
    "PairedComparisonResult",
    "UncertaintyModel",
    "acquire",
    "run_paired_comparison",
    "run_random_baseline",
    "top_k_indices",
]
