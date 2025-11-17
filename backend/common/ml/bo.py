"""
Bayesian Optimization Module for Materials Design.

Provides Gaussian Process-based optimization for intelligent materials discovery
within the DesignCampaign framework.

Session 19: Bayesian Optimization for Materials Design
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Try to import scipy for optimization
try:
    from scipy.stats import norm
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, BO will use stub implementation")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Candidate:
    """
    Candidate structure for optimization.

    Represents a point in the design space with optional prediction
    and uncertainty estimates.
    """
    structure_id: Optional[str] = None
    parameters: Dict[str, float] = None  # Continuous design parameters
    predicted_score: Optional[float] = None
    predicted_uncertainty: Optional[float] = None
    acquisition_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BOConfig:
    """
    Bayesian optimization configuration.

    Defines the target property, optimization settings, and
    acquisition function parameters.
    """
    target_property: str  # "bandgap", "formation_energy", etc.
    target_range: Tuple[float, float]  # Desired range (min, max)
    acquisition_function: str = "ei"  # "ei", "ucb", "poi"
    n_initial_random: int = 10  # Random samples before BO
    n_iterations: int = 50  # Max BO iterations
    batch_size: int = 5  # Candidates per iteration

    # Acquisition function parameters
    ucb_kappa: float = 2.0  # Exploration parameter for UCB
    ei_xi: float = 0.01  # Exploration parameter for EI

    # GP hyperparameters
    length_scale: float = 1.0
    noise_level: float = 0.1


@dataclass
class ParameterSpace:
    """
    Design parameter space definition.

    Defines the bounds and types of parameters to optimize.
    """
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {min, max, type}

    def sample_random(self, n_samples: int = 1) -> List[Dict[str, float]]:
        """Sample random points from the parameter space."""
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name, spec in self.parameters.items():
                param_type = spec.get("type", "continuous")
                if param_type == "continuous":
                    min_val = spec["min"]
                    max_val = spec["max"]
                    sample[name] = np.random.uniform(min_val, max_val)
                elif param_type == "integer":
                    min_val = int(spec["min"])
                    max_val = int(spec["max"])
                    sample[name] = float(np.random.randint(min_val, max_val + 1))
            samples.append(sample)
        return samples

    def clip_to_bounds(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Clip parameters to valid bounds."""
        clipped = {}
        for name, value in parameters.items():
            if name in self.parameters:
                spec = self.parameters[name]
                clipped[name] = np.clip(value, spec["min"], spec["max"])
            else:
                clipped[name] = value
        return clipped


# ============================================================================
# Gaussian Process (Simplified)
# ============================================================================

class SimpleGP:
    """
    Simplified Gaussian Process for surrogate modeling.

    This is a basic implementation. In production, use GPyTorch or scikit-learn.
    """

    def __init__(self, length_scale: float = 1.0, noise_level: float = 0.1):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel."""
        # Compute pairwise squared distances
        sq_dists = np.sum(X1**2, 1).reshape(-1, 1) + \
                   np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 * sq_dists / self.length_scale**2)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP to training data.

        Args:
            X: Training inputs (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        self.X_train = X
        self.y_train = y

        # Compute kernel matrix with noise
        K = self._rbf_kernel(X, X)
        K += self.noise_level**2 * np.eye(len(X))

        # Compute inverse (should use Cholesky in production)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Add jitter if singular
            K += 1e-6 * np.eye(len(X))
            self.K_inv = np.linalg.inv(K)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation.

        Args:
            X: Test inputs (n_samples, n_features)

        Returns:
            mean: Predicted means (n_samples,)
            std: Predicted standard deviations (n_samples,)
        """
        if self.X_train is None:
            # No training data, return prior
            mean = np.zeros(len(X))
            std = np.ones(len(X))
            return mean, std

        # Compute cross-kernel
        K_star = self._rbf_kernel(self.X_train, X)

        # Mean prediction
        mean = K_star.T @ self.K_inv @ self.y_train

        # Variance prediction
        K_star_star = self._rbf_kernel(X, X)
        var = np.diag(K_star_star) - np.sum(K_star.T @ self.K_inv * K_star.T, axis=1)
        std = np.sqrt(np.maximum(var, 1e-10))  # Ensure positive

        return mean, std


# ============================================================================
# Acquisition Functions
# ============================================================================

def expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_y: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    Args:
        mean: Predicted means
        std: Predicted standard deviations
        best_y: Current best observed value
        xi: Exploration parameter

    Returns:
        ei: Expected improvement values
    """
    if not SCIPY_AVAILABLE:
        # Fallback: use uncertainty-based acquisition
        return std

    improvement = mean - best_y - xi
    Z = improvement / (std + 1e-9)

    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std == 0.0] = 0.0

    return ei


def upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 2.0
) -> np.ndarray:
    """
    Upper Confidence Bound acquisition function.

    Args:
        mean: Predicted means
        std: Predicted standard deviations
        kappa: Exploration parameter (higher = more exploration)

    Returns:
        ucb: UCB values
    """
    return mean + kappa * std


def probability_of_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_y: float,
    xi: float = 0.01
) -> np.ndarray:
    """
    Probability of Improvement acquisition function.

    Args:
        mean: Predicted means
        std: Predicted standard deviations
        best_y: Current best observed value
        xi: Exploration parameter

    Returns:
        poi: Probability of improvement values
    """
    if not SCIPY_AVAILABLE:
        # Fallback: use uncertainty-based acquisition
        return std

    Z = (mean - best_y - xi) / (std + 1e-9)
    return norm.cdf(Z)


# ============================================================================
# Main BO Functions
# ============================================================================

def suggest_candidates(
    target_config: BOConfig,
    existing_data: List[Dict[str, Any]],
    parameter_space: ParameterSpace,
    n_suggestions: int = 5
) -> List[Candidate]:
    """
    Suggest new candidates using Bayesian optimization.

    Uses Gaussian Process surrogate model with chosen acquisition function
    to suggest promising candidates.

    Args:
        target_config: BO configuration
        existing_data: List of evaluated candidates with:
            - parameters: Dict[str, float]
            - value: float (measured property value)
        parameter_space: Parameter space definition
        n_suggestions: Number of candidates to suggest

    Returns:
        List of Candidate objects with suggested parameters

    Example:
        >>> config = BOConfig(target_property="bandgap", target_range=(2.0, 3.0))
        >>> param_space = ParameterSpace(parameters={
        ...     "doping_x": {"min": 0.0, "max": 1.0, "type": "continuous"}
        ... })
        >>> existing = [
        ...     {"parameters": {"doping_x": 0.0}, "value": 1.5},
        ...     {"parameters": {"doping_x": 0.5}, "value": 2.2},
        ... ]
        >>> candidates = suggest_candidates(config, existing, param_space, n_suggestions=3)
    """
    logger.info(
        f"BO suggestion: {len(existing_data)} existing points, "
        f"suggesting {n_suggestions} new candidates"
    )

    # Check if we have enough data to fit GP
    if len(existing_data) < target_config.n_initial_random:
        logger.info("Not enough data for BO, using random sampling")
        return _suggest_random(parameter_space, n_suggestions)

    # Extract training data
    X_train, y_train = _extract_training_data(existing_data, parameter_space)

    if len(X_train) == 0:
        logger.warning("No valid training data, using random sampling")
        return _suggest_random(parameter_space, n_suggestions)

    # Fit Gaussian Process
    gp = SimpleGP(
        length_scale=target_config.length_scale,
        noise_level=target_config.noise_level
    )
    gp.fit(X_train, y_train)

    # Optimize acquisition function
    best_y = np.max(y_train)
    candidates = []

    for i in range(n_suggestions):
        # Optimize acquisition function to find next candidate
        best_params = _optimize_acquisition(
            gp, best_y, parameter_space, target_config
        )

        # Predict at this point
        X_new = _params_to_array(best_params, parameter_space)
        mean, std = gp.predict(X_new.reshape(1, -1))

        # Compute acquisition score
        if target_config.acquisition_function == "ei":
            acq = expected_improvement(mean, std, best_y, target_config.ei_xi)[0]
        elif target_config.acquisition_function == "ucb":
            acq = upper_confidence_bound(mean, std, target_config.ucb_kappa)[0]
        elif target_config.acquisition_function == "poi":
            acq = probability_of_improvement(mean, std, best_y, target_config.ei_xi)[0]
        else:
            acq = std[0]  # Default to uncertainty

        candidate = Candidate(
            parameters=best_params,
            predicted_score=float(mean[0]),
            predicted_uncertainty=float(std[0]),
            acquisition_score=float(acq),
            metadata={
                "iteration": i,
                "acquisition_function": target_config.acquisition_function
            }
        )
        candidates.append(candidate)

        # Add to "training data" for next iteration (pseudo-update)
        # This prevents suggesting the same point multiple times
        X_train = np.vstack([X_train, X_new])
        y_train = np.append(y_train, mean[0])
        gp.fit(X_train, y_train)

    logger.info(f"Generated {len(candidates)} BO candidates")
    return candidates


def _suggest_random(
    parameter_space: ParameterSpace,
    n_suggestions: int
) -> List[Candidate]:
    """Suggest random candidates from parameter space."""
    samples = parameter_space.sample_random(n_suggestions)

    candidates = []
    for i, params in enumerate(samples):
        candidate = Candidate(
            parameters=params,
            metadata={"strategy": "random", "iteration": i}
        )
        candidates.append(candidate)

    return candidates


def _extract_training_data(
    data: List[Dict[str, Any]],
    parameter_space: ParameterSpace
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract training data for GP."""
    param_names = sorted(parameter_space.parameters.keys())

    X_list = []
    y_list = []

    for point in data:
        if "parameters" not in point or "value" not in point:
            continue

        params = point["parameters"]

        # Convert parameters to array
        x = [params.get(name, 0.0) for name in param_names]
        X_list.append(x)
        y_list.append(point["value"])

    if len(X_list) == 0:
        return np.array([]), np.array([])

    return np.array(X_list), np.array(y_list)


def _params_to_array(
    params: Dict[str, float],
    parameter_space: ParameterSpace
) -> np.ndarray:
    """Convert parameter dict to array."""
    param_names = sorted(parameter_space.parameters.keys())
    return np.array([params.get(name, 0.0) for name in param_names])


def _array_to_params(
    x: np.ndarray,
    parameter_space: ParameterSpace
) -> Dict[str, float]:
    """Convert array to parameter dict."""
    param_names = sorted(parameter_space.parameters.keys())
    params = {name: float(val) for name, val in zip(param_names, x)}
    return parameter_space.clip_to_bounds(params)


def _optimize_acquisition(
    gp: SimpleGP,
    best_y: float,
    parameter_space: ParameterSpace,
    config: BOConfig
) -> Dict[str, float]:
    """
    Optimize acquisition function to find next candidate.

    Uses random search or scipy.optimize if available.
    """
    param_names = sorted(parameter_space.parameters.keys())
    bounds = [(parameter_space.parameters[name]["min"],
               parameter_space.parameters[name]["max"])
              for name in param_names]

    # Acquisition function to minimize (negative)
    def neg_acquisition(x):
        mean, std = gp.predict(x.reshape(1, -1))

        if config.acquisition_function == "ei":
            acq = expected_improvement(mean, std, best_y, config.ei_xi)[0]
        elif config.acquisition_function == "ucb":
            acq = upper_confidence_bound(mean, std, config.ucb_kappa)[0]
        elif config.acquisition_function == "poi":
            acq = probability_of_improvement(mean, std, best_y, config.ei_xi)[0]
        else:
            acq = std[0]

        return -acq  # Minimize negative = maximize

    # Try scipy optimization if available
    if SCIPY_AVAILABLE:
        # Multi-start optimization
        best_acq = float('inf')
        best_x = None

        for _ in range(10):  # 10 random starts
            x0 = parameter_space.sample_random(1)[0]
            x0_array = _params_to_array(x0, parameter_space)

            result = minimize(
                neg_acquisition,
                x0_array,
                bounds=bounds,
                method='L-BFGS-B'
            )

            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x

        return _array_to_params(best_x, parameter_space)

    else:
        # Fallback: random search
        n_random = 1000
        random_samples = parameter_space.sample_random(n_random)

        best_acq = float('inf')
        best_params = None

        for params in random_samples:
            x = _params_to_array(params, parameter_space)
            acq = neg_acquisition(x)

            if acq < best_acq:
                best_acq = acq
                best_params = params

        return best_params


# ============================================================================
# Utility Functions
# ============================================================================

def compute_pareto_front(
    candidates: List[Dict[str, Any]],
    objectives: List[str]
) -> List[int]:
    """
    Compute Pareto front for multi-objective optimization.

    Args:
        candidates: List of evaluated candidates with objective values
        objectives: List of objective names

    Returns:
        Indices of candidates on the Pareto front
    """
    n = len(candidates)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue

        for j in range(n):
            if i == j or dominated[j]:
                continue

            # Check if j dominates i
            dominates = True
            strictly_better = False

            for obj in objectives:
                val_i = candidates[i].get(obj, 0.0)
                val_j = candidates[j].get(obj, 0.0)

                if val_j < val_i:  # Assuming minimization
                    dominates = False
                    break
                if val_j > val_i:
                    strictly_better = True

            if dominates and strictly_better:
                dominated[i] = True
                break

    return [i for i in range(n) if not dominated[i]]
