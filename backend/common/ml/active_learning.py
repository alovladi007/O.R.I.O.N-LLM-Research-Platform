"""
Active Learning Module for Smart Simulation Selection.

Implements uncertainty-aware candidate selection strategies that decide
which candidates should be evaluated with expensive simulations versus
trusted ML predictions.

Session 20: Active Learning for Smart Simulation Selection
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ALCandidate:
    """
    Active learning candidate with uncertainty estimates.

    Extends a structure/material with ML predictions and uncertainty
    to enable smart selection for simulation.
    """
    structure_id: Optional[str] = None
    parameters: Optional[Dict[str, float]] = None

    # ML predictions
    predicted_properties: Dict[str, float] = None
    predicted_uncertainties: Dict[str, float] = None

    # Selection scores
    acquisition_score: Optional[float] = None
    selection_strategy: Optional[str] = None

    # Simulation tracking
    needs_simulation: bool = False
    simulation_status: Optional[str] = None  # "pending", "running", "completed"

    # Metadata
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.predicted_properties is None:
            self.predicted_properties = {}
        if self.predicted_uncertainties is None:
            self.predicted_uncertainties = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ALConfig:
    """
    Active learning configuration.

    Defines thresholds and strategies for deciding when to trust
    ML predictions versus running expensive simulations.
    """
    # Uncertainty thresholds
    uncertainty_threshold: float = 0.1  # If uncertainty > this, run simulation
    high_value_threshold: float = 0.8  # If predicted score > this, always simulate

    # Selection strategy
    selection_strategy: str = "uncertainty"  # "uncertainty", "greedy_uncertainty", "expected_improvement"

    # Budget constraints
    max_simulations_per_iteration: int = 10
    simulation_budget_fraction: float = 0.2  # Fraction of candidates to simulate

    # Confidence thresholds by property
    property_confidence_thresholds: Optional[Dict[str, float]] = None

    # Retraining settings
    min_new_samples_for_retrain: int = 50
    retrain_every_n_iterations: int = 5

    def __post_init__(self):
        if self.property_confidence_thresholds is None:
            self.property_confidence_thresholds = {}


@dataclass
class SimulationBudget:
    """
    Tracks simulation budget and usage.

    Helps manage computational resources by limiting how many
    expensive simulations are run per iteration.
    """
    total_budget: int  # Total simulations allowed
    used: int = 0
    remaining: int = None

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.total_budget

    def consume(self, n: int = 1) -> bool:
        """
        Try to consume n simulations from budget.

        Returns True if successful, False if insufficient budget.
        """
        if self.remaining >= n:
            self.used += n
            self.remaining -= n
            return True
        return False

    @property
    def is_exhausted(self) -> bool:
        """Check if budget is fully consumed."""
        return self.remaining <= 0

    @property
    def usage_fraction(self) -> float:
        """Get fraction of budget used (0-1)."""
        if self.total_budget == 0:
            return 1.0
        return self.used / self.total_budget


# ============================================================================
# Active Learning Selection Strategies
# ============================================================================

def select_candidates_for_simulation(
    candidates: List[ALCandidate],
    config: ALConfig,
    n_select: Optional[int] = None
) -> List[int]:
    """
    Select which candidates should be simulated based on uncertainty.

    Uses active learning strategies to decide which candidates warrant
    expensive simulation versus trusting ML predictions.

    Args:
        candidates: List of candidates with predictions and uncertainties
        config: Active learning configuration
        n_select: Number of candidates to select (defaults to config.max_simulations_per_iteration)

    Returns:
        Indices of candidates to simulate

    Example:
        >>> candidates = [
        ...     ALCandidate(
        ...         structure_id="struct1",
        ...         predicted_properties={"bandgap": 2.1},
        ...         predicted_uncertainties={"bandgap": 0.3}
        ...     ),
        ...     ALCandidate(
        ...         structure_id="struct2",
        ...         predicted_properties={"bandgap": 2.0},
        ...         predicted_uncertainties={"bandgap": 0.05}
        ...     )
        ... ]
        >>> config = ALConfig(selection_strategy="uncertainty", max_simulations_per_iteration=1)
        >>> indices = select_candidates_for_simulation(candidates, config, n_select=1)
        >>> # Returns [0] since struct1 has higher uncertainty
    """
    if n_select is None:
        n_select = min(
            config.max_simulations_per_iteration,
            int(len(candidates) * config.simulation_budget_fraction)
        )

    logger.info(
        f"AL selection: {len(candidates)} candidates, selecting {n_select} for simulation "
        f"using strategy '{config.selection_strategy}'"
    )

    if n_select <= 0 or len(candidates) == 0:
        return []

    # Compute selection scores based on strategy
    strategy = config.selection_strategy

    if strategy == "uncertainty":
        scores = _compute_uncertainty_scores(candidates, config)
    elif strategy == "greedy_uncertainty":
        scores = _compute_greedy_uncertainty_scores(candidates, config)
    elif strategy == "expected_improvement":
        scores = _compute_expected_improvement_scores(candidates, config)
    else:
        logger.warning(f"Unknown strategy '{strategy}', defaulting to uncertainty")
        scores = _compute_uncertainty_scores(candidates, config)

    # Select top N by score
    n_select = min(n_select, len(candidates))
    top_indices = np.argsort(scores)[-n_select:][::-1]  # Descending order

    logger.info(f"Selected {len(top_indices)} candidates for simulation")
    return top_indices.tolist()


def _compute_uncertainty_scores(
    candidates: List[ALCandidate],
    config: ALConfig
) -> np.ndarray:
    """
    Pure uncertainty-based selection.

    Select candidates with highest prediction uncertainty.
    This maximizes information gain but ignores predicted value.
    """
    scores = []

    for cand in candidates:
        # Average uncertainty across all properties
        if cand.predicted_uncertainties:
            uncertainties = list(cand.predicted_uncertainties.values())
            avg_uncertainty = np.mean(uncertainties)
        else:
            avg_uncertainty = 0.0

        scores.append(avg_uncertainty)

    return np.array(scores)


def _compute_greedy_uncertainty_scores(
    candidates: List[ALCandidate],
    config: ALConfig
) -> np.ndarray:
    """
    Greedy uncertainty: combines predicted value and uncertainty.

    Balances exploitation (high predicted value) and exploration (high uncertainty).
    Score = predicted_value * uncertainty
    """
    scores = []

    for cand in candidates:
        # Get average predicted value (assuming higher is better, scaled 0-1)
        if cand.predicted_properties:
            values = list(cand.predicted_properties.values())
            avg_value = np.mean(values)
        else:
            avg_value = 0.0

        # Get average uncertainty
        if cand.predicted_uncertainties:
            uncertainties = list(cand.predicted_uncertainties.values())
            avg_uncertainty = np.mean(uncertainties)
        else:
            avg_uncertainty = 0.0

        # Combine: high value AND high uncertainty
        score = avg_value * avg_uncertainty
        scores.append(score)

    return np.array(scores)


def _compute_expected_improvement_scores(
    candidates: List[ALCandidate],
    config: ALConfig
) -> np.ndarray:
    """
    Expected improvement over current best.

    Estimates the expected improvement considering both predicted value
    and uncertainty (similar to BO acquisition function).
    """
    scores = []

    # Find current best predicted value
    all_values = []
    for cand in candidates:
        if cand.predicted_properties:
            all_values.extend(cand.predicted_properties.values())

    if all_values:
        best_value = max(all_values)
    else:
        best_value = 0.0

    for cand in candidates:
        # Get average predicted value
        if cand.predicted_properties:
            values = list(cand.predicted_properties.values())
            avg_value = np.mean(values)
        else:
            avg_value = 0.0

        # Get average uncertainty (as std)
        if cand.predicted_uncertainties:
            uncertainties = list(cand.predicted_uncertainties.values())
            avg_std = np.mean(uncertainties)
        else:
            avg_std = 1e-6

        # Simple EI approximation: improvement + exploration bonus
        improvement = max(0, avg_value - best_value)
        ei = improvement + avg_std  # Simplified (no normalization)

        scores.append(ei)

    return np.array(scores)


# ============================================================================
# Uncertainty Analysis
# ============================================================================

def classify_by_confidence(
    candidates: List[ALCandidate],
    config: ALConfig
) -> Dict[str, List[int]]:
    """
    Classify candidates into confidence categories.

    Separates candidates into:
    - high_confidence: Trust ML, no simulation needed
    - low_confidence: Uncertain, should simulate
    - high_value_uncertain: High predicted value but uncertain, prioritize simulation

    Args:
        candidates: List of candidates with predictions
        config: Active learning configuration

    Returns:
        Dictionary mapping categories to candidate indices
    """
    categories = {
        "high_confidence": [],
        "low_confidence": [],
        "high_value_uncertain": []
    }

    for i, cand in enumerate(candidates):
        # Compute average uncertainty
        if cand.predicted_uncertainties:
            uncertainties = list(cand.predicted_uncertainties.values())
            avg_uncertainty = np.mean(uncertainties)
        else:
            avg_uncertainty = 0.0

        # Compute average predicted value
        if cand.predicted_properties:
            values = list(cand.predicted_properties.values())
            avg_value = np.mean(values)
        else:
            avg_value = 0.0

        # Classify
        if avg_uncertainty < config.uncertainty_threshold:
            categories["high_confidence"].append(i)
        elif avg_value > config.high_value_threshold:
            categories["high_value_uncertain"].append(i)
        else:
            categories["low_confidence"].append(i)

    logger.info(
        f"Confidence classification: "
        f"{len(categories['high_confidence'])} high confidence, "
        f"{len(categories['low_confidence'])} low confidence, "
        f"{len(categories['high_value_uncertain'])} high value uncertain"
    )

    return categories


def estimate_dataset_quality(
    training_uncertainties: List[float]
) -> Dict[str, Any]:
    """
    Estimate quality of training dataset based on prediction uncertainties.

    Analyzes the distribution of uncertainties to assess whether
    the model needs retraining or more data.

    Args:
        training_uncertainties: Uncertainties from predictions on training set

    Returns:
        Quality metrics and recommendations
    """
    uncertainties = np.array(training_uncertainties)

    metrics = {
        "mean_uncertainty": float(np.mean(uncertainties)),
        "std_uncertainty": float(np.std(uncertainties)),
        "median_uncertainty": float(np.median(uncertainties)),
        "max_uncertainty": float(np.max(uncertainties)),
        "p95_uncertainty": float(np.percentile(uncertainties, 95)),
        "fraction_high_uncertainty": float(np.mean(uncertainties > 0.1)),
    }

    # Recommendations
    if metrics["mean_uncertainty"] > 0.15:
        recommendation = "HIGH_UNCERTAINTY: Model needs retraining with more diverse data"
    elif metrics["mean_uncertainty"] > 0.08:
        recommendation = "MODERATE_UNCERTAINTY: Consider collecting more training data"
    else:
        recommendation = "LOW_UNCERTAINTY: Model predictions are reliable"

    metrics["recommendation"] = recommendation

    logger.info(
        f"Dataset quality: mean_uncertainty={metrics['mean_uncertainty']:.3f}, "
        f"recommendation={recommendation}"
    )

    return metrics


# ============================================================================
# Model Retraining Logic
# ============================================================================

def should_retrain_model(
    iteration_count: int,
    new_samples_since_training: int,
    current_performance: Optional[Dict[str, float]],
    config: ALConfig
) -> Tuple[bool, str]:
    """
    Decide whether ML model should be retrained.

    Checks multiple criteria:
    - Enough new samples accumulated
    - Regular retraining interval reached
    - Performance degradation detected

    Args:
        iteration_count: Current iteration number
        new_samples_since_training: Number of new labeled samples
        current_performance: Current model metrics (e.g., {"mae": 0.15})
        config: Active learning configuration

    Returns:
        (should_retrain: bool, reason: str)
    """
    # Check sample count
    if new_samples_since_training >= config.min_new_samples_for_retrain:
        return True, f"Accumulated {new_samples_since_training} new samples"

    # Check iteration interval
    if config.retrain_every_n_iterations > 0:
        if iteration_count % config.retrain_every_n_iterations == 0:
            return True, f"Regular retraining interval ({config.retrain_every_n_iterations} iterations)"

    # Check performance degradation
    if current_performance is not None:
        # Example: if MAE increases above threshold
        mae = current_performance.get("mae", 0.0)
        if mae > 0.2:  # Threshold
            return True, f"Performance degraded (MAE={mae:.3f})"

    return False, "No retraining needed"


# ============================================================================
# Utility Functions
# ============================================================================

def compute_information_gain(
    before_uncertainties: List[float],
    after_uncertainties: List[float]
) -> Dict[str, float]:
    """
    Compute information gain from running simulations.

    Measures how much uncertainty was reduced by running simulations
    and updating the model.

    Args:
        before_uncertainties: Uncertainties before simulation/retraining
        after_uncertainties: Uncertainties after simulation/retraining

    Returns:
        Information gain metrics
    """
    before = np.array(before_uncertainties)
    after = np.array(after_uncertainties)

    # Compute metrics
    mean_reduction = float(np.mean(before) - np.mean(after))
    max_reduction = float(np.max(before) - np.max(after))

    # Relative reduction (%)
    if np.mean(before) > 0:
        relative_reduction = (mean_reduction / np.mean(before)) * 100
    else:
        relative_reduction = 0.0

    return {
        "mean_uncertainty_before": float(np.mean(before)),
        "mean_uncertainty_after": float(np.mean(after)),
        "mean_reduction": mean_reduction,
        "max_reduction": max_reduction,
        "relative_reduction_percent": relative_reduction
    }


def prioritize_candidates(
    candidates: List[ALCandidate],
    target_property: str,
    budget: SimulationBudget
) -> List[Tuple[int, str]]:
    """
    Prioritize candidates for simulation based on multiple criteria.

    Creates a prioritized list considering:
    - Uncertainty levels
    - Predicted value
    - Available budget

    Args:
        candidates: List of candidates
        target_property: Primary property to optimize
        budget: Simulation budget

    Returns:
        List of (candidate_index, reason) tuples in priority order
    """
    priorities = []

    for i, cand in enumerate(candidates):
        # Get uncertainty and value for target property
        uncertainty = cand.predicted_uncertainties.get(target_property, 0.0)
        value = cand.predicted_properties.get(target_property, 0.0)

        # Compute priority score (higher is better)
        # Prioritize: high value + high uncertainty
        priority_score = value * uncertainty

        # Determine reason
        if uncertainty > 0.2:
            reason = "high_uncertainty"
        elif value > 0.8:
            reason = "high_value"
        elif uncertainty > 0.1 and value > 0.5:
            reason = "promising_uncertain"
        else:
            reason = "exploration"

        priorities.append((i, priority_score, reason))

    # Sort by priority score (descending)
    priorities.sort(key=lambda x: x[1], reverse=True)

    # Return up to budget
    n_select = min(len(priorities), budget.remaining)
    result = [(idx, reason) for idx, score, reason in priorities[:n_select]]

    logger.info(f"Prioritized {len(result)} candidates for simulation (budget: {budget.remaining})")

    return result


def create_training_batch(
    simulated_candidates: List[Dict[str, Any]],
    format: str = "dict"
) -> Any:
    """
    Create a training batch from newly simulated candidates.

    Converts simulation results into format suitable for model retraining.

    Args:
        simulated_candidates: List of candidates with simulation results
            Each should have: structure, properties (ground truth)
        format: Output format ("dict", "arrays", "dataset")

    Returns:
        Training data in requested format
    """
    if format == "dict":
        # Return as list of dicts
        return [
            {
                "structure_id": cand.get("structure_id"),
                "features": cand.get("features"),
                "properties": cand.get("properties"),  # Ground truth
                "metadata": {
                    "source": "active_learning",
                    "iteration": cand.get("iteration", 0)
                }
            }
            for cand in simulated_candidates
        ]

    elif format == "arrays":
        # Return as numpy arrays (X, y)
        # This is a placeholder - actual implementation depends on model format
        logger.warning("Array format not fully implemented, returning dict")
        return create_training_batch(simulated_candidates, format="dict")

    else:
        logger.warning(f"Unknown format '{format}', defaulting to dict")
        return create_training_batch(simulated_candidates, format="dict")
