"""
ML vs Simulation comparison utilities.

This module provides utilities for comparing ML-predicted properties with
simulation results to evaluate model accuracy and reliability.

Comparison metrics include:
- Absolute error: |predicted - simulated|
- Percent error: ((predicted - simulated) / simulated) * 100
- Relative error: |predicted - simulated| / |simulated|
- Mean Absolute Error (MAE) across multiple properties
- Root Mean Squared Error (RMSE)

These comparisons are useful for:
- Model validation and benchmarking
- Identifying when ML predictions are unreliable
- Selecting which properties to trust from ML
- Deciding when expensive simulations are necessary
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import math

logger = logging.getLogger(__name__)


def calculate_error_metrics(
    predicted: float,
    simulated: float,
) -> Dict[str, float]:
    """
    Calculate error metrics between predicted and simulated values.

    Args:
        predicted: ML-predicted value
        simulated: Simulation result value

    Returns:
        Dictionary with error metrics:
        - error: Absolute difference (predicted - simulated)
        - percent_error: Percentage error
        - relative_error: Relative error normalized by simulated value
        - squared_error: Squared error for RMSE calculation
    """
    error = predicted - simulated
    abs_error = abs(error)

    # Percent error: ((predicted - simulated) / simulated) * 100
    # Handle division by zero for simulated values close to zero
    if abs(simulated) > 1e-10:
        percent_error = (error / simulated) * 100
        relative_error = abs_error / abs(simulated)
    else:
        # For values very close to zero, percent error is not meaningful
        percent_error = None
        relative_error = None

    squared_error = error ** 2

    return {
        "error": round(error, 6),
        "percent_error": round(percent_error, 3) if percent_error is not None else None,
        "relative_error": round(relative_error, 6) if relative_error is not None else None,
        "squared_error": round(squared_error, 6),
    }


def compare_ml_vs_simulation(
    predicted_properties: Any,  # PredictedProperties model
    simulation_result: Any,  # SimulationResult model
) -> Dict[str, Any]:
    """
    Compare ML predictions with simulation results.

    This function compares predicted properties against simulation results
    for the same structure, calculating error metrics for each property.

    The comparison is property-by-property:
    - Only properties present in both datasets are compared
    - Properties unique to one dataset are noted but not compared
    - Error metrics are calculated for each comparable property

    Args:
        predicted_properties: PredictedProperties model instance
        simulation_result: SimulationResult model instance

    Returns:
        Dictionary with comparison results:
        {
            "structure_id": UUID,
            "model_name": str,
            "model_version": str,
            "predicted_properties_id": UUID,
            "simulation_result_id": UUID,
            "comparisons": {
                "property_name": {
                    "predicted": float,
                    "simulated": float,
                    "error": float,
                    "percent_error": float,
                    "relative_error": float
                },
                ...
            },
            "summary": {
                "properties_compared": int,
                "mean_absolute_error": float,
                "root_mean_squared_error": float,
                "mean_percent_error": float,
                "properties_only_predicted": list,
                "properties_only_simulated": list,
            },
            "created_at": datetime
        }

    Example:
        >>> predicted = db.get(PredictedProperties, pred_id)
        >>> simulated = db.get(SimulationResult, sim_id)
        >>> comparison = compare_ml_vs_simulation(predicted, simulated)
        >>> print(f"Bandgap error: {comparison['comparisons']['bandgap']['error']} eV")
        Bandgap error: -0.109 eV
    """
    logger.info(
        f"Comparing ML predictions {predicted_properties.id} with "
        f"simulation result {simulation_result.id}"
    )

    # Verify both are for the same structure
    # (Assuming simulation_result has a structure_id through simulation_job)
    if hasattr(simulation_result, 'simulation_job'):
        sim_structure_id = simulation_result.simulation_job.structure_id
        if sim_structure_id != predicted_properties.structure_id:
            logger.warning(
                f"Structure mismatch: predicted={predicted_properties.structure_id}, "
                f"simulated={sim_structure_id}"
            )
            # Continue anyway but note the mismatch

    # Get predicted properties
    predicted_props = predicted_properties.properties

    # Get simulated properties from summary
    simulated_props = simulation_result.summary

    # Map common property names
    # (simulation results might use different names)
    property_name_map = {
        "bandgap": ["bandgap", "band_gap", "gap"],
        "formation_energy": ["formation_energy", "energy", "formation_e"],
        "stability_score": ["stability_score", "stability"],
    }

    # Find comparable properties
    comparisons = {}
    errors = []  # For calculating MAE and RMSE

    for pred_name, pred_value in predicted_props.items():
        if not isinstance(pred_value, (int, float)):
            continue  # Skip non-numeric properties

        # Try to find matching property in simulation results
        sim_value = None
        sim_name = None

        # First try exact match
        if pred_name in simulated_props:
            sim_value = simulated_props[pred_name]
            sim_name = pred_name
        else:
            # Try mapped names
            possible_names = property_name_map.get(pred_name, [pred_name])
            for possible_name in possible_names:
                if possible_name in simulated_props:
                    sim_value = simulated_props[possible_name]
                    sim_name = possible_name
                    break

        if sim_value is not None and isinstance(sim_value, (int, float)):
            # Calculate error metrics
            metrics = calculate_error_metrics(
                predicted=pred_value,
                simulated=sim_value
            )

            comparisons[pred_name] = {
                "predicted": round(pred_value, 6),
                "simulated": round(sim_value, 6),
                "error": metrics["error"],
                "percent_error": metrics["percent_error"],
                "relative_error": metrics["relative_error"],
            }

            errors.append({
                "abs_error": abs(metrics["error"]),
                "squared_error": metrics["squared_error"],
                "percent_error": abs(metrics["percent_error"]) if metrics["percent_error"] is not None else None,
            })

            logger.debug(
                f"Compared {pred_name}: predicted={pred_value}, "
                f"simulated={sim_value}, error={metrics['error']}"
            )

    # Calculate summary statistics
    properties_compared = len(comparisons)

    if errors:
        mean_absolute_error = sum(e["abs_error"] for e in errors) / len(errors)
        mean_squared_error = sum(e["squared_error"] for e in errors) / len(errors)
        root_mean_squared_error = math.sqrt(mean_squared_error)

        # Mean percent error (only for properties where it's defined)
        percent_errors = [e["percent_error"] for e in errors if e["percent_error"] is not None]
        mean_percent_error = sum(percent_errors) / len(percent_errors) if percent_errors else None
    else:
        mean_absolute_error = None
        root_mean_squared_error = None
        mean_percent_error = None

    # Find properties only in one dataset
    predicted_property_names = set(
        k for k, v in predicted_props.items()
        if isinstance(v, (int, float))
    )
    simulated_property_names = set(
        k for k, v in simulated_props.items()
        if isinstance(v, (int, float))
    )

    properties_only_predicted = list(predicted_property_names - simulated_property_names)
    properties_only_simulated = list(simulated_property_names - predicted_property_names)

    # Build result
    result = {
        "structure_id": str(predicted_properties.structure_id),
        "model_name": predicted_properties.model_name,
        "model_version": predicted_properties.model_version,
        "predicted_properties_id": str(predicted_properties.id),
        "simulation_result_id": str(simulation_result.id),
        "comparisons": comparisons,
        "summary": {
            "properties_compared": properties_compared,
            "mean_absolute_error": round(mean_absolute_error, 6) if mean_absolute_error else None,
            "root_mean_squared_error": round(root_mean_squared_error, 6) if root_mean_squared_error else None,
            "mean_percent_error": round(mean_percent_error, 3) if mean_percent_error else None,
            "properties_only_predicted": properties_only_predicted,
            "properties_only_simulated": properties_only_simulated,
        },
        "created_at": datetime.utcnow(),
    }

    logger.info(
        f"Comparison complete: {properties_compared} properties compared, "
        f"MAE={mean_absolute_error:.6f}" if mean_absolute_error else "no comparable properties"
    )

    return result


def evaluate_prediction_quality(
    comparison_result: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the quality of ML predictions based on comparison results.

    Assigns quality ratings (excellent, good, fair, poor) based on
    error thresholds for each property.

    Args:
        comparison_result: Output from compare_ml_vs_simulation()
        thresholds: Optional custom error thresholds per property
                   Default thresholds:
                   - bandgap: <0.1 eV = excellent, <0.3 eV = good, <0.5 eV = fair
                   - formation_energy: <0.1 eV/atom = excellent, <0.3 = good, <0.5 = fair

    Returns:
        Dictionary with quality assessments:
        {
            "overall_quality": str,  # excellent/good/fair/poor
            "property_quality": {
                "property_name": {
                    "quality": str,
                    "error": float,
                    "threshold_exceeded": bool
                }
            },
            "recommendations": list  # List of recommendations
        }
    """
    if thresholds is None:
        # Default thresholds (absolute error)
        thresholds = {
            "bandgap": {"excellent": 0.1, "good": 0.3, "fair": 0.5},
            "formation_energy": {"excellent": 0.1, "good": 0.3, "fair": 0.5},
            "stability_score": {"excellent": 0.05, "good": 0.1, "fair": 0.2},
        }

    comparisons = comparison_result.get("comparisons", {})
    property_quality = {}
    quality_scores = []

    for prop_name, comparison in comparisons.items():
        abs_error = abs(comparison.get("error", 0))

        # Get thresholds for this property
        prop_thresholds = thresholds.get(
            prop_name,
            {"excellent": 0.1, "good": 0.3, "fair": 0.5}  # Default
        )

        # Determine quality
        if abs_error <= prop_thresholds["excellent"]:
            quality = "excellent"
            score = 4
        elif abs_error <= prop_thresholds["good"]:
            quality = "good"
            score = 3
        elif abs_error <= prop_thresholds["fair"]:
            quality = "fair"
            score = 2
        else:
            quality = "poor"
            score = 1

        property_quality[prop_name] = {
            "quality": quality,
            "error": abs_error,
            "threshold_exceeded": quality == "poor",
        }
        quality_scores.append(score)

    # Overall quality (average of all properties)
    if quality_scores:
        avg_score = sum(quality_scores) / len(quality_scores)
        if avg_score >= 3.5:
            overall_quality = "excellent"
        elif avg_score >= 2.5:
            overall_quality = "good"
        elif avg_score >= 1.5:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
    else:
        overall_quality = "unknown"

    # Generate recommendations
    recommendations = []
    if overall_quality in ["poor", "fair"]:
        recommendations.append(
            "ML predictions show significant errors. Consider using simulation results instead."
        )
    if overall_quality == "poor":
        recommendations.append(
            "High prediction error detected. The ML model may need retraining or "
            "this structure may be outside the model's training distribution."
        )

    poor_properties = [
        name for name, qual in property_quality.items()
        if qual["quality"] == "poor"
    ]
    if poor_properties:
        recommendations.append(
            f"Properties with poor predictions: {', '.join(poor_properties)}. "
            "Use simulation values for these."
        )

    return {
        "overall_quality": overall_quality,
        "property_quality": property_quality,
        "recommendations": recommendations,
    }


def batch_compare(
    predictions: List[Any],
    simulation_results: List[Any],
) -> List[Dict[str, Any]]:
    """
    Compare multiple ML predictions with their corresponding simulation results.

    Args:
        predictions: List of PredictedProperties instances
        simulation_results: List of SimulationResult instances

    Returns:
        List of comparison results (one per prediction-simulation pair)
    """
    logger.info(f"Batch comparing {len(predictions)} predictions")

    results = []
    for pred, sim in zip(predictions, simulation_results):
        try:
            comparison = compare_ml_vs_simulation(pred, sim)
            results.append(comparison)
        except Exception as e:
            logger.error(f"Error comparing prediction {pred.id}: {e}")
            results.append({
                "error": str(e),
                "predicted_properties_id": str(pred.id),
                "simulation_result_id": str(sim.id),
            })

    return results
