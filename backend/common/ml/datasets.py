"""
Dataset Builder for GNN Training
=================================

This module provides functions to construct training datasets from:
- StructureFeatures (graph and scalar features)
- SimulationResult (target properties like energies, bandgaps)

The datasets are used to train GNN models for property prediction.

Session 14: ML Infrastructure
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset construction."""
    target_property: str  # "bandgap", "formation_energy", etc.
    min_samples: int = 10  # Minimum samples required
    train_fraction: float = 0.8  # Fraction for training (rest for validation)
    seed: int = 42  # Random seed for reproducibility
    max_samples: Optional[int] = None  # Maximum samples (for memory limits)
    filter_outliers: bool = True  # Remove outliers based on IQR
    outlier_iqr_factor: float = 3.0  # IQR multiplier for outlier detection


@dataclass
class Dataset:
    """Container for a constructed dataset."""
    # Features
    structure_ids: List[str]
    graph_features: List[Dict[str, Any]]
    scalar_features: List[Dict[str, Any]]

    # Targets
    targets: np.ndarray

    # Metadata
    target_property: str
    num_samples: int
    feature_version: str

    # Statistics
    target_mean: float
    target_std: float
    target_min: float
    target_max: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "structure_ids": self.structure_ids,
            "graph_features": self.graph_features,
            "scalar_features": self.scalar_features,
            "targets": self.targets.tolist(),
            "target_property": self.target_property,
            "num_samples": self.num_samples,
            "feature_version": self.feature_version,
            "statistics": {
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "target_min": self.target_min,
                "target_max": self.target_max,
            }
        }


@dataclass
class TrainValSplit:
    """Training and validation split."""
    train: Dataset
    val: Dataset
    config: DatasetConfig


def filter_outliers(
    targets: np.ndarray,
    indices: np.ndarray,
    iqr_factor: float = 3.0
) -> np.ndarray:
    """
    Filter outliers using IQR method.

    Args:
        targets: Target values
        indices: Indices to filter
        iqr_factor: IQR multiplier for outlier detection

    Returns:
        Filtered indices
    """
    q1 = np.percentile(targets, 25)
    q3 = np.percentile(targets, 75)
    iqr = q3 - q1

    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr

    mask = (targets >= lower_bound) & (targets <= upper_bound)
    filtered_indices = indices[mask]

    num_filtered = len(indices) - len(filtered_indices)
    if num_filtered > 0:
        logger.info(f"Filtered {num_filtered} outliers ({num_filtered/len(indices)*100:.1f}%)")

    return filtered_indices


async def build_regression_dataset(
    db_session: Any,
    config: DatasetConfig
) -> Dataset:
    """
    Build a regression dataset for property prediction.

    This function:
    1. Queries StructureFeatures and SimulationResults from database
    2. Filters structures that have both features and target property
    3. Optionally removes outliers
    4. Returns a Dataset object

    Args:
        db_session: AsyncSession for database queries
        config: Dataset configuration

    Returns:
        Dataset object with features and targets

    Example:
        >>> config = DatasetConfig(target_property="bandgap", min_samples=100)
        >>> dataset = await build_regression_dataset(db, config)
        >>> print(f"Built dataset with {dataset.num_samples} samples")
    """
    from sqlalchemy import select
    from src.api.models.structure_features import StructureFeatures
    from src.api.models.simulation import SimulationResult

    logger.info(f"Building dataset for target: {config.target_property}")

    # Query structures with features AND simulation results containing target
    # This is a simplified version - in production, you'd join across multiple tables
    query = select(StructureFeatures).limit(config.max_samples if config.max_samples else 10000)
    result = await db_session.execute(query)
    structure_features_list = result.scalars().all()

    logger.info(f"Found {len(structure_features_list)} structures with features")

    # Extract features and targets
    structure_ids = []
    graph_features = []
    scalar_features = []
    targets = []

    for sf in structure_features_list:
        # Check if we have a simulation result for this structure
        # (Simplified - in reality, you'd query SimulationResult table)
        # For now, we'll use mock targets based on scalar features

        # Mock target extraction (replace with actual query in production)
        target_value = extract_mock_target(sf.scalar_features, config.target_property)

        if target_value is not None:
            structure_ids.append(str(sf.structure_id))
            graph_features.append(sf.graph_repr)
            scalar_features.append(sf.scalar_features)
            targets.append(target_value)

    if len(targets) < config.min_samples:
        logger.warning(
            f"Insufficient samples: {len(targets)} < {config.min_samples} required"
        )
        raise ValueError(
            f"Dataset has {len(targets)} samples, but {config.min_samples} required"
        )

    targets_array = np.array(targets, dtype=np.float32)

    # Filter outliers if requested
    if config.filter_outliers:
        indices = np.arange(len(targets))
        filtered_indices = filter_outliers(
            targets_array,
            indices,
            iqr_factor=config.outlier_iqr_factor
        )

        structure_ids = [structure_ids[i] for i in filtered_indices]
        graph_features = [graph_features[i] for i in filtered_indices]
        scalar_features = [scalar_features[i] for i in filtered_indices]
        targets_array = targets_array[filtered_indices]

    # Compute statistics
    target_mean = float(np.mean(targets_array))
    target_std = float(np.std(targets_array))
    target_min = float(np.min(targets_array))
    target_max = float(np.max(targets_array))

    logger.info(f"Dataset statistics:")
    logger.info(f"  Samples: {len(targets_array)}")
    logger.info(f"  Target: {config.target_property}")
    logger.info(f"  Mean: {target_mean:.3f}")
    logger.info(f"  Std: {target_std:.3f}")
    logger.info(f"  Range: [{target_min:.3f}, {target_max:.3f}]")

    dataset = Dataset(
        structure_ids=structure_ids,
        graph_features=graph_features,
        scalar_features=scalar_features,
        targets=targets_array,
        target_property=config.target_property,
        num_samples=len(targets_array),
        feature_version="1.0.0",  # From StructureFeatures
        target_mean=target_mean,
        target_std=target_std,
        target_min=target_min,
        target_max=target_max
    )

    return dataset


async def split_train_val(
    dataset: Dataset,
    config: DatasetConfig
) -> TrainValSplit:
    """
    Split dataset into training and validation sets.

    Args:
        dataset: Full dataset
        config: Dataset configuration (contains train_fraction and seed)

    Returns:
        TrainValSplit with train and val datasets
    """
    num_samples = dataset.num_samples
    num_train = int(num_samples * config.train_fraction)

    # Shuffle indices
    np.random.seed(config.seed)
    indices = np.random.permutation(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    logger.info(f"Split: {len(train_indices)} train, {len(val_indices)} val")

    # Create train dataset
    train_dataset = Dataset(
        structure_ids=[dataset.structure_ids[i] for i in train_indices],
        graph_features=[dataset.graph_features[i] for i in train_indices],
        scalar_features=[dataset.scalar_features[i] for i in train_indices],
        targets=dataset.targets[train_indices],
        target_property=dataset.target_property,
        num_samples=len(train_indices),
        feature_version=dataset.feature_version,
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
        target_min=dataset.target_min,
        target_max=dataset.target_max
    )

    # Create val dataset
    val_dataset = Dataset(
        structure_ids=[dataset.structure_ids[i] for i in val_indices],
        graph_features=[dataset.graph_features[i] for i in val_indices],
        scalar_features=[dataset.scalar_features[i] for i in val_indices],
        targets=dataset.targets[val_indices],
        target_property=dataset.target_property,
        num_samples=len(val_indices),
        feature_version=dataset.feature_version,
        target_mean=dataset.target_mean,
        target_std=dataset.target_std,
        target_min=dataset.target_min,
        target_max=dataset.target_max
    )

    return TrainValSplit(train=train_dataset, val=val_dataset, config=config)


def extract_mock_target(scalar_features: Dict[str, Any], target_property: str) -> Optional[float]:
    """
    Extract mock target value from scalar features.

    In production, this would query SimulationResult table.
    For now, we use a deterministic function of scalar features.

    Args:
        scalar_features: Scalar features dictionary
        target_property: Target property name

    Returns:
        Mock target value or None
    """
    # Use avg_electronegativity and density to compute mock targets
    avg_electroneg = scalar_features.get("avg_electronegativity", 2.0)
    density = scalar_features.get("density", 3.0)
    num_elements = scalar_features.get("num_elements", 2)

    # Mock bandgap (0-5 eV)
    if target_property == "bandgap":
        # Higher electronegativity -> larger bandgap
        # More elements -> smaller bandgap
        bandgap = avg_electroneg * 1.5 - num_elements * 0.3
        return max(0.0, min(5.0, bandgap))

    # Mock formation energy (-8 to -2 eV/atom)
    elif target_property == "formation_energy":
        # More negative for higher electronegativity difference
        # (simplified - real formation energy is complex)
        formation_energy = -5.0 + (avg_electroneg - 2.0) * 2.0
        return max(-8.0, min(-2.0, formation_energy))

    # Mock stability score (0-1)
    elif target_property == "stability_score":
        stability = 0.7 - (num_elements - 2) * 0.1 + (density / 10.0)
        return max(0.0, min(1.0, stability))

    else:
        logger.warning(f"Unknown target property: {target_property}")
        return None


def export_dataset_to_pandas(dataset: Dataset) -> pd.DataFrame:
    """
    Export dataset to pandas DataFrame.

    Useful for exploratory data analysis and traditional ML methods.

    Args:
        dataset: Dataset object

    Returns:
        DataFrame with features and targets
    """
    rows = []

    for i in range(dataset.num_samples):
        row = {
            "structure_id": dataset.structure_ids[i],
            "target": dataset.targets[i],
            **dataset.scalar_features[i],  # Flatten scalar features
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    logger.info(f"Exported dataset to DataFrame: {df.shape}")
    return df


async def build_train_val_datasets(
    db_session: Any,
    target_property: str,
    train_fraction: float = 0.8,
    min_samples: int = 10,
    **kwargs
) -> TrainValSplit:
    """
    Convenience function to build and split dataset in one call.

    Args:
        db_session: Database session
        target_property: Property to predict
        train_fraction: Fraction for training
        min_samples: Minimum samples required
        **kwargs: Additional DatasetConfig parameters

    Returns:
        TrainValSplit object

    Example:
        >>> split = await build_train_val_datasets(
        ...     db,
        ...     target_property="bandgap",
        ...     train_fraction=0.8,
        ...     min_samples=100
        ... )
        >>> print(f"Train: {split.train.num_samples}, Val: {split.val.num_samples}")
    """
    config = DatasetConfig(
        target_property=target_property,
        train_fraction=train_fraction,
        min_samples=min_samples,
        **kwargs
    )

    # Build full dataset
    full_dataset = await build_regression_dataset(db_session, config)

    # Split into train/val
    split = await split_train_val(full_dataset, config)

    return split
