"""
CGCNN-style Graph Neural Network for Crystal Property Prediction
=================================================================

This module implements a simplified Crystal Graph Convolutional Neural Network (CGCNN)
for predicting material properties from crystal structures.

The model architecture:
1. Atom embeddings: Embed atomic features (atomic number, electronegativity, etc.)
2. Graph convolutions: Message passing between neighboring atoms
3. Pooling: Aggregate atom-level features to crystal-level
4. MLP: Predict target property

Reference:
  Xie & Grossman, "Crystal Graph Convolutional Neural Networks for an
  Accurate and Interpretable Prediction of Material Properties", PRL 2018

Session 15: GNN Model Integration
Session 20: Active Learning - Uncertainty Estimation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Conditional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. GNN model will run in stub mode.")
    PYTORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass
    torch = None


class CGCNNModel:
    """
    CGCNN-style model for crystal property prediction.

    This class provides a unified interface regardless of whether PyTorch is available.
    If PyTorch is not available, it runs in "stub mode" with deterministic predictions.

    Attributes:
        model_name: Name of the model
        target_property: Property being predicted (bandgap, formation_energy, etc.)
        hidden_dim: Dimensionality of hidden layers
        num_layers: Number of graph convolution layers
        device: torch.device (cpu or cuda) if PyTorch available
    """

    def __init__(
        self,
        model_name: str = "cgcnn_bandgap_v1",
        target_property: str = "bandgap",
        hidden_dim: int = 128,
        num_layers: int = 3,
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.target_property = target_property
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device_str = device

        if PYTORCH_AVAILABLE:
            self.device = torch.device(device)
            self._model = _CGCNNModelImpl(
                atom_feature_dim=4,  # atomic_number, mass, electroneg, radius
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=1  # Single target prediction
            ).to(self.device)
            self._model.eval()  # Inference mode by default
        else:
            self.device = None
            self._model = None

        logger.info(f"Initialized CGCNN model: {model_name}")
        logger.info(f"  Target: {target_property}")
        logger.info(f"  PyTorch available: {PYTORCH_AVAILABLE}")
        logger.info(f"  Device: {device}")

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load model weights from checkpoint file.

        Args:
            checkpoint_path: Path to .pth or .pt file

        Raises:
            RuntimeError: If PyTorch is not available
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load checkpoints")

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self._model.load_state_dict(checkpoint)

        self._model.eval()
        logger.info("Checkpoint loaded successfully")

    def predict(self, graph_repr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict property for a single structure.

        Args:
            graph_repr: Graph representation from features.py
                {
                    "atom_features": [...],
                    "neighbor_lists": {...},
                    "bond_distances": {...},
                    ...
                }

        Returns:
            Dictionary with predictions:
            {
                "prediction": float,
                "uncertainty": float (if available),
                "model_name": str,
                "model_version": str
            }
        """
        if PYTORCH_AVAILABLE:
            return self._predict_pytorch(graph_repr)
        else:
            return self._predict_stub(graph_repr)

    def predict_with_uncertainty(
        self,
        graph_repr: Dict[str, Any],
        method: str = "mc_dropout",
        n_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Predict with uncertainty estimation using MC dropout or ensemble.

        Args:
            graph_repr: Graph representation from features.py
            method: Uncertainty estimation method:
                - "mc_dropout": Monte Carlo dropout
                - "ensemble": Ensemble of models (if available)
                - "none": Single forward pass (no uncertainty)
            n_samples: Number of forward passes for MC dropout

        Returns:
            Dictionary with predictions and uncertainty:
            {
                "prediction": float,  # Mean prediction
                "uncertainty": float,  # Standard deviation
                "predictions_sample": List[float],  # All sampled predictions
                "method": str,
                "model_name": str,
                "model_version": str
            }

        Example:
            >>> model = CGCNNModel(model_name="cgcnn_bandgap_v1")
            >>> result = model.predict_with_uncertainty(graph_repr, method="mc_dropout", n_samples=20)
            >>> print(f"Prediction: {result['prediction']:.2f} ± {result['uncertainty']:.2f}")
        """
        if PYTORCH_AVAILABLE:
            if method == "mc_dropout":
                return self._predict_mc_dropout(graph_repr, n_samples)
            elif method == "ensemble":
                # Placeholder for ensemble methods
                logger.warning("Ensemble method not fully implemented, using MC dropout")
                return self._predict_mc_dropout(graph_repr, n_samples)
            else:
                # Single prediction with no uncertainty
                result = self._predict_pytorch(graph_repr)
                result["predictions_sample"] = [result["prediction"]]
                result["method"] = "single"
                return result
        else:
            return self._predict_stub_with_uncertainty(graph_repr)

    def _predict_pytorch(self, graph_repr: Dict[str, Any]) -> Dict[str, Any]:
        """PyTorch-based prediction."""
        # Convert graph_repr to PyTorch tensors
        graph_data = self._graph_to_tensors(graph_repr)

        with torch.no_grad():
            output = self._model(
                graph_data["atom_features"],
                graph_data["edge_index"],
                graph_data["edge_attr"]
            )

            prediction = output.item()

        return {
            "prediction": prediction,
            "uncertainty": 0.0,  # TODO: Add uncertainty estimation
            "model_name": self.model_name,
            "model_version": "1.0.0"
        }

    def _predict_stub(self, graph_repr: Dict[str, Any]) -> Dict[str, Any]:
        """Stub prediction when PyTorch is not available."""
        # Deterministic prediction based on graph features
        num_atoms = graph_repr.get("num_atoms", 1)
        num_edges = graph_repr.get("num_edges", 0)

        # Simple heuristic based on connectivity
        avg_coordination = num_edges / max(num_atoms, 1)

        if self.target_property == "bandgap":
            # Lower coordination -> higher bandgap (insulators)
            prediction = max(0.0, min(5.0, 4.0 - avg_coordination * 0.5))
        elif self.target_property == "formation_energy":
            # Higher coordination -> more negative formation energy
            prediction = -2.0 - avg_coordination * 0.5
        else:
            prediction = 1.0  # Default

        logger.info(f"Stub prediction: {prediction} (PyTorch not available)")

        return {
            "prediction": prediction,
            "uncertainty": 0.5,  # High uncertainty for stub
            "model_name": f"{self.model_name}_stub",
            "model_version": "0.1.0"
        }

    def _predict_mc_dropout(self, graph_repr: Dict[str, Any], n_samples: int = 20) -> Dict[str, Any]:
        """
        Monte Carlo dropout uncertainty estimation.

        Performs multiple forward passes with dropout enabled to estimate
        prediction uncertainty.

        Args:
            graph_repr: Graph representation
            n_samples: Number of MC samples

        Returns:
            Prediction with uncertainty
        """
        # Convert graph to tensors
        graph_data = self._graph_to_tensors(graph_repr)

        predictions = []

        # Enable dropout for MC sampling
        self._model.train()  # Enable dropout

        with torch.no_grad():
            for _ in range(n_samples):
                output = self._model(
                    graph_data["atom_features"],
                    graph_data["edge_index"],
                    graph_data["edge_attr"]
                )
                predictions.append(output.item())

        # Return to eval mode
        self._model.eval()

        # Compute statistics
        predictions = np.array(predictions)
        mean_pred = float(np.mean(predictions))
        std_pred = float(np.std(predictions))

        logger.debug(
            f"MC Dropout ({n_samples} samples): "
            f"prediction={mean_pred:.3f}, uncertainty={std_pred:.3f}"
        )

        return {
            "prediction": mean_pred,
            "uncertainty": std_pred,
            "predictions_sample": predictions.tolist(),
            "method": "mc_dropout",
            "n_samples": n_samples,
            "model_name": self.model_name,
            "model_version": "1.0.0"
        }

    def _predict_stub_with_uncertainty(self, graph_repr: Dict[str, Any]) -> Dict[str, Any]:
        """Stub prediction with simulated uncertainty."""
        base_pred = self._predict_stub(graph_repr)

        # Simulate MC dropout by adding noise
        predictions = []
        base_value = base_pred["prediction"]

        for _ in range(20):
            # Add random noise to simulate uncertainty
            noise = np.random.normal(0, 0.1 * abs(base_value) + 0.05)
            predictions.append(base_value + noise)

        predictions = np.array(predictions)

        return {
            "prediction": float(np.mean(predictions)),
            "uncertainty": float(np.std(predictions)),
            "predictions_sample": predictions.tolist(),
            "method": "mc_dropout_stub",
            "n_samples": 20,
            "model_name": f"{self.model_name}_stub",
            "model_version": "0.1.0"
        }

    def _graph_to_tensors(self, graph_repr: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert graph representation to PyTorch tensors.

        Args:
            graph_repr: Graph representation dictionary

        Returns:
            Dictionary of tensors for model input
        """
        # Extract atom features
        atom_features = []
        for atom in graph_repr["atom_features"]:
            features = [
                atom["atomic_number"],
                atom["atomic_mass"],
                atom["electronegativity"],
                atom["atomic_radius"]
            ]
            atom_features.append(features)

        atom_features_tensor = torch.tensor(atom_features, dtype=torch.float32, device=self.device)

        # Build edge list and edge attributes
        edge_list = []
        edge_attrs = []

        neighbor_lists = graph_repr["neighbor_lists"]
        bond_distances = graph_repr["bond_distances"]

        for i_str, neighbors in neighbor_lists.items():
            i = int(i_str)
            for j in neighbors:
                edge_list.append([i, j])

                # Get bond distance
                dist_key = f"{i}_{j}"
                dist = bond_distances.get(dist_key, 1.0)
                edge_attrs.append([dist])

        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=self.device)

        return {
            "atom_features": atom_features_tensor,
            "edge_index": edge_index,
            "edge_attr": edge_attr
        }


if PYTORCH_AVAILABLE:
    class _CGCNNModelImpl(nn.Module):
        """
        Internal PyTorch model implementation.

        This is a simplified CGCNN for demonstration. A production version would:
        - Use torch_geometric for graph operations
        - Implement proper edge convolutions
        - Add batch normalization and dropout
        - Support variable-size graphs efficiently
        """

        def __init__(
            self,
            atom_feature_dim: int = 4,
            hidden_dim: int = 128,
            num_layers: int = 3,
            output_dim: int = 1,
            dropout: float = 0.1
        ):
            super().__init__()

            self.atom_feature_dim = atom_feature_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout

            # Atom embedding
            self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)

            # Graph convolution layers (simplified)
            self.conv_layers = nn.ModuleList([
                nn.Linear(hidden_dim + 1, hidden_dim)  # +1 for edge feature (distance)
                for _ in range(num_layers)
            ])

            # Dropout layers for uncertainty estimation
            self.dropout_layers = nn.ModuleList([
                nn.Dropout(dropout)
                for _ in range(num_layers)
            ])

            # Output MLP with dropout
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )

        def forward(
            self,
            atom_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                atom_features: [num_atoms, atom_feature_dim]
                edge_index: [2, num_edges]
                edge_attr: [num_edges, edge_feature_dim]

            Returns:
                Predicted property [1]
            """
            # Embed atoms
            h = self.atom_embedding(atom_features)  # [num_atoms, hidden_dim]
            h = F.relu(h)

            # Graph convolutions (simplified message passing)
            for conv, dropout in zip(self.conv_layers, self.dropout_layers):
                h_new = h.clone()

                # For each edge, pass messages
                if edge_index.size(1) > 0:  # Check if edges exist
                    for e in range(edge_index.size(1)):
                        i, j = edge_index[0, e], edge_index[1, e]
                        edge_feat = edge_attr[e]

                        # Concatenate neighbor feature and edge feature
                        message = torch.cat([h[j], edge_feat])

                        # Update node i
                        h_new[i] = h_new[i] + conv(message)

                h = F.relu(h_new)
                h = dropout(h)  # Apply dropout for uncertainty estimation

            # Global pooling (mean over atoms)
            graph_feature = torch.mean(h, dim=0)  # [hidden_dim]

            # Predict property
            output = self.output_mlp(graph_feature)

            return output


# ============================================================================
# Model Registry
# ============================================================================

_MODEL_REGISTRY: Dict[str, CGCNNModel] = {}


def register_model(model: CGCNNModel):
    """Register a model in the global registry."""
    _MODEL_REGISTRY[model.model_name] = model
    logger.info(f"Registered model: {model.model_name}")


def get_model(model_name: str) -> Optional[CGCNNModel]:
    """Get a model from the registry."""
    return _MODEL_REGISTRY.get(model_name)


def list_models() -> List[str]:
    """List all registered models."""
    return list(_MODEL_REGISTRY.keys())


# Initialize default models
def _initialize_default_models():
    """Initialize default pre-trained models."""
    # These would be loaded from checkpoint files in production
    default_models = [
        CGCNNModel(
            model_name="cgcnn_bandgap_v1",
            target_property="bandgap",
            hidden_dim=128,
            num_layers=3
        ),
        CGCNNModel(
            model_name="cgcnn_formation_energy_v1",
            target_property="formation_energy",
            hidden_dim=128,
            num_layers=3
        ),
    ]

    for model in default_models:
        register_model(model)


# Auto-initialize on import
_initialize_default_models()
