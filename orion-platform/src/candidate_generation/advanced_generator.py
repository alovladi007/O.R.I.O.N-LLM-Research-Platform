"""
ORION Advanced Candidate Generation with Surrogate Models
========================================================

Implements ensemble surrogate models with uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.transforms import Compose, NormalizeFeatures

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

import logging
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


class GNNSurrogate(nn.Module):
    """Graph Neural Network for materials property prediction"""
    
    def __init__(self, num_node_features: int, hidden_dim: int = 128, num_layers: int = 3, 
                 dropout: float = 0.1, uncertainty_mode: str = 'dropout'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.uncertainty_mode = uncertainty_mode
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention mechanism
        self.attention = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=dropout)
        
        # Prediction heads
        self.property_heads = nn.ModuleDict({
            'bandgap': nn.Linear(hidden_dim, 1),
            'formation_energy': nn.Linear(hidden_dim, 1),
            'bulk_modulus': nn.Linear(hidden_dim, 1),
            'density': nn.Linear(hidden_dim, 1),
            'melting_point': nn.Linear(hidden_dim, 1),
            'thermal_conductivity': nn.Linear(hidden_dim, 1),
            'shear_modulus': nn.Linear(hidden_dim, 1),
            'youngs_modulus': nn.Linear(hidden_dim, 1)
        })
        
        # Uncertainty estimation heads (aleatoric)
        if uncertainty_mode in ['aleatoric', 'both']:
            self.uncertainty_heads = nn.ModuleDict({
                prop: nn.Linear(hidden_dim, 1) for prop in self.property_heads.keys()
            })
        
        self.dropout_layer = nn.Dropout(dropout)
        self.last_features = None
        
    def forward(self, data, return_uncertainty: bool = False, mc_samples: int = 50):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions with residual connections
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = self.dropout_layer(h_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        
        # Attention mechanism
        h = self.attention(h, edge_index)
        h = F.relu(h)
        
        # Global pooling
        h_pool = global_mean_pool(h, batch)
        self.last_features = h_pool
        
        predictions = {}
        uncertainties = {}
        
        if self.training or not return_uncertainty:
            # Standard forward pass
            for prop, head in self.property_heads.items():
                predictions[prop] = head(h_pool).squeeze(-1)
                
                if hasattr(self, 'uncertainty_heads'):
                    # Aleatoric uncertainty (data noise)
                    log_var = self.uncertainty_heads[prop](h_pool).squeeze(-1)
                    uncertainties[prop] = torch.exp(0.5 * log_var)  # std = exp(0.5 * log_var)
        else:
            # Monte Carlo Dropout for epistemic uncertainty
            self.train()  # Enable dropout
            
            mc_predictions = {prop: [] for prop in self.property_heads.keys()}
            mc_uncertainties = {prop: [] for prop in self.property_heads.keys()}
            
            for _ in range(mc_samples):
                sample_preds = {}
                sample_uncerts = {}
                
                for prop, head in self.property_heads.items():
                    pred = head(h_pool).squeeze(-1)
                    sample_preds[prop] = pred
                    
                    if hasattr(self, 'uncertainty_heads'):
                        log_var = self.uncertainty_heads[prop](h_pool).squeeze(-1)
                        sample_uncerts[prop] = torch.exp(0.5 * log_var)
                
                for prop in self.property_heads.keys():
                    mc_predictions[prop].append(sample_preds[prop])
                    if hasattr(self, 'uncertainty_heads'):
                        mc_uncertainties[prop].append(sample_uncerts[prop])
            
            # Compute statistics
            for prop in self.property_heads.keys():
                mc_stack = torch.stack(mc_predictions[prop])
                predictions[prop] = mc_stack.mean(dim=0)  # Epistemic mean
                
                # Total uncertainty = epistemic + aleatoric
                epistemic_std = mc_stack.std(dim=0)
                
                if hasattr(self, 'uncertainty_heads'):
                    aleatoric_std = torch.stack(mc_uncertainties[prop]).mean(dim=0)
                    total_uncertainty = torch.sqrt(epistemic_std**2 + aleatoric_std**2)
                else:
                    total_uncertainty = epistemic_std
                
                uncertainties[prop] = total_uncertainty
            
            self.eval()  # Back to eval mode
        
        if return_uncertainty:
            return predictions, uncertainties
        return predictions


class EnsembleSurrogate:
    """Ensemble of diverse surrogate models"""
    
    def __init__(self, models: List[str] = None):
        if models is None:
            models = ['gnn', 'rf', 'xgb']
        
        self.models = {}
        self.model_weights = {}
        self.physics_checker = None  # Will be injected
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.model_weights[name] = weight
    
    def predict(self, X, return_uncertainty: bool = False) -> Union[Dict, Tuple[Dict, Dict]]:
        """Ensemble prediction with uncertainty quantification"""
        all_predictions = defaultdict(list)
        all_uncertainties = defaultdict(list)
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    preds, uncerts = model.predict_with_uncertainty(X)
                    for prop in preds:
                        all_predictions[prop].append(preds[prop] * self.model_weights[name])
                        all_uncertainties[prop].append(uncerts[prop])
                else:
                    preds = model.predict(X)
                    for prop in preds:
                        all_predictions[prop].append(preds[prop] * self.model_weights[name])
                        all_uncertainties[prop].append(np.zeros_like(preds[prop]))
            
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                continue
        
        # Combine predictions
        ensemble_predictions = {}
        ensemble_uncertainties = {}
        
        for prop in all_predictions:
            pred_array = np.array(all_predictions[prop])
            uncert_array = np.array(all_uncertainties[prop])
            
            # Weighted average for predictions
            weights = np.array([self.model_weights[name] for name in self.models.keys()])
            weights = weights / weights.sum()
            
            ensemble_predictions[prop] = np.average(pred_array, axis=0, weights=weights)
            
            # Uncertainty combination: model disagreement + average individual uncertainty
            model_disagreement = np.std(pred_array, axis=0)
            avg_individual_uncertainty = np.mean(uncert_array, axis=0)
            ensemble_uncertainties[prop] = np.sqrt(model_disagreement**2 + avg_individual_uncertainty**2)
        
        if return_uncertainty:
            return ensemble_predictions, ensemble_uncertainties
        return ensemble_predictions
    
    def validate_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Validate predictions using physics constraints"""
        if not self.physics_checker:
            logger.warning("Physics checker not available, skipping validation")
            return np.ones(len(list(predictions.values())[0]), dtype=bool)
        
        valid_mask = np.ones(len(list(predictions.values())[0]), dtype=bool)
        
        for i in range(len(valid_mask)):
            sample_pred = {prop: values[i] for prop, values in predictions.items()}
            is_valid, _ = self.physics_checker.validate_candidate({'predictions': sample_pred})
            valid_mask[i] = is_valid
        
        return valid_mask


class UncertaintyQuantifier:
    """Comprehensive uncertainty quantification and calibration"""
    
    def __init__(self):
        self.calibration_models = {}
        self.is_calibrated = False
        
    def calibrate_uncertainty(self, predictions: np.ndarray, uncertainties: np.ndarray, 
                            true_values: np.ndarray, method: str = 'isotonic'):
        """Calibrate uncertainty estimates using observed errors"""
        
        # Compute actual errors
        errors = np.abs(predictions - true_values)
        
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(uncertainties, errors)
            self.calibration_models['isotonic'] = calibrator
        
        elif method == 'temperature_scaling':
            # Temperature scaling for neural network outputs
            from scipy.optimize import minimize_scalar
            
            def negative_log_likelihood(temperature):
                scaled_uncertainties = uncertainties / temperature
                # Assume Gaussian likelihood
                nll = 0.5 * np.sum((errors / scaled_uncertainties)**2 + np.log(2 * np.pi * scaled_uncertainties**2))
                return nll
            
            result = minimize_scalar(negative_log_likelihood, bounds=(0.01, 10.0), method='bounded')
            self.calibration_models['temperature'] = result.x
        
        self.is_calibrated = True
        logger.info(f"Uncertainty calibration completed using {method}")
    
    def apply_calibration(self, uncertainties: np.ndarray, method: str = 'isotonic') -> np.ndarray:
        """Apply calibration to new uncertainty estimates"""
        if not self.is_calibrated:
            logger.warning("Uncertainty calibration not performed. Returning original uncertainties.")
            return uncertainties
        
        if method == 'isotonic' and 'isotonic' in self.calibration_models:
            return self.calibration_models['isotonic'].predict(uncertainties)
        elif method == 'temperature_scaling' and 'temperature' in self.calibration_models:
            return uncertainties / self.calibration_models['temperature']
        else:
            return uncertainties


class DiversitySampler:
    """Ensures diversity in generated candidates"""
    
    def __init__(self, diversity_weight: float = 0.3, risk_aversion: float = 1.0):
        self.diversity_weight = diversity_weight
        self.risk_aversion = risk_aversion
        self.previous_candidates = []
        
    def rank_candidates(self, candidates: List[Dict[str, Any]], 
                       target_property: str = 'bandgap',
                       target_value: float = 2.0,
                       maximize: bool = False,
                       return_details: bool = False) -> List[Tuple[int, float]]:
        """Rank candidates by property target and diversity"""
        
        ranked_candidates = []
        
        for i, candidate in enumerate(candidates):
            predictions = candidate.get('predictions', {})
            uncertainties = candidate.get('uncertainties', {})
            
            # Property score
            if target_property in predictions:
                pred_value = predictions[target_property]
                uncertainty = uncertainties.get(target_property, 0.0)
                
                if maximize:
                    property_score = pred_value - self.risk_aversion * uncertainty
                else:
                    distance = abs(pred_value - target_value)
                    property_score = 1.0 / (1.0 + distance) - self.risk_aversion * uncertainty
            else:
                property_score = 0.0
            
            # Diversity score
            embedding = candidate.get('embedding', None)
            if embedding is not None and len(self.previous_candidates) > 0:
                diversity_bonus = self._compute_diversity_bonus(
                    embedding, 
                    [c.get('embedding') for c in self.previous_candidates[-10:]]
                )
            else:
                diversity_bonus = 1.0
            
            # Combined score
            total_score = (1 - self.diversity_weight) * property_score + self.diversity_weight * diversity_bonus
            
            # Penalty for physics violations
            if candidate.get('physics_valid', True) is False:
                total_score *= 0.1
            
            details = {
                'property_score': property_score,
                'diversity_bonus': diversity_bonus,
                'total_score': total_score,
                'uncertainty': uncertainties.get(target_property, 0.0),
                'prediction': predictions.get(target_property, 0.0)
            }
            
            if return_details:
                ranked_candidates.append((i, total_score, details))
            else:
                ranked_candidates.append((i, total_score))
        
        # Sort by score (descending)
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_candidates
    
    def _compute_diversity_bonus(self, embedding: np.ndarray, 
                               previous_embeddings: List[np.ndarray]) -> float:
        """Compute diversity bonus based on embedding distance"""
        if not previous_embeddings or len(previous_embeddings) == 0:
            return 1.0  # Maximum diversity for first candidate
        
        # Compute minimum distance to all previous candidates
        distances = []
        for prev_emb in previous_embeddings:
            if prev_emb is not None:
                # Cosine distance
                distance = 1.0 - np.dot(embedding, prev_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(prev_emb)
                )
                distances.append(distance)
        
        if distances:
            min_distance = min(distances)
            return min_distance  # Higher distance = higher diversity bonus
        else:
            return 1.0
    
    def select_batch(self, candidates: List[Dict[str, Any]], 
                    batch_size: int,
                    target_property: str = 'bandgap',
                    target_value: float = 2.0,
                    maximize: bool = False) -> List[int]:
        """Select a diverse batch of candidates for simulation"""
        
        if len(candidates) <= batch_size:
            return list(range(len(candidates)))
        
        selected_indices = []
        remaining_candidates = list(enumerate(candidates))
        
        while len(selected_indices) < batch_size and remaining_candidates:
            # Rank remaining candidates
            candidate_subset = [candidates[i] for i, _ in remaining_candidates]
            rankings = self.rank_candidates(
                candidate_subset, target_property, target_value, maximize, return_details=True
            )
            
            # Select best candidate
            best_local_idx, best_score, details = rankings[0]
            best_global_idx = remaining_candidates[best_local_idx][0]
            
            selected_indices.append(best_global_idx)
            self.previous_candidates.append(candidates[best_global_idx])
            
            # Remove selected candidate
            remaining_candidates.pop(best_local_idx)
            
            # Update diversity calculations for remaining candidates
            # (This is a simplified greedy approach; could be improved with optimization)
        
        return selected_indices
    
    def adaptive_risk_adjustment(self, success_history: List[bool], 
                               window_size: int = 50) -> float:
        """Adaptively adjust risk aversion based on recent success"""
        
        if len(success_history) < window_size:
            return self.risk_aversion
        
        recent_success_rate = np.mean(success_history[-window_size:])
        
        # If success rate is high, reduce risk aversion (be more exploratory)
        # If success rate is low, increase risk aversion (be more conservative)
        if recent_success_rate > 0.7:
            adjusted_risk = self.risk_aversion * 0.8
        elif recent_success_rate < 0.3:
            adjusted_risk = self.risk_aversion * 1.2
        else:
            adjusted_risk = self.risk_aversion
        
        # Clamp to reasonable bounds
        adjusted_risk = np.clip(adjusted_risk, 0.5, 3.0)
        
        logger.info(f"Adjusted risk aversion: {self.risk_aversion:.2f} -> {adjusted_risk:.2f} "
                   f"(success rate: {recent_success_rate:.1%})")
        
        self.risk_aversion = adjusted_risk
        return adjusted_risk