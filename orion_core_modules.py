"""
ORION: Complete Implementation - Core Modules
============================================

This module implements the complete ORION system with:
1. Bottleneck analysis and monitoring
2. Physics sanity checks
3. Ensemble diversity
4. Uncertainty quantification (aleatoric + epistemic)
5. Surrogate predictor training pipeline
6. Conflict resolution service
7. Evaluation framework

Author: ORION Development Team
License: MIT
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

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import time
import json
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings
import psutil
import GPUtil
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. BOTTLENECK ANALYSIS & MONITORING
# =====================================================================

@dataclass
class PerformanceMetrics:
    """Track system performance across different stages"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    processing_time: float = 0.0
    throughput: float = 0.0
    queue_size: int = 0
    error_rate: float = 0.0

class BottleneckAnalyzer:
    """Real-time system bottleneck detection and analysis"""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_window))
        self.bottleneck_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'gpu_usage': 90.0,
            'gpu_memory': 85.0,
            'error_rate': 5.0,
            'queue_size': 50
        }
        
    def monitor_performance(func):
        """Decorator for monitoring function performance"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            start_memory = psutil.virtual_memory().percent
            
            # GPU metrics if available
            gpu_usage = gpu_memory = 0.0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    gpu_memory = gpus[0].memoryUtil * 100
            except:
                pass
            
            try:
                result = func(*args, **kwargs)
                error_occurred = False
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                error_occurred = True
                raise
            finally:
                end_time = time.time()
                processing_time = end_time - start_time
                
                metrics = PerformanceMetrics(
                    cpu_usage=psutil.cpu_percent() - start_cpu,
                    memory_usage=psutil.virtual_memory().percent - start_memory,
                    gpu_usage=gpu_usage,
                    gpu_memory=gpu_memory,
                    processing_time=processing_time,
                    error_rate=1.0 if error_occurred else 0.0
                )
                
                self.record_metrics(func.__name__, metrics)
                
            return result
        return wrapper
    
    def record_metrics(self, stage: str, metrics: PerformanceMetrics):
        """Record performance metrics for a stage"""
        for key, value in metrics.__dict__.items():
            self.metrics_history[f"{stage}_{key}"].append(value)
    
    def detect_bottlenecks(self) -> Dict[str, List[str]]:
        """Detect current system bottlenecks"""
        bottlenecks = defaultdict(list)
        
        for metric_key, history in self.metrics_history.items():
            if not history:
                continue
                
            stage, metric = metric_key.rsplit('_', 1)
            if metric in self.bottleneck_thresholds:
                recent_avg = np.mean(list(history)[-10:])  # Last 10 measurements
                threshold = self.bottleneck_thresholds[metric]
                
                if recent_avg > threshold:
                    bottlenecks[stage].append(f"{metric}: {recent_avg:.1f}% (threshold: {threshold}%)")
        
        return dict(bottlenecks)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'bottlenecks': self.detect_bottlenecks(),
            'stage_performance': {},
            'recommendations': []
        }
        
        # Aggregate by stage
        stages = set(key.rsplit('_', 1)[0] for key in self.metrics_history.keys())
        for stage in stages:
            stage_metrics = {}
            for metric in ['cpu_usage', 'memory_usage', 'processing_time', 'error_rate']:
                key = f"{stage}_{metric}"
                if key in self.metrics_history and self.metrics_history[key]:
                    values = list(self.metrics_history[key])
                    stage_metrics[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'p95': np.percentile(values, 95),
                        'recent_trend': np.mean(values[-5:]) - np.mean(values[-15:-5]) if len(values) >= 15 else 0
                    }
            report['stage_performance'][stage] = stage_metrics
        
        # Generate recommendations
        bottlenecks = report['bottlenecks']
        if bottlenecks:
            for stage, issues in bottlenecks.items():
                if 'cpu_usage' in str(issues):
                    report['recommendations'].append(f"Scale {stage} horizontally or optimize CPU-bound operations")
                if 'memory_usage' in str(issues):
                    report['recommendations'].append(f"Increase memory for {stage} or implement memory optimization")
                if 'gpu' in str(issues):
                    report['recommendations'].append(f"Add GPU resources for {stage} or optimize batch sizes")
        
        return report

# =====================================================================
# 2. PHYSICS SANITY CHECK LAYER
# =====================================================================

class PhysicsSanityChecker:
    """Validate predictions against physical constraints"""
    
    def __init__(self):
        self.constraints = {
            'bandgap': {'min': 0.0, 'max': 15.0, 'unit': 'eV'},
            'density': {'min': 0.1, 'max': 25.0, 'unit': 'g/cm³'},
            'melting_point': {'min': 0.0, 'max': 4000.0, 'unit': 'K'},
            'dielectric_constant': {'min': 1.0, 'max': 10000.0, 'unit': 'dimensionless'},
            'bulk_modulus': {'min': 0.1, 'max': 500.0, 'unit': 'GPa'},
            'shear_modulus': {'min': 0.1, 'max': 300.0, 'unit': 'GPa'},
            'youngs_modulus': {'min': 0.1, 'max': 1000.0, 'unit': 'GPa'},
            'thermal_conductivity': {'min': 0.001, 'max': 2000.0, 'unit': 'W/m·K'},
            'electrical_conductivity': {'min': 1e-20, 'max': 1e8, 'unit': 'S/m'},
            'formation_energy': {'min': -10.0, 'max': 10.0, 'unit': 'eV/atom'}
        }
        
        # Chemical composition constraints
        self.element_constraints = {
            'electronegativity_difference': {'max': 4.0},
            'atomic_radius_ratio': {'min': 0.2, 'max': 5.0},
            'valence_electron_range': {'min': 1, 'max': 8}
        }
    
    def check_property_constraints(self, predictions: Dict[str, float]) -> Dict[str, bool]:
        """Check if predicted properties satisfy physical constraints"""
        results = {}
        violations = []
        
        for prop, value in predictions.items():
            if prop in self.constraints:
                constraint = self.constraints[prop]
                is_valid = constraint['min'] <= value <= constraint['max']
                results[prop] = is_valid
                
                if not is_valid:
                    violations.append(
                        f"{prop}: {value:.3f} {constraint['unit']} "
                        f"(valid range: {constraint['min']}-{constraint['max']} {constraint['unit']})"
                    )
        
        if violations:
            logger.warning(f"Physics constraint violations: {'; '.join(violations)}")
        
        return results
    
    def check_thermodynamic_consistency(self, predictions: Dict[str, float]) -> bool:
        """Check thermodynamic relationships between properties"""
        checks = []
        
        # Check if formation energy is reasonable for stability
        if 'formation_energy' in predictions:
            fe = predictions['formation_energy']
            if fe > 2.0:  # Very positive formation energy suggests instability
                checks.append(False)
                logger.warning(f"High formation energy ({fe:.3f} eV/atom) suggests thermodynamic instability")
            else:
                checks.append(True)
        
        # Check mechanical property relationships (Bulk modulus should be > Shear modulus)
        if 'bulk_modulus' in predictions and 'shear_modulus' in predictions:
            bulk = predictions['bulk_modulus']
            shear = predictions['shear_modulus']
            is_valid = bulk >= shear
            checks.append(is_valid)
            if not is_valid:
                logger.warning(f"Bulk modulus ({bulk:.1f}) < Shear modulus ({shear:.1f})")
        
        # Check Poisson's ratio bounds if calculable
        if all(k in predictions for k in ['bulk_modulus', 'shear_modulus']):
            bulk = predictions['bulk_modulus']
            shear = predictions['shear_modulus']
            poisson = (3*bulk - 2*shear) / (6*bulk + 2*shear)
            is_valid = -1.0 <= poisson <= 0.5
            checks.append(is_valid)
            if not is_valid:
                logger.warning(f"Calculated Poisson's ratio ({poisson:.3f}) outside physical bounds [-1, 0.5]")
        
        return all(checks) if checks else True
    
    def validate_candidate(self, candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Comprehensive validation of a material candidate"""
        errors = []
        
        # Extract predictions
        if 'predictions' not in candidate:
            errors.append("No predictions found in candidate")
            return False, errors
        
        predictions = candidate['predictions']
        
        # Check property constraints
        constraint_results = self.check_property_constraints(predictions)
        failed_constraints = [prop for prop, valid in constraint_results.items() if not valid]
        if failed_constraints:
            errors.extend([f"Property constraint violation: {prop}" for prop in failed_constraints])
        
        # Check thermodynamic consistency
        if not self.check_thermodynamic_consistency(predictions):
            errors.append("Thermodynamic inconsistency detected")
        
        # Check chemical composition if available
        if 'composition' in candidate:
            comp_errors = self._validate_composition(candidate['composition'])
            errors.extend(comp_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_composition(self, composition: Dict[str, float]) -> List[str]:
        """Validate chemical composition"""
        errors = []
        
        # Check if composition sums to ~1.0
        total = sum(composition.values())
        if not (0.95 <= total <= 1.05):
            errors.append(f"Composition sum ({total:.3f}) not close to 1.0")
        
        # Check for reasonable stoichiometry
        values = list(composition.values())
        if any(v <= 0 for v in values):
            errors.append("Negative or zero composition fractions")
        
        if max(values) > 0.9:  # Single element dominance
            errors.append("Single element dominance (>90%) may not form compound")
        
        return errors

# =====================================================================
# 3. SURROGATE PREDICTOR MODELS WITH ENSEMBLE DIVERSITY
# =====================================================================

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
            'density': nn.Linear(hidden_dim, 1)
        })
        
        # Uncertainty estimation heads (aleatoric)
        if uncertainty_mode in ['aleatoric', 'both']:
            self.uncertainty_heads = nn.ModuleDict({
                prop: nn.Linear(hidden_dim, 1) for prop in self.property_heads.keys()
            })
        
        self.dropout_layer = nn.Dropout(dropout)
        
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
        self.physics_checker = PhysicsSanityChecker()
        
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
        valid_mask = np.ones(len(list(predictions.values())[0]), dtype=bool)
        
        for i in range(len(valid_mask)):
            sample_pred = {prop: values[i] for prop, values in predictions.items()}
            is_valid, _ = self.physics_checker.validate_candidate({'predictions': sample_pred})
            valid_mask[i] = is_valid
        
        return valid_mask

# =====================================================================
# 4. UNCERTAINTY QUANTIFICATION MODULE
# =====================================================================

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
    
    def plot_calibration_curves(self, predictions: np.ndarray, uncertainties: np.ndarray,
                              true_values: np.ndarray, n_bins: int = 10, save_path: str = None):
        """Plot uncertainty calibration curves"""
        errors = np.abs(predictions - true_values)
        
        # Create bins based on uncertainty quantiles
        bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        bin_centers = (bin_boundaries[1:] + bin_boundaries[:-1]) / 2
        
        mean_uncertainties = []
        mean_errors = []
        error_bars = []
        
        for i in range(n_bins):
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                mean_uncertainties.append(np.mean(uncertainties[mask]))
                mean_errors.append(np.mean(errors[mask]))
                error_bars.append(np.std(errors[mask]) / np.sqrt(np.sum(mask)))
            else:
                mean_uncertainties.append(bin_centers[i])
                mean_errors.append(0)
                error_bars.append(0)
        
        # Plot calibration curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration plot
        ax1.errorbar(mean_uncertainties, mean_errors, yerr=error_bars, 
                    marker='o', capsize=5, capthick=2, label='Observed')
        ax1.plot([0, max(mean_uncertainties)], [0, max(mean_uncertainties)], 
                'r--', label='Perfect calibration')
        ax1.set_xlabel('Predicted Uncertainty')
        ax1.set_ylabel('Observed Error')
        ax1.set_title('Uncertainty Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution of uncertainties vs errors
        ax2.scatter(uncertainties, errors, alpha=0.5, s=10)
        ax2.plot([0, max(uncertainties)], [0, max(uncertainties)], 'r--', label='Perfect calibration')
        ax2.set_xlabel('Predicted Uncertainty')
        ax2.set_ylabel('Observed Error')
        ax2.set_title('Uncertainty vs Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_uncertainty_metrics(self, predictions: np.ndarray, uncertainties: np.ndarray,
                                  true_values: np.ndarray) -> Dict[str, float]:
        """Compute uncertainty quantification metrics"""
        errors = np.abs(predictions - true_values)
        
        # Calibration error (reliability)
        n_bins = 10
        bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        calibration_error = 0.0
        
        for i in range(n_bins):
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                mean_uncertainty = np.mean(uncertainties[mask])
                mean_error = np.mean(errors[mask])
                calibration_error += np.abs(mean_uncertainty - mean_error) * np.sum(mask)
        
        calibration_error /= len(uncertainties)
        
        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainties)
        
        # Coverage at different confidence levels
        coverage_metrics = {}
        for confidence in [0.68, 0.95, 0.99]:  # 1σ, 2σ, 3σ
            z_score = {0.68: 1.0, 0.95: 1.96, 0.99: 2.58}[confidence]
            in_interval = errors <= z_score * uncertainties
            coverage_metrics[f'coverage_{int(confidence*100)}'] = np.mean(in_interval)
        
        # Correlation between uncertainty and error
        uncertainty_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
        
        return {
            'calibration_error': calibration_error,
            'sharpness': sharpness,
            'uncertainty_error_correlation': uncertainty_error_corr,
            **coverage_metrics
        }

# =====================================================================
# 5. TRAINING PIPELINE WITH HYPERPARAMETER TUNING
# =====================================================================

class SurrogateTrainingPipeline:
    """Complete training pipeline for surrogate models"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.physics_checker = PhysicsSanityChecker()
        
        # Model tracking
        self.models = {}
        self.training_history = {}
        self.best_models = {}
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'model_types': ['gnn', 'rf', 'xgb'],
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'random_state': 42,
            'n_trials': 50,  # For hyperparameter optimization
            'early_stopping_patience': 10,
            'batch_size': 32,
            'max_epochs': 100,
            'learning_rate': 1e-3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    @BottleneckAnalyzer.monitor_performance
    def prepare_data(self, dataset: Any, target_properties: List[str]) -> Tuple[Any, Any, Any]:
        """Prepare training, validation, and test sets"""
        
        # Split dataset
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        
        train_idx, temp_idx = train_test_split(
            indices, test_size=1-self.config['train_split'], 
            random_state=self.config['random_state']
        )
        
        val_ratio = self.config['val_split'] / (self.config['val_split'] + self.config['test_split'])
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1-val_ratio,
            random_state=self.config['random_state']
        )
        
        # Create data loaders
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        logger.info(f"Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
        
        return train_loader, val_loader, test_loader
    
    def optimize_hyperparameters(self, model_type: str, train_loader: DataLoader, 
                                val_loader: DataLoader, target_properties: List[str]) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna-style approach"""
        
        def objective(trial_params: Dict[str, Any]) -> float:
            # Create model with trial parameters
            if model_type == 'gnn':
                model = GNNSurrogate(
                    num_node_features=trial_params['num_node_features'],
                    hidden_dim=trial_params['hidden_dim'],
                    num_layers=trial_params['num_layers'],
                    dropout=trial_params['dropout'],
                    uncertainty_mode='both'
                ).to(self.config['device'])
                
                optimizer = torch.optim.Adam(model.parameters(), lr=trial_params['learning_rate'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
                
                # Training loop
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(self.config['max_epochs']):
                    # Training
                    model.train()
                    train_losses = []
                    
                    for batch in train_loader:
                        batch = batch.to(self.config['device'])
                        optimizer.zero_grad()
                        
                        predictions = model(batch)
                        loss = 0
                        
                        for prop in target_properties:
                            if hasattr(batch, prop):
                                target = getattr(batch, prop)
                                pred = predictions[prop]
                                loss += F.mse_loss(pred, target)
                        
                        loss.backward()
                        optimizer.step()
                        train_losses.append(loss.item())
                    
                    # Validation
                    model.eval()
                    val_losses = []
                    
                    with torch.no_grad():
                        for batch in val_loader:
                            batch = batch.to(self.config['device'])
                            predictions = model(batch)
                            
                            loss = 0
                            for prop in target_properties:
                                if hasattr(batch, prop):
                                    target = getattr(batch, prop)
                                    pred = predictions[prop]
                                    loss += F.mse_loss(pred, target)
                            
                            val_losses.append(loss.item())
                    
                    val_loss = np.mean(val_losses)
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.config['early_stopping_patience']:
                            break
                
                return best_val_loss
            
            else:
                # Implement other model types (RF, XGB) here
                return float('inf')
        
        # Hyperparameter search space
        search_spaces = {
            'gnn': {
                'hidden_dim': [64, 128, 256, 512],
                'num_layers': [2, 3, 4, 5],
                'dropout': [0.0, 0.1, 0.2, 0.3],
                'learning_rate': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
                'num_node_features': [train_loader.dataset[0].x.size(1)]
            }
        }
        
        if model_type not in search_spaces:
            raise ValueError(f"Unknown model type: {model_type}")
        
        search_space = search_spaces[model_type]
        best_params = None
        best_score = float('inf')
        
        # Simple grid search (replace with Optuna for production)
        import itertools
        
        param_combinations = []
        for param_name, values in search_space.items():
            if len(param_combinations) == 0:
                param_combinations = [{param_name: v} for v in values]
            else:
                new_combinations = []
                for combo in param_combinations:
                    for value in values:
                        new_combo = combo.copy()
                        new_combo[param_name] = value
                        new_combinations.append(new_combo)
                param_combinations = new_combinations
        
        # Limit trials
        if len(param_combinations) > self.config['n_trials']:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:self.config['n_trials']]
        
        logger.info(f"Starting hyperparameter optimization for {model_type} with {len(param_combinations)} trials")
        
        for i, params in enumerate(param_combinations):
            try:
                score = objective(params)
                if score < best_score:
                    best_score = score
                    best_params = params
                
                if i % 10 == 0:
                    logger.info(f"Trial {i}/{len(param_combinations)}, Best score: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"Trial {i} failed: {e}")
                continue
        
        logger.info(f"Hyperparameter optimization completed. Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    @BottleneckAnalyzer.monitor_performance
    def train_model(self, model_type: str, train_loader: DataLoader, val_loader: DataLoader,
                   target_properties: List[str], best_params: Dict[str, Any] = None) -> Any:
        """Train a single model with best parameters"""
        
        if model_type == 'gnn':
            model = GNNSurrogate(
                num_node_features=best_params['num_node_features'],
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout'],
                uncertainty_mode='both'
            ).to(self.config['device'])
            
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # Training history
            history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            logger.info(f"Training {model_type} model...")
            
            for epoch in range(self.config['max_epochs']):
                # Training phase
                model.train()
                train_losses = []
                
                for batch in train_loader:
                    batch = batch.to(self.config['device'])
                    optimizer.zero_grad()
                    
                    predictions = model(batch)
                    loss = 0
                    
                    # Multi-task loss
                    for prop in target_properties:
                        if hasattr(batch, prop):
                            target = getattr(batch, prop)
                            pred = predictions[prop]
                            
                            # MSE loss for property prediction
                            mse_loss = F.mse_loss(pred, target)
                            loss += mse_loss
                            
                            # Add uncertainty regularization if available
                            if hasattr(model, 'uncertainty_heads'):
                                log_var = model.uncertainty_heads[prop](model.last_features)
                                # Negative log-likelihood with learnable variance
                                nll_loss = 0.5 * (torch.exp(-log_var) * (pred - target)**2 + log_var)
                                loss += nll_loss.mean()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())
                
                # Validation phase
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.config['device'])
                        predictions = model(batch)
                        
                        loss = 0
                        for prop in target_properties:
                            if hasattr(batch, prop):
                                target = getattr(batch, prop)
                                pred = predictions[prop]
                                loss += F.mse_loss(pred, target)
                        
                        val_losses.append(loss.item())
                
                # Update metrics
                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                current_lr = optimizer.param_groups[0]['lr']
                
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['learning_rate'].append(current_lr)
                
                scheduler.step(val_loss)
                
                # Early stopping and model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Logging
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.6f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Store training history
            self.training_history[model_type] = history
            
            return model
        
        else:
            # Implement other model types
            logger.warning(f"Model type {model_type} not implemented yet")
            return None
    
    def evaluate_model(self, model: Any, test_loader: DataLoader, 
                      target_properties: List[str]) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        model.eval()
        all_predictions = {prop: [] for prop in target_properties}
        all_targets = {prop: [] for prop in target_properties}
        all_uncertainties = {prop: [] for prop in target_properties}
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.config['device'])
                
                # Get predictions with uncertainty
                if hasattr(model, 'forward'):
                    predictions, uncertainties = model(batch, return_uncertainty=True, mc_samples=50)
                else:
                    predictions = model.predict(batch)
                    uncertainties = {prop: np.zeros_like(pred) for prop, pred in predictions.items()}
                
                for prop in target_properties:
                    if hasattr(batch, prop):
                        target = getattr(batch, prop)
                        pred = predictions[prop]
                        uncert = uncertainties.get(prop, torch.zeros_like(pred))
                        
                        all_predictions[prop].extend(pred.cpu().numpy())
                        all_targets[prop].extend(target.cpu().numpy())
                        all_uncertainties[prop].extend(uncert.cpu().numpy())
        
        # Compute metrics
        metrics = {}
        
        for prop in target_properties:
            if len(all_predictions[prop]) > 0:
                pred_array = np.array(all_predictions[prop])
                target_array = np.array(all_targets[prop])
                uncert_array = np.array(all_uncertainties[prop])
                
                # Basic metrics
                mae = mean_absolute_error(target_array, pred_array)
                r2 = r2_score(target_array, pred_array)
                rmse = np.sqrt(np.mean((pred_array - target_array)**2))
                
                # Physics validation
                physics_violations = 0
                for i in range(len(pred_array)):
                    is_valid, _ = self.physics_checker.validate_candidate({
                        'predictions': {prop: pred_array[i]}
                    })
                    if not is_valid:
                        physics_violations += 1
                
                physics_violation_rate = physics_violations / len(pred_array)
                
                # Uncertainty metrics
                uncertainty_metrics = self.uncertainty_quantifier.compute_uncertainty_metrics(
                    pred_array, uncert_array, target_array
                )
                
                metrics[prop] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'physics_violation_rate': physics_violation_rate,
                    **uncertainty_metrics
                }
        
        return metrics
    
    def run_complete_pipeline(self, dataset: Any, target_properties: List[str]) -> Dict[str, Any]:
        """Run the complete training and evaluation pipeline"""
        
        logger.info("Starting complete surrogate training pipeline...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(dataset, target_properties)
        
        results = {
            'models': {},
            'metrics': {},
            'training_history': {},
            'best_parameters': {},
            'ensemble_metrics': {}
        }
        
        ensemble_models = {}
        
        # Train each model type
        for model_type in self.config['model_types']:
            logger.info(f"Training {model_type} model...")
            
            try:
                # Hyperparameter optimization
                best_params = self.optimize_hyperparameters(
                    model_type, train_loader, val_loader, target_properties
                )
                
                # Train model with best parameters
                model = self.train_model(
                    model_type, train_loader, val_loader, target_properties, best_params
                )
                
                if model is not None:
                    # Evaluate model
                    metrics = self.evaluate_model(model, test_loader, target_properties)
                    
                    # Store results
                    results['models'][model_type] = model
                    results['metrics'][model_type] = metrics
                    results['training_history'][model_type] = self.training_history.get(model_type, {})
                    results['best_parameters'][model_type] = best_params
                    
                    ensemble_models[model_type] = model
                    
                    logger.info(f"{model_type} model training completed successfully")
                
            except Exception as e:
                logger.error(f"Failed to train {model_type} model: {e}")
                continue
        
        # Create and evaluate ensemble
        if len(ensemble_models) > 1:
            logger.info("Creating ensemble model...")
            ensemble = EnsembleSurrogate()
            
            for name, model in ensemble_models.items():
                ensemble.add_model(name, model, weight=1.0)
            
            # Evaluate ensemble (simplified - would need proper implementation)
            # ensemble_metrics = self.evaluate_ensemble(ensemble, test_loader, target_properties)
            # results['ensemble_metrics'] = ensemble_metrics
            
            results['ensemble'] = ensemble
        
        # Performance analysis
        performance_report = self.bottleneck_analyzer.get_performance_report()
        results['performance_analysis'] = performance_report
        
        logger.info("Complete pipeline finished successfully!")
        logger.info(f"Performance report: {performance_report}")
        
        return results

# =====================================================================
# 6. UNCERTAINTY-AWARE CANDIDATE RANKING
# =====================================================================

class UncertaintyAwareRanker:
    """Rank material candidates considering uncertainty and risk preferences"""
    
    def __init__(self, risk_aversion: float = 1.5, diversity_weight: float = 0.2):
        self.risk_aversion = risk_aversion  # λ parameter
        self.diversity_weight = diversity_weight
        self.physics_checker = PhysicsSanityChecker()
        
    def compute_candidate_score(self, prediction: Dict[str, float], 
                              uncertainty: Dict[str, float],
                              target_property: str = 'bandgap',
                              target_value: float = 2.0,
                              maximize: bool = False) -> float:
        """Compute risk-adjusted score for a candidate"""
        
        if target_property not in prediction or target_property not in uncertainty:
            return -float('inf')
        
        pred_value = prediction[target_property]
        uncert_value = uncertainty[target_property]
        
        # Distance from target (smaller is better)
        target_distance = abs(pred_value - target_value)
        
        # Convert to utility (larger is better)
        if maximize:
            utility = pred_value
        else:
            utility = -target_distance
        
        # Risk-adjusted score: μ - λσ
        risk_adjusted_score = utility - self.risk_aversion * uncert_value
        
        return risk_adjusted_score
    
    def rank_candidates(self, candidates: List[Dict[str, Any]], 
                       target_property: str = 'bandgap',
                       target_value: float = 2.0,
                       maximize: bool = False,
                       return_details: bool = False) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Rank candidates with uncertainty-aware scoring"""
        
        ranked_candidates = []
        
        for i, candidate in enumerate(candidates):
            # Extract predictions and uncertainties
            predictions = candidate.get('predictions', {})
            uncertainties = candidate.get('uncertainties', {})
            
            # Physics validation
            is_valid, violations = self.physics_checker.validate_candidate(candidate)
            
            if not is_valid:
                # Heavily penalize physics violations
                score = -1000.0
                details = {'physics_valid': False, 'violations': violations}
            else:
                # Compute risk-adjusted score
                score = self.compute_candidate_score(
                    predictions, uncertainties, target_property, target_value, maximize
                )
                
                # Add diversity bonus (if embedding available)
                diversity_bonus = 0.0
                if 'embedding' in candidate:
                    diversity_bonus = self._compute_diversity_bonus(
                        candidate['embedding'], [c.get('embedding') for c in candidates[:i]]
                    )
                
                score += self.diversity_weight * diversity_bonus
                
                details = {
                    'physics_valid': True,
                    'base_score': score - self.diversity_weight * diversity_bonus,
                    'diversity_bonus': diversity_bonus,
                    'uncertainty': uncertainties.get(target_property, 0.0),
                    'prediction': predictions.get(target_property, 0.0)
                }
            
            if return_details:
                ranked_candidates.append((i, score, details))
            else:
                ranked_candidates.append((i, score))
        
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
        
        return adjusted_risk

# =====================================================================
# 7. PROVENANCE-WEIGHTED CONSENSUS ALGORITHM
# =====================================================================

@dataclass
class SourceMetadata:
    """Metadata for literature sources"""
    doi: str
    publication_date: str
    citation_count: int
    journal_impact_factor: float
    source_type: str  # 'journal', 'preprint', 'patent', etc.
    confidence_score: float = 1.0  # Manual curation score
    
class ProvenanceWeightedConsensus:
    """Handle conflicting property values using source reliability"""
    
    def __init__(self):
        self.source_reliability = {}  # DOI -> reliability score
        self.journal_reliability = {}  # Journal -> base reliability
        self.property_variance_thresholds = {
            'bandgap': 0.2,  # eV
            'formation_energy': 0.1,  # eV/atom
            'bulk_modulus': 10.0,  # GPa
            'density': 0.5,  # g/cm³
        }
        
    def compute_source_weight(self, metadata: SourceMetadata) -> float:
        """Compute reliability weight for a source"""
        
        # Base weight from citation count (logarithmic scaling)
        citation_weight = np.log10(max(metadata.citation_count, 1) + 1)
        
        # Freshness factor (prefer recent publications)
        current_year = 2025
        pub_year = int(metadata.publication_date[:4])
        age_years = current_year - pub_year
        freshness_factor = np.exp(-age_years / 10.0)  # Half-life of 10 years
        
        # Journal impact factor
        impact_weight = min(metadata.journal_impact_factor / 10.0, 3.0)  # Cap at 3x weight
        
        # Source type modifier
        type_modifiers = {
            'journal': 1.0,
            'preprint': 0.7,
            'patent': 0.8,
            'thesis': 0.6,
            'conference': 0.9
        }
        type_modifier = type_modifiers.get(metadata.source_type, 0.5)
        
        # Combine weights
        total_weight = (citation_weight * freshness_factor * impact_weight * 
                       type_modifier * metadata.confidence_score)
        
        return max(total_weight, 0.1)  # Minimum weight
    
    def resolve_property_conflict(self, property_values: List[Tuple[float, SourceMetadata]], 
                                property_name: str) -> Dict[str, Any]:
        """Resolve conflicting property values"""
        
        if len(property_values) == 1:
            value, metadata = property_values[0]
            return {
                'consensus_value': value,
                'uncertainty': 0.0,
                'num_sources': 1,
                'weight_sum': self.compute_source_weight(metadata),
                'is_disputed': False,
                'source_spread': 0.0
            }
        
        # Compute weights
        values = []
        weights = []
        
        for value, metadata in property_values:
            weight = self.compute_source_weight(metadata)
            values.append(value)
            weights.append(weight)
            
            # Update source reliability tracking
            self.source_reliability[metadata.doi] = weight
        
        values = np.array(values)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Weighted consensus
        consensus_value = np.average(values, weights=weights)
        
        # Uncertainty estimation
        weighted_variance = np.average((values - consensus_value)**2, weights=weights)
        uncertainty = np.sqrt(weighted_variance)
        
        # Detect disputes
        threshold = self.property_variance_thresholds.get(property_name, 0.1)
        is_disputed = uncertainty > threshold
        
        source_spread = np.max(values) - np.min(values)
        
        result = {
            'consensus_value': consensus_value,
            'uncertainty': uncertainty,
            'num_sources': len(values),
            'weight_sum': weights.sum(),
            'is_disputed': is_disputed,
            'source_spread': source_spread,
            'individual_values': list(zip(values, weights)),
            'threshold': threshold
        }
        
        if is_disputed:
            logger.warning(f"Property {property_name} disputed: "
                          f"consensus={consensus_value:.3f}±{uncertainty:.3f}, "
                          f"spread={source_spread:.3f}, threshold={threshold:.3f}")
        
        return result
    
    def update_source_reliability(self, doi: str, feedback_score: float):
        """Update source reliability based on user feedback"""
        current_reliability = self.source_reliability.get(doi, 1.0)
        
        # Exponential moving average update
        alpha = 0.1  # Learning rate
        new_reliability = (1 - alpha) * current_reliability + alpha * feedback_score
        
        self.source_reliability[doi] = max(new_reliability, 0.1)  # Floor at 0.1
        
        logger.info(f"Updated source reliability for {doi}: "
                   f"{current_reliability:.3f} -> {new_reliability:.3f}")
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate source reliability report"""
        if not self.source_reliability:
            return {'message': 'No reliability data available'}
        
        reliabilities = list(self.source_reliability.values())
        
        return {
            'num_sources': len(self.source_reliability),
            'mean_reliability': np.mean(reliabilities),
            'std_reliability': np.std(reliabilities),
            'min_reliability': np.min(reliabilities),
            'max_reliability': np.max(reliabilities),
            'top_sources': sorted(self.source_reliability.items(), 
                                key=lambda x: x[1], reverse=True)[:10],
            'bottom_sources': sorted(self.source_reliability.items(), 
                                   key=lambda x: x[1])[:10]
        }

# =====================================================================
# 8. REAL-TIME CONFLICT RESOLUTION SERVICE
# =====================================================================

class ConflictResolutionService:
    """Real-time service for handling knowledge graph updates and conflicts"""
    
    def __init__(self):
        self.consensus_resolver = ProvenanceWeightedConsensus()
        self.update_queue = []
        self.processing_lock = threading.Lock()
        self.is_running = False
        
        # Conflict detection cache
        self.conflict_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def start_service(self):
        """Start the conflict resolution service"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_updates, daemon=True)
        self.worker_thread.start()
        logger.info("Conflict resolution service started")
    
    def stop_service(self):
        """Stop the conflict resolution service"""
        self.is_running = False
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()
        logger.info("Conflict resolution service stopped")
    
    def submit_update(self, material_id: str, property_name: str, 
                     value: float, metadata: SourceMetadata, priority: int = 1):
        """Submit a property update for processing"""
        update = {
            'material_id': material_id,
            'property_name': property_name,
            'value': value,
            'metadata': metadata,
            'priority': priority,
            'timestamp': time.time()
        }
        
        with self.processing_lock:
            self.update_queue.append(update)
            # Sort by priority (higher first)
            self.update_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.debug(f"Submitted update for {material_id}.{property_name} = {value}")
    
    def _process_updates(self):
        """Background worker to process updates"""
        while self.is_running:
            try:
                with self.processing_lock:
                    if not self.update_queue:
                        continue
                    
                    update = self.update_queue.pop(0)
                
                self._handle_single_update(update)
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                time.sleep(1.0)  # Longer delay on error
    
    def _handle_single_update(self, update: Dict[str, Any]):
        """Handle a single property update"""
        material_id = update['material_id']
        property_name = update['property_name']
        new_value = update['value']
        metadata = update['metadata']
        
        # Check for existing values (this would query the knowledge graph)
        existing_values = self._get_existing_values(material_id, property_name)
        
        # Add new value
        all_values = existing_values + [(new_value, metadata)]
        
        # Resolve conflicts
        resolution = self.consensus_resolver.resolve_property_conflict(
            all_values, property_name
        )
        
        # Update knowledge graph with consensus
        self._update_knowledge_graph(material_id, property_name, resolution)
        
        # Cache conflicts for quick lookup
        cache_key = f"{material_id}_{property_name}"
        self.conflict_cache[cache_key] = {
            'resolution': resolution,
            'timestamp': time.time()
        }
        
        # Alert if disputed
        if resolution['is_disputed']:
            self._send_dispute_alert(material_id, property_name, resolution)
        
        logger.debug(f"Processed update for {material_id}.{property_name}: "
                    f"consensus={resolution['consensus_value']:.3f}±{resolution['uncertainty']:.3f}")
    
    def _get_existing_values(self, material_id: str, property_name: str) -> List[Tuple[float, SourceMetadata]]:
        """Get existing property values from knowledge graph"""
        # This would query Neo4j or other graph database
        # For now, return empty list (placeholder)
        return []
    
    def _update_knowledge_graph(self, material_id: str, property_name: str, 
                              resolution: Dict[str, Any]):
        """Update the knowledge graph with consensus value"""
        # This would update Neo4j with the new consensus value
        # Including uncertainty, source count, etc.
        pass
    
    def _send_dispute_alert(self, material_id: str, property_name: str, 
                          resolution: Dict[str, Any]):
        """Send alert for disputed properties"""
        alert_message = (
            f"DISPUTE ALERT: {material_id}.{property_name}\n"
            f"Consensus: {resolution['consensus_value']:.3f} ± {resolution['uncertainty']:.3f}\n"
            f"Sources: {resolution['num_sources']}, Spread: {resolution['source_spread']:.3f}\n"
            f"Threshold: {resolution['threshold']:.3f}"
        )
        
        logger.warning(alert_message)
        
        # In production, this would send notifications via:
        # - Slack/Teams webhook
        # - Email alert
        # - Dashboard notification
        # - Queue for human review
    
    def query_property(self, material_id: str, property_name: str) -> Dict[str, Any]:
        """Query property with conflict resolution"""
        
        # Check cache first
        cache_key = f"{material_id}_{property_name}"
        if cache_key in self.conflict_cache:
            cached = self.conflict_cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['resolution']
        
        # Query knowledge graph
        existing_values = self._get_existing_values(material_id, property_name)
        
        if not existing_values:
            return {'error': 'Property not found'}
        
        # Resolve conflicts
        resolution = self.consensus_resolver.resolve_property_conflict(
            existing_values, property_name
        )
        
        # Update cache
        self.conflict_cache[cache_key] = {
            'resolution': resolution,
            'timestamp': time.time()
        }
        
        return resolution
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        with self.processing_lock:
            queue_size = len(self.update_queue)
        
        return {
            'is_running': self.is_running,
            'queue_size': queue_size,
            'cache_size': len(self.conflict_cache),
            'total_sources': len(self.consensus_resolver.source_reliability),
            'reliability_stats': self.consensus_resolver.get_reliability_report()
        }

# =====================================================================
# 9. COMPREHENSIVE EVALUATION FRAMEWORK
# =====================================================================

class MaterialsScienceBenchmark:
    """Comprehensive evaluation framework for materials science models"""
    
    def __init__(self):
        self.benchmark_datasets = {}
        self.evaluation_metrics = {}
        self.physics_checker = PhysicsSanityChecker()
        
    def load_benchmark_datasets(self) -> Dict[str, Any]:
        """Load standard materials science benchmark datasets"""
        
        # This would load real benchmark datasets like:
        # - Materials Project test set
        # - JARVIS-DFT validation set  
        # - OQMD experimental subset
        # - Custom curated datasets
        
        # For demonstration, create synthetic benchmarks
        benchmarks = {
            'bandgap_prediction': self._create_bandgap_benchmark(),
            'formation_energy_prediction': self._create_formation_energy_benchmark(),
            'bulk_modulus_prediction': self._create_bulk_modulus_benchmark(),
            'multi_property_prediction': self._create_multi_property_benchmark()
        }
        
        self.benchmark_datasets = benchmarks
        return benchmarks
    
    def _create_bandgap_benchmark(self) -> Dict[str, Any]:
        """Create bandgap prediction benchmark"""
        np.random.seed(42)
        
        # Synthetic data representing diverse material chemistries
        n_samples = 1000
        materials = []
        
        for i in range(n_samples):
            # Simple features: composition-based
            material = {
                'material_id': f'mp-{i:05d}',
                'composition': self._generate_composition(),
                'crystal_system': np.random.choice(['cubic', 'tetragonal', 'orthorhombic', 'hexagonal']),
                'bandgap_exp': np.random.lognormal(0.5, 0.8),  # Experimental bandgap
                'bandgap_dft': None,  # To be predicted
                'formation_energy': np.random.normal(-2.0, 1.0),
                'is_stable': np.random.choice([True, False], p=[0.7, 0.3])
            }
            
            # Add some correlation between properties
            if material['formation_energy'] > 0:
                material['bandgap_exp'] *= 0.5  # Unstable materials often metallic
            
            materials.append(material)
        
        return {
            'name': 'Bandgap Prediction Benchmark',
            'description': 'Predict electronic bandgap from composition and structure',
            'materials': materials,
            'target_property': 'bandgap_exp',
            'evaluation_metrics': ['mae', 'rmse', 'r2', 'physics_validity'],
            'units': 'eV'
        }
    
    def _create_formation_energy_benchmark(self) -> Dict[str, Any]:
        """Create formation energy prediction benchmark"""
        np.random.seed(43)
        n_samples = 800
        materials = []
        
        for i in range(n_samples):
            material = {
                'material_id': f'fe-{i:05d}',
                'composition': self._generate_composition(),
                'formation_energy_exp': np.random.normal(-1.5, 1.2),
                'formation_energy_dft': None,
                'hull_distance': abs(np.random.normal(0, 0.3)),
                'volume_per_atom': np.random.uniform(10, 30)
            }
            materials.append(material)
        
        return {
            'name': 'Formation Energy Benchmark',
            'description': 'Predict formation energy from composition',
            'materials': materials,
            'target_property': 'formation_energy_exp',
            'evaluation_metrics': ['mae', 'rmse', 'r2', 'thermodynamic_consistency'],
            'units': 'eV/atom'
        }
    
    def _create_bulk_modulus_benchmark(self) -> Dict[str, Any]:
        """Create bulk modulus prediction benchmark"""
        np.random.seed(44)
        n_samples = 600
        materials = []
        
        for i in range(n_samples):
            material = {
                'material_id': f'bm-{i:05d}',
                'composition': self._generate_composition(),
                'bulk_modulus_exp': np.random.uniform(50, 400),
                'bulk_modulus_dft': None,
                'shear_modulus': np.random.uniform(30, 250),
                'density': np.random.uniform(2, 12)
            }
            
            # Ensure bulk >= shear modulus (physics constraint)
            material['shear_modulus'] = min(material['shear_modulus'], 
                                          material['bulk_modulus_exp'] * 0.9)
            
            materials.append(material)
        
        return {
            'name': 'Bulk Modulus Benchmark',
            'description': 'Predict mechanical properties from composition',
            'materials': materials,
            'target_property': 'bulk_modulus_exp',
            'evaluation_metrics': ['mae', 'rmse', 'r2', 'mechanical_consistency'],
            'units': 'GPa'
        }
    
    def _create_multi_property_benchmark(self) -> Dict[str, Any]:
        """Create multi-property prediction benchmark"""
        np.random.seed(45)
        n_samples = 500
        materials = []
        
        for i in range(n_samples):
            # Generate correlated properties
            formation_energy = np.random.normal(-1.0, 1.0)
            bandgap = max(0, np.random.normal(2.0, 1.5) - 0.5 * formation_energy)
            bulk_modulus = max(10, np.random.normal(150, 50) - 20 * formation_energy)
            
            material = {
                'material_id': f'mp-{i:05d}',
                'composition': self._generate_composition(),
                'bandgap_exp': bandgap,
                'formation_energy_exp': formation_energy,
                'bulk_modulus_exp': bulk_modulus,
                'density_exp': np.random.uniform(2, 10)
            }
            materials.append(material)
        
        return {
            'name': 'Multi-Property Benchmark',
            'description': 'Predict multiple properties simultaneously',
            'materials': materials,
            'target_properties': ['bandgap_exp', 'formation_energy_exp', 'bulk_modulus_exp'],
            'evaluation_metrics': ['mae', 'rmse', 'r2', 'cross_property_consistency'],
            'units': {'bandgap_exp': 'eV', 'formation_energy_exp': 'eV/atom', 'bulk_modulus_exp': 'GPa'}
        }
    
    def _generate_composition(self) -> Dict[str, float]:
        """Generate realistic chemical composition"""
        elements = ['Li', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'O', 'F', 'Cl', 'N', 'C']
        
        # Choose 1-4 elements
        n_elements = np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1])
        chosen_elements = np.random.choice(elements, size=n_elements, replace=False)
        
        # Generate random fractions that sum to 1
        fractions = np.random.dirichlet(np.ones(n_elements))
        
        composition = {}
        for elem, frac in zip(chosen_elements, fractions):
            composition[elem] = float(frac)
        
        return composition
    
    def evaluate_model(self, model: Any, benchmark_name: str, 
                      prediction_method: str = 'predict') -> Dict[str, Any]:
        """Evaluate a model on a specific benchmark"""
        
        if benchmark_name not in self.benchmark_datasets:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark = self.benchmark_datasets[benchmark_name]
        materials = benchmark['materials']
        
        # Make predictions
        predictions = []
        targets = []
        uncertainties = []
        
        logger.info(f"Evaluating model on {benchmark_name} ({len(materials)} samples)")
        
        for material in materials:
            try:
                # This would need proper feature extraction for real models
                # For now, use dummy predictions
                if hasattr(model, prediction_method):
                    pred_result = getattr(model, prediction_method)(material)
                    
                    if isinstance(pred_result, tuple):
                        pred, uncert = pred_result
                    else:
                        pred = pred_result
                        uncert = 0.0
                else:
                    # Dummy prediction for demonstration
                    pred = np.random.normal(2.0, 0.5)
                    uncert = np.random.uniform(0.1, 0.3)
                
                predictions.append(pred)
                uncertainties.append(uncert)
                
                # Get target value
                if 'target_property' in benchmark:
                    target = material[benchmark['target_property']]
                else:
                    # Multi-property case
                    target = [material[prop] for prop in benchmark['target_properties']]
                
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"Failed to predict for {material['material_id']}: {e}")
                continue
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        uncertainties = np.array(uncertainties)
        
        # Compute metrics
        results = self._compute_benchmark_metrics(
            predictions, targets, uncertainties, benchmark
        )
        
        # Add benchmark metadata
        results['benchmark_name'] = benchmark_name
        results['benchmark_description'] = benchmark['description']
        results['n_samples'] = len(predictions)
        results['target_property'] = benchmark.get('target_property', benchmark.get('target_properties'))
        
        return results
    
    def _compute_benchmark_metrics(self, predictions: np.ndarray, targets: np.ndarray,
                                 uncertainties: np.ndarray, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics"""
        
        results = {}
        
        # Basic regression metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(np.mean((predictions - targets)**2))
        r2 = r2_score(targets, predictions)
        
        results.update({
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_target': np.mean(targets),
            'std_target': np.std(targets),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions)
        })
        
        # Physics validity check
        if 'physics_validity' in benchmark.get('evaluation_metrics', []):
            physics_violations = 0
            for i, pred in enumerate(predictions):
                is_valid, _ = self.physics_checker.validate_candidate({
                    'predictions': {benchmark.get('target_property', 'property'): pred}
                })
                if not is_valid:
                    physics_violations += 1
            
            results['physics_violation_rate'] = physics_violations / len(predictions)
        
        # Uncertainty metrics
        if len(uncertainties) > 0 and np.any(uncertainties > 0):
            errors = np.abs(predictions - targets)
            
            # Uncertainty correlation
            uncert_error_corr = np.corrcoef(uncertainties, errors)[0, 1]
            results['uncertainty_error_correlation'] = uncert_error_corr
            
            # Coverage metrics
            for confidence in [0.68, 0.95]:
                z_score = 1.0 if confidence == 0.68 else 1.96
                in_interval = errors <= z_score * uncertainties
                results[f'coverage_{int(confidence*100)}'] = np.mean(in_interval)
        
        # Property-specific metrics
        if benchmark['name'] == 'Formation Energy Benchmark':
            # Thermodynamic consistency
            stable_mask = targets < 0  # Stable compounds
            if np.any(stable_mask):
                stable_mae = mean_absolute_error(targets[stable_mask], predictions[stable_mask])
                results['stable_compounds_mae'] = stable_mae
        
        elif benchmark['name'] == 'Bulk Modulus Benchmark':
            # Mechanical consistency (if shear modulus available)
            # This would check bulk >= shear constraint
            pass
        
        # Error distribution analysis
        errors = predictions - targets
        results.update({
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_absolute_error': np.max(np.abs(errors)),
            'error_percentiles': {
                'p10': np.percentile(np.abs(errors), 10),
                'p50': np.percentile(np.abs(errors), 50),
                'p90': np.percentile(np.abs(errors), 90),
                'p95': np.percentile(np.abs(errors), 95)
            }
        })
        
        return results
    
    def run_comprehensive_evaluation(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive evaluation across all benchmarks and models"""
        
        if not self.benchmark_datasets:
            self.load_benchmark_datasets()
        
        logger.info(f"Running comprehensive evaluation on {len(models)} models and {len(self.benchmark_datasets)} benchmarks")
        
        all_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            model_results = {}
            
            for benchmark_name in self.benchmark_datasets.keys():
                try:
                    benchmark_results = self.evaluate_model(model, benchmark_name)
                    model_results[benchmark_name] = benchmark_results
                    
                    logger.info(f"  {benchmark_name}: MAE={benchmark_results['mae']:.3f}, R²={benchmark_results['r2']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {model_name} on {benchmark_name}: {e}")
                    model_results[benchmark_name] = {'error': str(e)}
            
            all_results[model_name] = model_results
        
        # Compute cross-model statistics
        summary = self._compute_evaluation_summary(all_results)
        all_results['summary'] = summary
        
        return all_results
    
    def _compute_evaluation_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across models and benchmarks"""
        
        summary = {
            'best_models': {},
            'average_performance': {},
            'model_rankings': {}
        }
        
        # Find best model for each benchmark
        for benchmark_name in self.benchmark_datasets.keys():
            best_model = None
            best_r2 = -float('inf')
            
            benchmark_scores = {}
            
            for model_name, model_results in all_results.items():
                if model_name == 'summary':
                    continue
                    
                if benchmark_name in model_results and 'r2' in model_results[benchmark_name]:
                    r2 = model_results[benchmark_name]['r2']
                    benchmark_scores[model_name] = r2
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model_name
            
            summary['best_models'][benchmark_name] = {
                'model': best_model,
                'r2': best_r2,
                'all_scores': benchmark_scores
            }
        
        # Average performance across benchmarks
        model_avg_scores = {}
        for model_name, model_results in all_results.items():
            if model_name == 'summary':
                continue
            
            r2_scores = []
            mae_scores = []
            
            for benchmark_name, benchmark_results in model_results.items():
                if isinstance(benchmark_results, dict) and 'r2' in benchmark_results:
                    r2_scores.append(benchmark_results['r2'])
                    mae_scores.append(benchmark_results['mae'])
            
            if r2_scores:
                model_avg_scores[model_name] = {
                    'avg_r2': np.mean(r2_scores),
                    'avg_mae': np.mean(mae_scores),
                    'std_r2': np.std(r2_scores),
                    'n_benchmarks': len(r2_scores)
                }
        
        summary['average_performance'] = model_avg_scores
        
        # Rank models by average R²
        ranked_models = sorted(model_avg_scores.items(), 
                             key=lambda x: x[1]['avg_r2'], reverse=True)
        summary['model_rankings'] = {
            rank + 1: {'model': model, 'avg_r2': scores['avg_r2']}
            for rank, (model, scores) in enumerate(ranked_models)
        }
        
        return summary
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ORION MATERIALS SCIENCE MODEL EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Summary section
        if 'summary' in results:
            summary = results['summary']
            report_lines.append("EXECUTIVE SUMMARY")
            report_lines.append("-" * 50)
            
            if 'model_rankings' in summary:
                report_lines.append("Model Rankings (by average R²):")
                for rank, info in summary['model_rankings'].items():
                    report_lines.append(f"  {rank}. {info['model']}: R² = {info['avg_r2']:.3f}")
                report_lines.append("")
            
            if 'best_models' in summary:
                report_lines.append("Best Model per Benchmark:")
                for benchmark, info in summary['best_models'].items():
                    if info['model']:
                        report_lines.append(f"  {benchmark}: {info['model']} (R² = {info['r2']:.3f})")
                report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 50)
        
        for model_name, model_results in results.items():
            if model_name == 'summary':
                continue
                
            report_lines.append(f"\nModel: {model_name}")
            report_lines.append("~" * (len(model_name) + 7))
            
            for benchmark_name, benchmark_results in model_results.items():
                if isinstance(benchmark_results, dict) and 'mae' in benchmark_results:
                    report_lines.append(f"\n  {benchmark_name}:")
                    report_lines.append(f"    MAE: {benchmark_results['mae']:.4f}")
                    report_lines.append(f"    RMSE: {benchmark_results['rmse']:.4f}")
                    report_lines.append(f"    R²: {benchmark_results['r2']:.4f}")
                    
                    if 'physics_violation_rate' in benchmark_results:
                        rate = benchmark_results['physics_violation_rate']
                        report_lines.append(f"    Physics Violations: {rate:.1%}")
                    
                    if 'uncertainty_error_correlation' in benchmark_results:
                        corr = benchmark_results['uncertainty_error_correlation']
                        report_lines.append(f"    Uncertainty-Error Correlation: {corr:.3f}")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text

# =====================================================================
# EXAMPLE USAGE AND TESTING
# =====================================================================

def main():
    """Example usage of the ORION system components"""
    
    print("ORION: Complete Implementation - Running Examples...")
    print("="*60)
    
    # 1. Bottleneck Analysis Example
    print("\n1. Bottleneck Analysis")
    analyzer = BottleneckAnalyzer()
    
    # Simulate some operations
    @analyzer.monitor_performance
    def dummy_processing(data_size):
        time.sleep(0.1 * data_size)  # Simulate work
        return f"Processed {data_size} items"
    
    for i in range(5):
        result = dummy_processing(np.random.randint(1, 4))
    
    performance_report = analyzer.get_performance_report()
    print(f"Performance Report: {performance_report}")
    
    # 2. Physics Sanity Checker Example
    print("\n2. Physics Sanity Checker")
    checker = PhysicsSanityChecker()
    
    test_candidate = {
        'predictions': {
            'bandgap': 2.5,
            'density': 5.2,
            'bulk_modulus': 150.0,
            'shear_modulus': 80.0
        }
    }
    
    is_valid, errors = checker.validate_candidate(test_candidate)
    print(f"Candidate valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # 3. Uncertainty Quantifier Example
    print("\n3. Uncertainty Quantification")
    uncertainty_quantifier = UncertaintyQuantifier()
    
    # Generate synthetic data
    n_samples = 100
    true_values = np.random.normal(2.0, 1.0, n_samples)
    predictions = true_values + np.random.normal(0, 0.3, n_samples)
    uncertainties = np.random.uniform(0.1, 0.5, n_samples)
    
    # Calibrate uncertainty
    uncertainty_quantifier.calibrate_uncertainty(predictions, uncertainties, true_values)
    
    # Compute metrics
    metrics = uncertainty_quantifier.compute_uncertainty_metrics(
        predictions, uncertainties, true_values
    )
    print(f"Uncertainty Metrics: {metrics}")
    
    # 4. Conflict Resolution Example
    print("\n4. Conflict Resolution")
    resolver = ProvenanceWeightedConsensus()
    
    # Simulate conflicting property values
    property_values = [
        (2.1, SourceMetadata("10.1000/journal1", "2023-01-15", 50, 3.2, "journal")),
        (2.3, SourceMetadata("10.1000/journal2", "2022-06-10", 20, 2.1, "journal")),
        (1.9, SourceMetadata("arxiv.2301.12345", "2023-02-01", 5, 0.0, "preprint"))
    ]
    
    resolution = resolver.resolve_property_conflict(property_values, 'bandgap')
    print(f"Consensus Resolution: {resolution}")
    
    # 5. Evaluation Framework Example
    print("\n5. Evaluation Framework")
    benchmark = MaterialsScienceBenchmark()
    datasets = benchmark.load_benchmark_datasets()
    print(f"Loaded {len(datasets)} benchmark datasets")
    
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset['materials'])} materials")
    
    # 6. Uncertainty-Aware Ranking Example
    print("\n6. Uncertainty-Aware Ranking")
    ranker = UncertaintyAwareRanker(risk_aversion=1.5)
    
    # Create synthetic candidates
    candidates = []
    for i in range(10):
        candidate = {
            'predictions': {'bandgap': np.random.uniform(1.0, 3.0)},
            'uncertainties': {'bandgap': np.random.uniform(0.1, 0.5)},
            'embedding': np.random.randn(128)
        }
        candidates.append(candidate)
    
    ranked = ranker.rank_candidates(candidates, target_property='bandgap', 
                                  target_value=2.0, return_details=True)
    
    print("Top 3 ranked candidates:")
    for i, (idx, score, details) in enumerate(ranked[:3]):
        pred = candidates[idx]['predictions']['bandgap']
        uncert = candidates[idx]['uncertainties']['bandgap']
        print(f"  {i+1}. Candidate {idx}: prediction={pred:.2f}±{uncert:.2f}, score={score:.3f}")
    
    print("\nORION: Complete Implementation - Examples Completed!")
    print("="*60)

if __name__ == "__main__":
    main()
