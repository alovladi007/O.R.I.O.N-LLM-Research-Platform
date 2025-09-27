"""
ORION Comprehensive Evaluation Framework
=======================================

Benchmarking and evaluation for materials science models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MaterialsScienceBenchmark:
    """Comprehensive evaluation framework for materials science models"""
    
    def __init__(self):
        self.benchmark_datasets = {}
        self.evaluation_metrics = {}
        self.physics_checker = None  # Will be injected
        
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
            'multi_property_prediction': self._create_multi_property_benchmark(),
            'stability_prediction': self._create_stability_benchmark(),
            'synthesis_prediction': self._create_synthesis_benchmark()
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
                'density_exp': np.random.uniform(2, 10),
                'melting_point_exp': np.random.uniform(500, 3000),
                'thermal_conductivity_exp': np.random.lognormal(2, 1)
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
    
    def _create_stability_benchmark(self) -> Dict[str, Any]:
        """Create stability prediction benchmark"""
        np.random.seed(46)
        n_samples = 1200
        materials = []
        
        for i in range(n_samples):
            formation_energy = np.random.normal(-1.0, 1.5)
            hull_distance = max(0, formation_energy + np.random.normal(0.5, 0.3))
            
            material = {
                'material_id': f'stab-{i:05d}',
                'composition': self._generate_composition(),
                'formation_energy': formation_energy,
                'hull_distance': hull_distance,
                'is_stable': hull_distance < 0.025,  # 25 meV/atom threshold
                'decomposition_products': self._generate_decomposition() if hull_distance > 0.1 else None
            }
            materials.append(material)
        
        return {
            'name': 'Stability Prediction Benchmark',
            'description': 'Predict thermodynamic stability and decomposition',
            'materials': materials,
            'target_property': 'is_stable',
            'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'task_type': 'classification'
        }
    
    def _create_synthesis_benchmark(self) -> Dict[str, Any]:
        """Create synthesis prediction benchmark"""
        np.random.seed(47)
        n_samples = 800
        materials = []
        
        for i in range(n_samples):
            material = {
                'material_id': f'syn-{i:05d}',
                'composition': self._generate_composition(),
                'synthesis_temperature': np.random.uniform(300, 1500),
                'synthesis_pressure': np.random.choice([1, 10, 100, 1000]),  # atm
                'synthesis_method': np.random.choice(['solid_state', 'sol_gel', 'hydrothermal', 'cvd', 'flux']),
                'synthesis_success': np.random.choice([True, False], p=[0.6, 0.4]),
                'synthesis_time': np.random.lognormal(3, 1)  # hours
            }
            materials.append(material)
        
        return {
            'name': 'Synthesis Prediction Benchmark',
            'description': 'Predict synthesis conditions and success',
            'materials': materials,
            'target_properties': ['synthesis_temperature', 'synthesis_success'],
            'evaluation_metrics': ['mae', 'accuracy', 'condition_accuracy'],
            'task_type': 'mixed'
        }
    
    def _generate_composition(self) -> Dict[str, float]:
        """Generate realistic chemical composition"""
        elements = ['Li', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 
                   'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                   'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 
                   'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 
                   'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 
                   'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 
                   'Pb', 'Bi', 'O', 'F', 'Cl', 'N', 'C', 'B', 'H']
        
        # Choose 1-4 elements
        n_elements = np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1])
        chosen_elements = np.random.choice(elements, size=n_elements, replace=False)
        
        # Generate random fractions that sum to 1
        fractions = np.random.dirichlet(np.ones(n_elements))
        
        composition = {}
        for elem, frac in zip(chosen_elements, fractions):
            composition[elem] = float(frac)
        
        return composition
    
    def _generate_decomposition(self) -> List[str]:
        """Generate plausible decomposition products"""
        products = ['oxide', 'carbide', 'nitride', 'sulfide', 'halide', 'element']
        n_products = np.random.choice([2, 3, 4], p=[0.5, 0.4, 0.1])
        return list(np.random.choice(products, size=n_products, replace=False))
    
    def evaluate_model(self, model: Any, benchmark_name: str, 
                      prediction_method: str = 'predict') -> Dict[str, Any]:
        """Evaluate a model on a specific benchmark"""
        
        if benchmark_name not in self.benchmark_datasets:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark = self.benchmark_datasets[benchmark_name]
        materials = benchmark['materials']
        
        # Prepare data
        X_data = self._prepare_features(materials, benchmark_name)
        
        # Get predictions
        start_time = time.time()
        
        if hasattr(model, prediction_method):
            predictions = getattr(model, prediction_method)(X_data)
        else:
            raise ValueError(f"Model does not have method: {prediction_method}")
        
        prediction_time = time.time() - start_time
        
        # Evaluate based on task type
        if benchmark.get('task_type') == 'classification':
            results = self._evaluate_classification(predictions, materials, benchmark)
        elif benchmark.get('task_type') == 'mixed':
            results = self._evaluate_mixed(predictions, materials, benchmark)
        else:
            results = self._evaluate_regression(predictions, materials, benchmark)
        
        # Add timing and metadata
        results['prediction_time'] = prediction_time
        results['num_samples'] = len(materials)
        results['benchmark_name'] = benchmark_name
        
        # Physics validation if available
        if self.physics_checker and 'physics_validity' in benchmark['evaluation_metrics']:
            physics_results = self._evaluate_physics_validity(predictions, materials)
            results['physics_validity'] = physics_results
        
        return results
    
    def _prepare_features(self, materials: List[Dict], benchmark_name: str) -> Any:
        """Prepare feature matrix from materials data"""
        # This is a simplified version - real implementation would extract
        # composition features, structure features, etc.
        
        features = []
        for material in materials:
            feature_vec = []
            
            # Composition features (simplified)
            if 'composition' in material:
                comp = material['composition']
                # Add element fractions (padding to fixed size)
                element_fracs = [comp.get(elem, 0.0) for elem in ['O', 'Si', 'Al', 'Fe', 'Ca']]
                feature_vec.extend(element_fracs)
            
            # Other features
            for key in ['volume_per_atom', 'density', 'formation_energy']:
                if key in material:
                    feature_vec.append(material[key])
                else:
                    feature_vec.append(0.0)
            
            features.append(feature_vec)
        
        return np.array(features)
    
    def _evaluate_regression(self, predictions: np.ndarray, 
                           materials: List[Dict], 
                           benchmark: Dict) -> Dict[str, float]:
        """Evaluate regression predictions"""
        
        if 'target_property' in benchmark:
            # Single property
            target_prop = benchmark['target_property']
            y_true = np.array([m[target_prop] for m in materials])
            
            results = {
                'mae': mean_absolute_error(y_true, predictions),
                'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
                'r2': r2_score(y_true, predictions),
                'mape': np.mean(np.abs((y_true - predictions) / (y_true + 1e-10))) * 100
            }
            
        else:
            # Multiple properties
            results = {}
            for i, prop in enumerate(benchmark['target_properties']):
                y_true = np.array([m[prop] for m in materials])
                y_pred = predictions[:, i] if predictions.ndim > 1 else predictions
                
                results[f'{prop}_mae'] = mean_absolute_error(y_true, y_pred)
                results[f'{prop}_rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                results[f'{prop}_r2'] = r2_score(y_true, y_pred)
        
        return results
    
    def _evaluate_classification(self, predictions: np.ndarray,
                               materials: List[Dict],
                               benchmark: Dict) -> Dict[str, float]:
        """Evaluate classification predictions"""
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        target_prop = benchmark['target_property']
        y_true = np.array([m[target_prop] for m in materials])
        
        # Convert to binary if needed
        if predictions.dtype == float and predictions.max() <= 1.0:
            y_pred = (predictions > 0.5).astype(int)
            y_pred_proba = predictions
        else:
            y_pred = predictions
            y_pred_proba = None
        
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            results['auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return results
    
    def _evaluate_mixed(self, predictions: Dict[str, np.ndarray],
                       materials: List[Dict],
                       benchmark: Dict) -> Dict[str, float]:
        """Evaluate mixed regression/classification tasks"""
        
        results = {}
        
        for prop in benchmark['target_properties']:
            y_true = np.array([m[prop] for m in materials])
            y_pred = predictions[prop]
            
            if isinstance(y_true[0], bool) or len(np.unique(y_true)) < 10:
                # Classification metric
                from sklearn.metrics import accuracy_score
                results[f'{prop}_accuracy'] = accuracy_score(y_true, y_pred > 0.5)
            else:
                # Regression metric
                results[f'{prop}_mae'] = mean_absolute_error(y_true, y_pred)
                results[f'{prop}_r2'] = r2_score(y_true, y_pred)
        
        return results
    
    def _evaluate_physics_validity(self, predictions: Any, 
                                 materials: List[Dict]) -> Dict[str, float]:
        """Evaluate physics validity of predictions"""
        
        if not self.physics_checker:
            return {'physics_validity_rate': 0.0}
        
        valid_count = 0
        total_count = len(materials)
        violations = []
        
        for i, material in enumerate(materials):
            # Create candidate with predictions
            if isinstance(predictions, dict):
                pred_dict = {k: v[i] for k, v in predictions.items()}
            else:
                pred_dict = {'predicted_value': predictions[i]}
            
            candidate = {
                'predictions': pred_dict,
                'composition': material.get('composition', {})
            }
            
            is_valid, errors = self.physics_checker.validate_candidate(candidate)
            if is_valid:
                valid_count += 1
            else:
                violations.extend(errors)
        
        # Count violation types
        violation_counts = {}
        for violation in violations:
            vtype = violation.split(':')[0]
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        return {
            'physics_validity_rate': valid_count / total_count,
            'total_violations': len(violations),
            'violation_types': violation_counts
        }
    
    def compare_models(self, models: Dict[str, Any], 
                      benchmark_names: List[str] = None) -> pd.DataFrame:
        """Compare multiple models across benchmarks"""
        
        if benchmark_names is None:
            benchmark_names = list(self.benchmark_datasets.keys())
        
        results = []
        
        for model_name, model in models.items():
            for benchmark_name in benchmark_names:
                try:
                    eval_results = self.evaluate_model(model, benchmark_name)
                    eval_results['model_name'] = model_name
                    results.append(eval_results)
                except Exception as e:
                    logger.error(f"Error evaluating {model_name} on {benchmark_name}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Aggregate metrics
        metric_cols = [col for col in df.columns if col not in 
                      ['model_name', 'benchmark_name', 'prediction_time', 'num_samples']]
        
        summary = df.groupby(['model_name', 'benchmark_name'])[metric_cols].mean()
        
        return summary
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            metrics: List[str] = None,
                            save_path: Optional[str] = None):
        """Plot model comparison results"""
        
        if metrics is None:
            metrics = ['mae', 'r2', 'physics_validity_rate']
        
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6*len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            # Prepare data for plotting
            plot_data = comparison_df[metric].unstack(level=0)
            
            # Create bar plot
            plot_data.plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xlabel('Benchmark')
            ax.set_ylabel(metric)
            ax.legend(title='Model')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return fig