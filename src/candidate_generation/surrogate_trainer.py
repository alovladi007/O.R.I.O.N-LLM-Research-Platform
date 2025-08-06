"""
ORION Surrogate Predictor Training Pipeline
==========================================

Training and optimization for surrogate models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Callable
from pathlib import Path
import json
import optuna
from collections import defaultdict

from .advanced_generator import GNNSurrogate, EnsembleSurrogate, UncertaintyQuantifier
from ..core.advanced_monitoring import AdvancedBottleneckAnalyzer

logger = logging.getLogger(__name__)


class SurrogatePredictorTrainer:
    """Training pipeline for surrogate predictors with ensemble diversity"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.training_history = {}
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.bottleneck_analyzer = AdvancedBottleneckAnalyzer()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_epochs': 200,
            'early_stopping_patience': 20,
            'batch_size': 32,
            'validation_split': 0.2,
            'test_split': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'n_trials': 50,  # Hyperparameter optimization trials
            'ensemble_models': ['gnn', 'rf', 'xgb'],
            'target_properties': ['bandgap', 'formation_energy', 'bulk_modulus', 'density'],
            'save_dir': 'models/surrogates'
        }
    
    def optimize_hyperparameters(self, train_loader: DataLoader, val_loader: DataLoader,
                               model_type: str = 'gnn', target_properties: List[str] = None) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        if target_properties is None:
            target_properties = self.config['target_properties']
        
        def objective(trial):
            # Suggest hyperparameters
            if model_type == 'gnn':
                params = {
                    'num_node_features': train_loader.dataset[0].x.size(1),
                    'hidden_dim': trial.suggest_int('hidden_dim', 64, 512, step=64),
                    'num_layers': trial.suggest_int('num_layers', 2, 5),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                    'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-3),
                    'uncertainty_mode': trial.suggest_categorical('uncertainty_mode', ['dropout', 'both'])
                }
                
                # Create model
                model = GNNSurrogate(
                    num_node_features=params['num_node_features'],
                    hidden_dim=params['hidden_dim'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    uncertainty_mode=params['uncertainty_mode']
                ).to(self.config['device'])
                
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
                
                # Train for a few epochs
                val_losses = []
                for epoch in range(20):  # Quick evaluation
                    train_loss = self._train_epoch(model, train_loader, optimizer, target_properties)
                    val_loss = self._validate_epoch(model, val_loader, target_properties)
                    val_losses.append(val_loss)
                    
                    # Early stopping for bad hyperparameters
                    if epoch > 5 and val_loss > val_losses[0] * 2:
                        break
                
                return min(val_losses)
            
            else:
                # Implement other model types
                raise NotImplementedError(f"Model type {model_type} not implemented")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        study.optimize(objective, n_trials=self.config['n_trials'])
        
        best_params = study.best_params
        logger.info(f"Best parameters found: {best_params}")
        
        return best_params
    
    @AdvancedBottleneckAnalyzer.monitor_performance
    def train_model(self, model_type: str, train_loader: DataLoader, val_loader: DataLoader,
                   target_properties: List[str], best_params: Dict[str, Any] = None) -> Any:
        """Train a single model with best parameters"""
        
        if model_type == 'gnn':
            model = GNNSurrogate(
                num_node_features=best_params['num_node_features'],
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout'],
                uncertainty_mode=best_params.get('uncertainty_mode', 'both')
            ).to(self.config['device'])
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=best_params['learning_rate'],
                weight_decay=best_params.get('weight_decay', 1e-4)
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
            
            # Training history
            history = {
                'train_loss': [], 
                'val_loss': [], 
                'learning_rate': [],
                'property_losses': {prop: [] for prop in target_properties}
            }
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            logger.info(f"Training {model_type} model...")
            
            for epoch in range(self.config['max_epochs']):
                # Training
                train_loss = self._train_epoch(model, train_loader, optimizer, target_properties)
                
                # Validation
                val_loss = self._validate_epoch(model, val_loader, target_properties)
                
                # Update metrics
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
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                              f"Val Loss={val_loss:.4f}, LR={current_lr:.6f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Store training history
            self.training_history[model_type] = history
            
            return model
        
        else:
            # Implement other model types
            raise NotImplementedError(f"Model type {model_type} not implemented")
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, target_properties: List[str]) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0
        n_batches = 0
        
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
                    if hasattr(model, 'uncertainty_heads') and prop in model.uncertainty_heads:
                        # Negative log-likelihood with learnable variance
                        log_var = model.uncertainty_heads[prop](model.last_features).squeeze(-1)
                        nll_loss = 0.5 * (torch.exp(-log_var) * (pred - target)**2 + log_var)
                        loss += nll_loss.mean() * 0.1  # Weight the uncertainty loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                       target_properties: List[str]) -> float:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0
        n_batches = 0
        
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
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train_ensemble(self, dataset: Any, target_properties: List[str] = None) -> EnsembleSurrogate:
        """Train ensemble of diverse models"""
        
        if target_properties is None:
            target_properties = self.config['target_properties']
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        
        # Train individual models
        ensemble = EnsembleSurrogate()
        
        for model_type in self.config['ensemble_models']:
            logger.info(f"Training {model_type} model...")
            
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(
                train_loader, val_loader, model_type, target_properties
            )
            
            # Train with best parameters
            model = self.train_model(
                model_type, train_loader, val_loader, target_properties, best_params
            )
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, test_loader, target_properties)
            logger.info(f"{model_type} test metrics: {test_metrics}")
            
            # Add to ensemble with weight based on performance
            weight = 1.0 / (test_metrics['mae'] + 0.1)  # Higher weight for lower error
            ensemble.add_model(model_type, model, weight)
        
        # Calibrate uncertainty estimates
        self._calibrate_ensemble_uncertainty(ensemble, test_loader, target_properties)
        
        return ensemble
    
    def evaluate_model(self, model: Any, test_loader: DataLoader, 
                      target_properties: List[str]) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        
        model.eval()
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)
        all_uncertainties = defaultdict(list)
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.config['device'])
                
                # Get predictions with uncertainty
                if hasattr(model, 'forward'):
                    predictions, uncertainties = model(batch, return_uncertainty=True, mc_samples=50)
                else:
                    predictions = model.predict(batch)
                    uncertainties = {prop: torch.zeros_like(pred) for prop, pred in predictions.items()}
                
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
            if prop in all_predictions:
                y_true = np.array(all_targets[prop])
                y_pred = np.array(all_predictions[prop])
                y_uncert = np.array(all_uncertainties[prop])
                
                from sklearn.metrics import mean_absolute_error, r2_score
                
                metrics[f'{prop}_mae'] = mean_absolute_error(y_true, y_pred)
                metrics[f'{prop}_rmse'] = np.sqrt(np.mean((y_true - y_pred)**2))
                metrics[f'{prop}_r2'] = r2_score(y_true, y_pred)
                
                # Uncertainty metrics
                in_bounds = np.abs(y_true - y_pred) <= 2 * y_uncert
                metrics[f'{prop}_coverage'] = np.mean(in_bounds)
                metrics[f'{prop}_mean_uncertainty'] = np.mean(y_uncert)
        
        # Overall metrics
        all_maes = [v for k, v in metrics.items() if k.endswith('_mae')]
        metrics['mae'] = np.mean(all_maes) if all_maes else 0.0
        
        return metrics
    
    def _calibrate_ensemble_uncertainty(self, ensemble: EnsembleSurrogate, 
                                      test_loader: DataLoader,
                                      target_properties: List[str]):
        """Calibrate uncertainty estimates for ensemble"""
        
        logger.info("Calibrating ensemble uncertainty estimates...")
        
        # Collect predictions and true values
        all_predictions = defaultdict(list)
        all_uncertainties = defaultdict(list)
        all_targets = defaultdict(list)
        
        for batch in test_loader:
            predictions, uncertainties = ensemble.predict(batch, return_uncertainty=True)
            
            for prop in target_properties:
                if hasattr(batch, prop):
                    all_predictions[prop].extend(predictions[prop])
                    all_uncertainties[prop].extend(uncertainties[prop])
                    all_targets[prop].extend(getattr(batch, prop).numpy())
        
        # Calibrate each property
        for prop in target_properties:
            if prop in all_predictions:
                y_pred = np.array(all_predictions[prop])
                y_uncert = np.array(all_uncertainties[prop])
                y_true = np.array(all_targets[prop])
                
                self.uncertainty_quantifier.calibrate_uncertainty(
                    y_pred, y_uncert, y_true, method='isotonic'
                )
    
    def save_models(self, ensemble: EnsembleSurrogate, save_dir: Optional[str] = None):
        """Save trained models"""
        if save_dir is None:
            save_dir = self.config['save_dir']
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for name, model in ensemble.models.items():
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), save_path / f"{name}_model.pt")
            else:
                # Save sklearn models
                import joblib
                joblib.dump(model, save_path / f"{name}_model.pkl")
        
        # Save ensemble metadata
        metadata = {
            'model_weights': ensemble.model_weights,
            'training_history': self.training_history,
            'config': self.config
        }
        
        with open(save_path / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {save_path}")
    
    def load_models(self, load_dir: str) -> EnsembleSurrogate:
        """Load trained models"""
        load_path = Path(load_dir)
        
        # Load metadata
        with open(load_path / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        ensemble = EnsembleSurrogate()
        
        # Load individual models
        for model_type in metadata['model_weights'].keys():
            if model_type == 'gnn':
                # Recreate model architecture
                model = GNNSurrogate(
                    num_node_features=100,  # This should be saved in metadata
                    hidden_dim=256,
                    num_layers=3,
                    dropout=0.1,
                    uncertainty_mode='both'
                )
                model.load_state_dict(torch.load(load_path / f"{model_type}_model.pt"))
                model.eval()
            else:
                # Load sklearn models
                import joblib
                model = joblib.load(load_path / f"{model_type}_model.pkl")
            
            ensemble.add_model(model_type, model, metadata['model_weights'][model_type])
        
        logger.info(f"Models loaded from {load_path}")
        return ensemble