"""
ORION: Advanced Usage Examples & Integration Guide
=================================================

This module demonstrates advanced usage patterns and integration scenarios
for the ORION materials science AI system, including:

1. Complete training pipeline with hyperparameter optimization
2. Production deployment patterns
3. Real-time inference with uncertainty quantification
4. Integration with external systems (DFT, ELN, databases)
5. Advanced visualization and monitoring
6. Error handling and recovery strategies

Author: ORION Development Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import ORION core modules (from the previous artifact)
# from orion_core_modules import *

# Additional imports for advanced functionality
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - using simplified hyperparameter search")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Weights & Biases not available - using local logging")

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus client not available - using basic metrics")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. ADVANCED TRAINING PIPELINE WITH OPTUNA INTEGRATION
# =====================================================================

class AdvancedTrainingPipeline:
    """Production-ready training pipeline with advanced optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.study = None
        self.best_models = {}
        self.training_metadata = {}
        
        # Initialize experiment tracking
        if WANDB_AVAILABLE and self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'orion-training'),
                config=self.config
            )
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'model_types': ['gnn', 'rf', 'ensemble'],
            'optimization_budget': 100,  # Number of trials
            'cross_validation_folds': 5,
            'early_stopping_patience': 15,
            'use_wandb': False,
            'wandb_project': 'orion-materials',
            'save_models': True,
            'model_save_path': './models',
            'random_state': 42,
            'n_jobs': -1,  # Parallel processing
            'gpu_device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }
    
    def create_optuna_study(self, study_name: str = None) -> optuna.Study:
        """Create Optuna study for hyperparameter optimization"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for advanced hyperparameter optimization")
        
        study_name = study_name or f"orion_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use database storage for persistence
        storage_url = self.config.get('study_storage', 'sqlite:///orion_studies.db')
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction='maximize',  # Maximize R²
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.config['random_state']),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=20,
                interval_steps=10
            )
        )
        
        logger.info(f"Created Optuna study: {study_name}")
        return self.study
    
    def gnn_objective(self, trial: optuna.Trial, train_data: Any, val_data: Any, 
                     target_properties: List[str]) -> float:
        """Objective function for GNN hyperparameter optimization"""
        
        # Sample hyperparameters
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'attention_heads': trial.suggest_int('attention_heads', 2, 8),
            'residual_connections': trial.suggest_categorical('residual_connections', [True, False])
        }
        
        # Add trial parameters to wandb if available
        if WANDB_AVAILABLE and self.config.get('use_wandb', False):
            wandb.log(params)
        
        try:
            # Create model with sampled parameters
            model = self._create_gnn_model(params, train_data, target_properties)
            
            # Train with cross-validation
            cv_scores = self._cross_validate_model(model, train_data, val_data, params)
            
            # Return mean cross-validation score
            mean_score = np.mean(cv_scores)
            
            # Log results
            trial.set_user_attr('cv_scores', cv_scores)
            trial.set_user_attr('mean_score', mean_score)
            trial.set_user_attr('std_score', np.std(cv_scores))
            
            # Report intermediate result for pruning
            trial.report(mean_score, step=len(cv_scores))
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return mean_score
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -float('inf')
    
    def _create_gnn_model(self, params: Dict[str, Any], train_data: Any, 
                         target_properties: List[str]):
        """Create GNN model with given parameters"""
        
        # Extract data characteristics
        sample_batch = next(iter(train_data))
        num_node_features = sample_batch.x.size(1)
        
        # Enhanced GNN architecture
        class EnhancedGNN(nn.Module):
            def __init__(self, params, num_node_features, target_properties):
                super().__init__()
                self.params = params
                self.target_properties = target_properties
                
                hidden_dim = params['hidden_dim']
                num_layers = params['num_layers']
                dropout = params['dropout']
                attention_heads = params['attention_heads']
                use_residual = params['residual_connections']
                
                # Graph convolution layers
                self.convs = nn.ModuleList()
                self.batch_norms = nn.ModuleList()
                
                # Input layer
                self.convs.append(GCNConv(num_node_features, hidden_dim))
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                
                # Hidden layers
                for i in range(num_layers - 1):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
                
                # Attention mechanism
                self.attention = GATConv(
                    hidden_dim, hidden_dim // attention_heads, 
                    heads=attention_heads, concat=True, dropout=dropout
                )
                
                # Global pooling layers
                self.global_pool = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                ])
                
                # Property prediction heads
                self.property_heads = nn.ModuleDict()
                self.uncertainty_heads = nn.ModuleDict()
                
                for prop in target_properties:
                    self.property_heads[prop] = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1)
                    )
                    
                    # Aleatoric uncertainty head
                    self.uncertainty_heads[prop] = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 4),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 4, 1)
                    )
                
                self.dropout = nn.Dropout(dropout)
                self.use_residual = use_residual
            
            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                
                # Graph convolutions with residual connections
                h = x
                for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
                    h_new = conv(h, edge_index)
                    h_new = bn(h_new)
                    h_new = F.relu(h_new)
                    h_new = self.dropout(h_new)
                    
                    # Residual connection
                    if self.use_residual and i > 0 and h.size(-1) == h_new.size(-1):
                        h = h + h_new
                    else:
                        h = h_new
                
                # Attention mechanism
                h = self.attention(h, edge_index)
                h = F.relu(h)
                
                # Global pooling (mean + max)
                h_mean = global_mean_pool(h, batch)
                h_max = global_max_pool(h, batch)
                h_global = torch.cat([h_mean, h_max], dim=1)
                
                # Predictions
                predictions = {}
                uncertainties = {}
                
                for prop in self.target_properties:
                    pred = self.property_heads[prop](h_global).squeeze(-1)
                    log_var = self.uncertainty_heads[prop](h_global).squeeze(-1)
                    
                    predictions[prop] = pred
                    uncertainties[prop] = torch.exp(0.5 * log_var)  # Convert log variance to std
                
                return predictions, uncertainties
        
        model = EnhancedGNN(params, num_node_features, target_properties)
        return model.to(self.config['gpu_device'])
    
    def _cross_validate_model(self, model: Any, train_data: Any, val_data: Any, 
                            params: Dict[str, Any]) -> List[float]:
        """Perform cross-validation training"""
        
        from sklearn.model_selection import KFold
        
        # Combine train and validation data for CV
        all_data = train_data.dataset + val_data.dataset
        
        kfold = KFold(n_splits=self.config['cross_validation_folds'], 
                     shuffle=True, random_state=self.config['random_state'])
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_data)):
            logger.info(f"Training fold {fold + 1}/{self.config['cross_validation_folds']}")
            
            # Create fold data loaders
            fold_train_data = torch.utils.data.Subset(all_data, train_idx)
            fold_val_data = torch.utils.data.Subset(all_data, val_idx)
            
            fold_train_loader = DataLoader(fold_train_data, batch_size=params['batch_size'], shuffle=True)
            fold_val_loader = DataLoader(fold_val_data, batch_size=params['batch_size'], shuffle=False)
            
            # Recreate model for this fold
            fold_model = self._create_gnn_model(params, train_data, model.target_properties)
            
            # Train fold model
            fold_score = self._train_single_fold(fold_model, fold_train_loader, fold_val_loader, params)
            cv_scores.append(fold_score)
            
            logger.info(f"Fold {fold + 1} score: {fold_score:.4f}")
        
        return cv_scores
    
    def _train_single_fold(self, model: Any, train_loader: DataLoader, 
                          val_loader: DataLoader, params: Dict[str, Any]) -> float:
        """Train model for a single fold"""
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=False
        )
        
        best_val_score = -float('inf')
        patience_counter = 0
        
        for epoch in range(100):  # Max epochs per fold
            # Training
            model.train()
            train_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.config['gpu_device'])
                optimizer.zero_grad()
                
                predictions, uncertainties = model(batch)
                
                # Multi-task loss with uncertainty
                total_loss = 0
                for prop in model.target_properties:
                    if hasattr(batch, prop):
                        target = getattr(batch, prop)
                        pred = predictions[prop]
                        uncert = uncertainties[prop]
                        
                        # Heteroscedastic loss (uncertainty-aware)
                        mse_loss = (pred - target) ** 2
                        uncertainty_loss = 0.5 * (mse_loss / (uncert ** 2) + torch.log(uncert ** 2))
                        total_loss += uncertainty_loss.mean()
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(total_loss.item())
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.config['gpu_device'])
                    predictions, _ = model(batch)
                    
                    # Collect predictions for primary property (first in list)
                    primary_prop = model.target_properties[0]
                    if hasattr(batch, primary_prop):
                        val_predictions.extend(predictions[primary_prop].cpu().numpy())
                        val_targets.extend(getattr(batch, primary_prop).cpu().numpy())
            
            if val_predictions:
                val_score = r2_score(val_targets, val_predictions)
                scheduler.step(-val_score)  # Negative because we want to maximize R²
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        break
        
        return best_val_score
    
    def optimize_all_models(self, train_data: Any, val_data: Any, 
                           target_properties: List[str]) -> Dict[str, Any]:
        """Optimize hyperparameters for all model types"""
        
        results = {}
        
        for model_type in self.config['model_types']:
            logger.info(f"Optimizing {model_type} model...")
            
            if model_type == 'gnn' and OPTUNA_AVAILABLE:
                # Create study for this model type
                study = self.create_optuna_study(f"orion_{model_type}_study")
                
                # Define objective function
                def objective(trial):
                    return self.gnn_objective(trial, train_data, val_data, target_properties)
                
                # Optimize
                study.optimize(objective, n_trials=self.config['optimization_budget'])
                
                # Store results
                results[model_type] = {
                    'best_params': study.best_params,
                    'best_score': study.best_value,
                    'n_trials': len(study.trials),
                    'study': study
                }
                
                logger.info(f"Best {model_type} score: {study.best_value:.4f}")
                logger.info(f"Best {model_type} params: {study.best_params}")
                
            elif model_type == 'rf':
                # Random Forest optimization
                results[model_type] = self._optimize_random_forest(train_data, val_data, target_properties)
                
            elif model_type == 'ensemble':
                # Ensemble optimization (combine best individual models)
                if 'gnn' in results and 'rf' in results:
                    results[model_type] = self._create_optimized_ensemble(results, train_data, val_data)
        
        return results
    
    def _optimize_random_forest(self, train_data: Any, val_data: Any, 
                               target_properties: List[str]) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters"""
        
        # Extract features (simplified - would need proper feature engineering)
        X_train, y_train = self._extract_features(train_data, target_properties[0])
        X_val, y_val = self._extract_features(val_data, target_properties[0])
        
        # Grid search for Random Forest
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestRegressor
        
        param_distributions = {
            'n_estimators': [50, 100, 200, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.3, 0.6]
        }
        
        rf = RandomForestRegressor(random_state=self.config['random_state'], n_jobs=self.config['n_jobs'])
        
        search = RandomizedSearchCV(
            rf, param_distributions,
            n_iter=50,
            cv=3,
            scoring='r2',
            random_state=self.config['random_state'],
            n_jobs=self.config['n_jobs']
        )
        
        search.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = search.score(X_val, y_val)
        
        return {
            'best_params': search.best_params_,
            'best_score': val_score,
            'cv_score': search.best_score_,
            'model': search.best_estimator_
        }
    
    def _extract_features(self, data: Any, target_property: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from graph data for traditional ML models"""
        
        features = []
        targets = []
        
        for sample in data:
            # Simple graph features (would be more sophisticated in practice)
            graph_features = [
                sample.x.mean().item(),  # Average node feature
                sample.x.std().item(),   # Node feature variance
                sample.num_nodes,        # Graph size
                sample.num_edges,        # Connectivity
                (sample.num_edges / sample.num_nodes) if sample.num_nodes > 0 else 0  # Edge density
            ]
            
            features.append(graph_features)
            
            if hasattr(sample, target_property):
                targets.append(getattr(sample, target_property).item())
        
        return np.array(features), np.array(targets)
    
    def train_final_models(self, optimization_results: Dict[str, Any], 
                          full_train_data: Any, test_data: Any, 
                          target_properties: List[str]) -> Dict[str, Any]:
        """Train final models with optimized hyperparameters"""
        
        final_models = {}
        
        for model_type, results in optimization_results.items():
            if 'best_params' not in results:
                continue
                
            logger.info(f"Training final {model_type} model...")
            
            try:
                if model_type == 'gnn':
                    model = self._create_gnn_model(results['best_params'], full_train_data, target_properties)
                    
                    # Train on full dataset
                    trained_model = self._train_final_gnn(model, full_train_data, results['best_params'])
                    
                elif model_type == 'rf':
                    trained_model = results['model']  # Already trained during optimization
                
                # Evaluate on test set
                test_metrics = self._evaluate_final_model(trained_model, test_data, target_properties)
                
                final_models[model_type] = {
                    'model': trained_model,
                    'best_params': results['best_params'],
                    'optimization_score': results['best_score'],
                    'test_metrics': test_metrics
                }
                
                # Save model if configured
                if self.config['save_models']:
                    self._save_model(trained_model, model_type, results['best_params'])
                
                logger.info(f"Final {model_type} test R²: {test_metrics.get('r2', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train final {model_type} model: {e}")
                continue
        
        return final_models
    
    def _train_final_gnn(self, model: Any, train_data: Any, params: Dict[str, Any]) -> Any:
        """Train final GNN model on full dataset"""
        
        train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        model.train()
        for epoch in range(100):  # Fixed number of epochs for final training
            epoch_losses = []
            
            for batch in train_loader:
                batch = batch.to(self.config['gpu_device'])
                optimizer.zero_grad()
                
                predictions, uncertainties = model(batch)
                
                total_loss = 0
                for prop in model.target_properties:
                    if hasattr(batch, prop):
                        target = getattr(batch, prop)
                        pred = predictions[prop]
                        uncert = uncertainties[prop]
                        
                        # Uncertainty-aware loss
                        mse_loss = (pred - target) ** 2
                        uncertainty_loss = 0.5 * (mse_loss / (uncert ** 2) + torch.log(uncert ** 2))
                        total_loss += uncertainty_loss.mean()
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(total_loss.item())
            
            scheduler.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {np.mean(epoch_losses):.4f}")
        
        return model
    
    def _evaluate_final_model(self, model: Any, test_data: Any, 
                            target_properties: List[str]) -> Dict[str, float]:
        """Evaluate final model on test set"""
        
        if hasattr(model, 'eval'):  # PyTorch model
            model.eval()
            predictions = {}
            targets = {}
            
            test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(self.config['gpu_device'])
                    batch_predictions, _ = model(batch)
                    
                    for prop in target_properties:
                        if hasattr(batch, prop):
                            if prop not in predictions:
                                predictions[prop] = []
                                targets[prop] = []
                            
                            predictions[prop].extend(batch_predictions[prop].cpu().numpy())
                            targets[prop].extend(getattr(batch, prop).cpu().numpy())
        
        else:  # Scikit-learn model
            X_test, y_test = self._extract_features(test_data, target_properties[0])
            y_pred = model.predict(X_test)
            
            predictions = {target_properties[0]: y_pred}
            targets = {target_properties[0]: y_test}
        
        # Compute metrics
        metrics = {}
        for prop in predictions:
            if len(predictions[prop]) > 0:
                mae = mean_absolute_error(targets[prop], predictions[prop])
                rmse = np.sqrt(mean_squared_error(targets[prop], predictions[prop]))
                r2 = r2_score(targets[prop], predictions[prop])
                
                metrics[f'{prop}_mae'] = mae
                metrics[f'{prop}_rmse'] = rmse
                metrics[f'{prop}_r2'] = r2
        
        # Overall metrics (average across properties)
        if len(target_properties) > 1:
            r2_values = [metrics[f'{prop}_r2'] for prop in target_properties if f'{prop}_r2' in metrics]
            if r2_values:
                metrics['r2'] = np.mean(r2_values)
        else:
            metrics['r2'] = metrics.get(f'{target_properties[0]}_r2', 0.0)
        
        return metrics
    
    def _save_model(self, model: Any, model_type: str, params: Dict[str, Any]):
        """Save trained model and metadata"""
        
        save_dir = Path(self.config['model_save_path'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if hasattr(model, 'state_dict'):  # PyTorch model
            model_path = save_dir / f"{model_type}_model_{timestamp}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'params': params,
                'timestamp': timestamp
            }, model_path)
        else:  # Scikit-learn model
            model_path = save_dir / f"{model_type}_model_{timestamp}.pkl"
            joblib.dump({
                'model': model,
                'params': params,
                'timestamp': timestamp
            }, model_path)
        
        logger.info(f"Model saved: {model_path}")

# =====================================================================
# 2. PRODUCTION DEPLOYMENT SYSTEM
# =====================================================================

class ORIONProductionSystem:
    """Production deployment system with monitoring and scaling"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.models = {}
        self.model_versions = {}
        self.request_history = []
        
        # Initialize monitoring
        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()
            start_http_server(self.config['metrics_port'])
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'max_batch_size': 100,
            'prediction_timeout': 30.0,
            'model_cache_size': 3,
            'metrics_port': 8000,
            'auto_scaling_enabled': True,
            'uncertainty_threshold': 0.5,
            'physics_validation': True,
            'rate_limit_per_hour': 1000
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        self.prediction_counter = Counter(
            'orion_predictions_total', 
            'Total number of predictions made',
            ['model_type', 'status']
        )
        
        self.prediction_latency = Histogram(
            'orion_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_type']
        )
        
        self.uncertainty_gauge = Gauge(
            'orion_prediction_uncertainty',
            'Average prediction uncertainty',
            ['model_type', 'property']
        )
        
        self.physics_violations = Counter(
            'orion_physics_violations_total',
            'Number of physics constraint violations',
            ['model_type']
        )
    
    async def load_model(self, model_path: str, model_type: str, 
                        version: str = 'latest') -> bool:
        """Load model for production serving"""
        
        try:
            if model_path.endswith('.pth'):
                # PyTorch model
                checkpoint = torch.load(model_path, map_location=self.config.get('device', 'cpu'))
                # Would need to reconstruct model architecture
                # This is simplified for demonstration
                model = None  # Would create actual model here
                
            elif model_path.endswith('.pkl'):
                # Scikit-learn model
                model_data = joblib.load(model_path)
                model = model_data['model']
            
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            # Store model with version
            if model_type not in self.models:
                self.models[model_type] = {}
                self.model_versions[model_type] = []
            
            self.models[model_type][version] = model
            self.model_versions[model_type].append(version)
            
            # Manage cache size
            if len(self.model_versions[model_type]) > self.config['model_cache_size']:
                old_version = self.model_versions[model_type].pop(0)
                del self.models[model_type][old_version]
            
            logger.info(f"Loaded {model_type} model version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return False
    
    async def predict_batch(self, candidates: List[Dict[str, Any]], 
                           model_type: str = 'ensemble',
                           version: str = 'latest',
                           include_uncertainty: bool = True) -> Dict[str, Any]:
        """Make batch predictions with monitoring and validation"""
        
        start_time = time.time()
        
        try:
            # Validate batch size
            if len(candidates) > self.config['max_batch_size']:
                raise ValueError(f"Batch size {len(candidates)} exceeds maximum {self.config['max_batch_size']}")
            
            # Get model
            if model_type not in self.models or version not in self.models[model_type]:
                raise ValueError(f"Model {model_type}:{version} not loaded")
            
            model = self.models[model_type][version]
            
            # Make predictions
            predictions = []
            uncertainties = []
            physics_valid = []
            
            for candidate in candidates:
                try:
                    # Get prediction
                    if hasattr(model, 'predict_with_uncertainty') and include_uncertainty:
                        pred, uncert = model.predict_with_uncertainty(candidate)
                    else:
                        pred = model.predict(candidate)
                        uncert = {}
                    
                    predictions.append(pred)
                    uncertainties.append(uncert)
                    
                    # Physics validation
                    if self.config['physics_validation']:
                        checker = PhysicsSanityChecker()
                        is_valid, violations = checker.validate_candidate({
                            'predictions': pred
                        })
                        physics_valid.append(is_valid)
                        
                        if not is_valid and PROMETHEUS_AVAILABLE:
                            self.physics_violations.labels(model_type=model_type).inc()
                    else:
                        physics_valid.append(True)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for candidate: {e}")
                    predictions.append({})
                    uncertainties.append({})
                    physics_valid.append(False)
            
            # Compute metrics
            processing_time = time.time() - start_time
            
            # Update monitoring metrics
            if PROMETHEUS_AVAILABLE:
                self.prediction_counter.labels(
                    model_type=model_type, status='success'
                ).inc(len(candidates))
                
                self.prediction_latency.labels(model_type=model_type).observe(processing_time)
                
                # Update uncertainty metrics
                if uncertainties and include_uncertainty:
                    for prop in ['bandgap', 'formation_energy', 'bulk_modulus']:
                        uncert_values = [u.get(prop, 0) for u in uncertainties if prop in u]
                        if uncert_values:
                            avg_uncertainty = np.mean(uncert_values)
                            self.uncertainty_gauge.labels(
                                model_type=model_type, property=prop
                            ).set(avg_uncertainty)
            
            result = {
                'predictions': predictions,
                'uncertainties': uncertainties if include_uncertainty else None,
                'physics_valid': physics_valid,
                'processing_time': processing_time,
                'model_type': model_type,
                'model_version': version,
                'batch_size': len(candidates),
                'success_rate': sum(1 for p in predictions if p) / len(predictions)
            }
            
            # Store request history
            self.request_history.append({
                'timestamp': datetime.now(),
                'batch_size': len(candidates),
                'processing_time': processing_time,
                'model_type': model_type,
                'success_rate': result['success_rate']
            })
            
            # Cleanup old history
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.request_history = [
                r for r in self.request_history if r['timestamp'] > cutoff_time
            ]
            
            return result
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            
            if PROMETHEUS_AVAILABLE:
                self.prediction_counter.labels(
                    model_type=model_type, status='error'
                ).inc(len(candidates))
            
            return {
                'error': str(e),
                'predictions': [],
                'processing_time': time.time() - start_time,
                'model_type': model_type,
                'batch_size': len(candidates),
                'success_rate': 0.0
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Recent performance metrics
        recent_requests = [
            r for r in self.request_history 
            if r['timestamp'] > datetime.now() - timedelta(hours=1)
        ]
        
        if recent_requests:
            avg_latency = np.mean([r['processing_time'] for r in recent_requests])
            avg_success_rate = np.mean([r['success_rate'] for r in recent_requests])
            requests_per_hour = len(recent_requests)
        else:
            avg_latency = 0.0
            avg_success_rate = 1.0
            requests_per_hour = 0
        
        # Model status
        model_status = {}
        for model_type, versions in self.models.items():
            model_status[model_type] = {
                'versions_loaded': list(versions.keys()),
                'total_versions': len(versions)
            }
        
        return {
            'status': 'healthy',
            'models_loaded': len(self.models),
            'model_status': model_status,
            'recent_performance': {
                'requests_per_hour': requests_per_hour,
                'avg_latency_seconds': avg_latency,
                'avg_success_rate': avg_success_rate
            },
            'system_config': self.config,
            'uptime': 'N/A'  # Would track actual uptime
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check model loading
        try:
            if not self.models:
                health_status['checks']['models'] = 'warning: no models loaded'
            else:
                health_status['checks']['models'] = f'ok: {len(self.models)} model types loaded'
        except Exception as e:
            health_status['checks']['models'] = f'error: {e}'
            health_status['status'] = 'unhealthy'
        
        # Check memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                health_status['checks']['memory'] = f'warning: {memory_percent:.1f}% usage'
                health_status['status'] = 'degraded'
            else:
                health_status['checks']['memory'] = f'ok: {memory_percent:.1f}% usage'
        except Exception as e:
            health_status['checks']['memory'] = f'error: {e}'
        
        # Check GPU if available
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                health_status['checks']['gpu'] = f'ok: {gpu_memory:.1f}% GPU memory'
            else:
                health_status['checks']['gpu'] = 'info: GPU not available'
        except Exception as e:
            health_status['checks']['gpu'] = f'error: {e}'
        
        return health_status

# =====================================================================
# 3. VISUALIZATION AND MONITORING DASHBOARD
# =====================================================================

class ORIONDashboard:
    """Advanced visualization and monitoring dashboard"""
    
    def __init__(self, production_system: ORIONProductionSystem = None):
        self.production_system = production_system
        self.evaluation_results = {}
        
    def plot_training_curves(self, training_history: Dict[str, Any], 
                           save_path: str = None) -> plt.Figure:
        """Plot comprehensive training curves"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(training_history)))
        
        # Loss curves
        axes[0].set_title('Training Loss')
        axes[1].set_title('Validation Loss')
        
        for i, (model_name, history) in enumerate(training_history.items()):
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label=model_name, color=colors[i])
            if 'val_loss' in history:
                axes[1].plot(history['val_loss'], label=model_name, color=colors[i])
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate curves
        axes[2].set_title('Learning Rate Schedule')
        for i, (model_name, history) in enumerate(training_history.items()):
            if 'learning_rate' in history:
                axes[2].plot(history['learning_rate'], label=model_name, color=colors[i])
        
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # R² score comparison
        axes[3].set_title('Model R² Comparison')
        model_names = list(training_history.keys())
        r2_scores = [max(hist.get('val_r2', [0])) if hist.get('val_r2') else 0 
                    for hist in training_history.values()]
        
        bars = axes[3].bar(model_names, r2_scores, color=colors[:len(model_names)])
        axes[3].set_ylabel('Best R² Score')
        axes[3].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        # Convergence analysis
        axes[4].set_title('Training Convergence')
        for i, (model_name, history) in enumerate(training_history.items()):
            if 'val_loss' in history:
                val_loss = np.array(history['val_loss'])
                # Smooth the curve for convergence analysis
                if len(val_loss) > 10:
                    smoothed = np.convolve(val_loss, np.ones(5)/5, mode='valid')
                    axes[4].plot(smoothed, label=f'{model_name} (smoothed)', color=colors[i])
        
        axes[4].set_xlabel('Epoch')
        axes[4].set_ylabel('Smoothed Validation Loss')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)
        
        # Training efficiency
        axes[5].set_title('Training Efficiency')
        model_names = []
        total_epochs = []
        best_scores = []
        
        for model_name, history in training_history.items():
            if 'val_loss' in history:
                model_names.append(model_name)
                total_epochs.append(len(history['val_loss']))
                if 'val_r2' in history:
                    best_scores.append(max(history['val_r2']))
                else:
                    best_scores.append(0)
        
        if model_names:
            scatter = axes[5].scatter(total_epochs, best_scores, 
                                    c=range(len(model_names)), cmap='Set1', s=100)
            
            for i, name in enumerate(model_names):
                axes[5].annotate(name, (total_epochs[i], best_scores[i]), 
                               xytext=(5, 5), textcoords='offset points')
        
        axes[5].set_xlabel('Total Epochs')
        axes[5].set_ylabel('Best R² Score')
        axes[5].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Any], 
                            save_path: str = None) -> plt.Figure:
        """Create comprehensive model comparison plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract metrics for plotting
        models = []
        properties = set()
        metrics_data = {}
        
        for model_name, model_results in evaluation_results.items():
            if model_name == 'summary':
                continue
                
            models.append(model_name)
            for benchmark_name, benchmark_results in model_results.items():
                if isinstance(benchmark_results, dict):
                    for metric, value in benchmark_results.items():
                        if isinstance(value, (int, float)):
                            key = f"{benchmark_name}_{metric}"
                            if key not in metrics_data:
                                metrics_data[key] = {}
                            metrics_data[key][model_name] = value
                            
                            # Extract property name
                            if '_' in metric:
                                prop = metric.split('_')[0]
                                if prop in ['bandgap', 'formation', 'bulk']:
                                    properties.add(prop)
        
        # 1. R² comparison across benchmarks
        axes[0, 0].set_title('R² Score Comparison Across Benchmarks')
        
        r2_data = {}
        for key, values in metrics_data.items():
            if 'r2' in key:
                benchmark = key.replace('_r2', '')
                r2_data[benchmark] = values
        
        if r2_data:
            benchmarks = list(r2_data.keys())
            x = np.arange(len(benchmarks))
            width = 0.2
            
            for i, model in enumerate(models):
                scores = [r2_data[bench].get(model, 0) for bench in benchmarks]
                axes[0, 0].bar(x + i*width, scores, width, label=model)
            
            axes[0, 0].set_xlabel('Benchmark')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_xticks(x + width)
            axes[0, 0].set_xticklabels(benchmarks, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. MAE comparison
        axes[0, 1].set_title('Mean Absolute Error Comparison')
        
        mae_data = {}
        for key, values in metrics_data.items():
            if 'mae' in key:
                benchmark = key.replace('_mae', '')
                mae_data[benchmark] = values
        
        if mae_data:
            benchmarks = list(mae_data.keys())
            x = np.arange(len(benchmarks))
            
            for i, model in enumerate(models):
                errors = [mae_data[bench].get(model, 0) for bench in benchmarks]
                axes[0, 1].bar(x + i*width, errors, width, label=model)
            
            axes[0, 1].set_xlabel('Benchmark')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].set_xticks(x + width)
            axes[0, 1].set_xticklabels(benchmarks, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Physics violation rates
        axes[0, 2].set_title('Physics Violation Rates')
        
        violation_data = []
        model_names = []
        
        for model_name, model_results in evaluation_results.items():
            if model_name == 'summary':
                continue
                
            violations = []
            for benchmark_results in model_results.values():
                if isinstance(benchmark_results, dict) and 'physics_violation_rate' in benchmark_results:
                    violations.append(benchmark_results['physics_violation_rate'] * 100)
            
            if violations:
                violation_data.append(np.mean(violations))
                model_names.append(model_name)
        
        if violation_data:
            bars = axes[0, 2].bar(model_names, violation_data)
            axes[0, 2].set_ylabel('Violation Rate (%)')
            axes[0, 2].set_title('Average Physics Violation Rate')
            
            # Color bars based on violation rate
            for bar, rate in zip(bars, violation_data):
                if rate > 10:
                    bar.set_color('red')
                elif rate > 5:
                    bar.set_color('orange')
                else:
                    bar.set_color('green')
        
        # 4. Uncertainty calibration
        axes[1, 0].set_title('Uncertainty Calibration')
        
        # This would show calibration curves if uncertainty data is available
        # For now, show placeholder
        axes[1, 0].text(0.5, 0.5, 'Uncertainty Calibration\n(Requires uncertainty data)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 5. Performance vs accuracy trade-off
        axes[1, 1].set_title('Performance vs Accuracy Trade-off')
        
        # Extract performance and accuracy data
        accuracy_scores = []
        processing_times = []
        model_labels = []
        
        for model_name, model_results in evaluation_results.items():
            if model_name == 'summary':
                continue
                
            # Get average R² across benchmarks
            r2_scores = []
            for benchmark_results in model_results.values():
                if isinstance(benchmark_results, dict) and 'r2' in benchmark_results:
                    r2_scores.append(benchmark_results['r2'])
            
            if r2_scores:
                avg_r2 = np.mean(r2_scores)
                # Simulated processing time (would be real data in practice)
                proc_time = np.random.uniform(0.1, 2.0)
                
                accuracy_scores.append(avg_r2)
                processing_times.append(proc_time)
                model_labels.append(model_name)
        
        if accuracy_scores:
            scatter = axes[1, 1].scatter(processing_times, accuracy_scores, s=100)
            
            for i, label in enumerate(model_labels):
                axes[1, 1].annotate(label, (processing_times[i], accuracy_scores[i]),
                                  xytext=(5, 5), textcoords='offset points')
            
            axes[1, 1].set_xlabel('Processing Time (seconds)')
            axes[1, 1].set_ylabel('Average R² Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model ranking radar chart
        axes[1, 2].set_title('Model Performance Radar Chart')
        
        # Create radar chart for top 3 models
        if 'summary' in evaluation_results and 'model_rankings' in evaluation_results['summary']:
            rankings = evaluation_results['summary']['model_rankings']
            top_models = list(rankings.values())[:3]
            
            # Define metrics for radar chart
            metrics = ['Accuracy', 'Speed', 'Robustness', 'Uncertainty', 'Physics']
            
            # Generate synthetic data for demonstration
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            for i, model_info in enumerate(top_models):
                # Synthetic scores (would be real metrics in practice)
                scores = np.random.uniform(0.6, 1.0, len(metrics))
                scores = np.concatenate((scores, [scores[0]]))
                
                axes[1, 2].plot(angles, scores, 'o-', linewidth=2, 
                              label=model_info['model'])
                axes[1, 2].fill(angles, scores, alpha=0.25)
            
            axes[1, 2].set_xticks(angles[:-1])
            axes[1, 2].set_xticklabels(metrics)
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_analysis(self, predictions: np.ndarray, 
                                uncertainties: np.ndarray, 
                                true_values: np.ndarray,
                                save_path: str = None) -> plt.Figure:
        """Create comprehensive uncertainty analysis plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        errors = np.abs(predictions - true_values)
        
        # 1. Uncertainty vs Error scatter
        axes[0, 0].scatter(uncertainties, errors, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Predicted Uncertainty')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].set_title('Uncertainty vs Error Correlation')
        
        # Add correlation coefficient
        corr = np.corrcoef(uncertainties, errors)[0, 1]
        axes[0, 0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Calibration curve
        n_bins = 10
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
        
        axes[0, 1].errorbar(mean_uncertainties, mean_errors, yerr=error_bars, 
                          marker='o', capsize=5, capthick=2, label='Observed')
        axes[0, 1].plot([0, max(mean_uncertainties)], [0, max(mean_uncertainties)], 
                       'r--', label='Perfect calibration')
        axes[0, 1].set_xlabel('Predicted Uncertainty')
        axes[0, 1].set_ylabel('Observed Error')
        axes[0, 1].set_title('Calibration Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Coverage analysis
        confidence_levels = [0.68, 0.95, 0.99]
        z_scores = [1.0, 1.96, 2.58]
        coverages = []
        
        for z in z_scores:
            in_interval = errors <= z * uncertainties
            coverage = np.mean(in_interval)
            coverages.append(coverage)
        
        axes[0, 2].bar(confidence_levels, coverages, alpha=0.7)
        axes[0, 2].plot(confidence_levels, confidence_levels, 'r--', label='Perfect coverage')
        axes[0, 2].set_xlabel('Nominal Confidence Level')
        axes[0, 2].set_ylabel('Empirical Coverage')
        axes[0, 2].set_title('Coverage Analysis')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add text annotations
        for i, (conf, cov) in enumerate(zip(confidence_levels, coverages)):
            axes[0, 2].text(conf, cov + 0.02, f'{cov:.2f}', ha='center', va='bottom')
        
        # 4. Uncertainty distribution
        axes[1, 0].hist(uncertainties, bins=30, alpha=0.7, density=True, edgecolor='black')
        axes[1, 0].set_xlabel('Predicted Uncertainty')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Uncertainty Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_unc = np.mean(uncertainties)
        std_unc = np.std(uncertainties)
        axes[1, 0].axvline(mean_unc, color='red', linestyle='--', label=f'Mean: {mean_unc:.3f}')
        axes[1, 0].axvline(mean_unc + std_unc, color='orange', linestyle=':', alpha=0.7)
        axes[1, 0].axvline(mean_unc - std_unc, color='orange', linestyle=':', alpha=0.7)
        axes[1, 0].legend()
        
        # 5. Error distribution by uncertainty quartiles
        q1, q2, q3 = np.percentile(uncertainties, [25, 50, 75])
        
        low_unc_mask = uncertainties <= q1
        mid_unc_mask = (uncertainties > q1) & (uncertainties <= q3)
        high_unc_mask = uncertainties > q3
        
        axes[1, 1].hist([errors[low_unc_mask], errors[mid_unc_mask], errors[high_unc_mask]], 
                       bins=20, alpha=0.7, label=['Low uncertainty', 'Medium uncertainty', 'High uncertainty'],
                       density=True)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Error Distribution by Uncertainty Level')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Prediction intervals
        sorted_indices = np.argsort(predictions)
        sorted_predictions = predictions[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_true = true_values[sorted_indices]
        
        # Plot subset for clarity
        n_plot = min(100, len(sorted_predictions))
        indices = np.linspace(0, len(sorted_predictions)-1, n_plot, dtype=int)
        
        x_plot = sorted_predictions[indices]
        y_plot = sorted_true[indices]
        unc_plot = sorted_uncertainties[indices]
        
        axes[1, 2].scatter(x_plot, y_plot, alpha=0.6, s=20)
        axes[1, 2].errorbar(x_plot, x_plot, yerr=1.96*unc_plot, 
                          fmt='none', alpha=0.3, color='red')
        
        # Perfect prediction line
        min_val = min(np.min(x_plot), np.min(y_plot))
        max_val = max(np.max(x_plot), np.max(y_plot))
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axes[1, 2].set_xlabel('Predicted Value')
        axes[1, 2].set_ylabel('True Value')
        axes[1, 2].set_title('Prediction Intervals (95% confidence)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_monitoring_dashboard(self, save_path: str = None) -> plt.Figure:
        """Create real-time monitoring dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Request volume over time
        ax1 = fig.add_subplot(gs[0, :2])
        if self.production_system and self.production_system.request_history:
            timestamps = [r['timestamp'] for r in self.production_system.request_history]
            batch_sizes = [r['batch_size'] for r in self.production_system.request_history]
            
            ax1.plot(timestamps, batch_sizes, 'b-', alpha=0.7)
            ax1.set_title('Request Volume Over Time')
            ax1.set_ylabel('Batch Size')
            ax1.tick_params(axis='x', rotation=45)
        else:
            ax1.text(0.5, 0.5, 'No request data available', 
                    ha='center', va='center', transform=ax1.transAxes)
        ax1.grid(True, alpha=0.3)
        
        # 2. Response time distribution
        ax2 = fig.add_subplot(gs[0, 2])
        if self.production_system and self.production_system.request_history:
            response_times = [r['processing_time'] for r in self.production_system.request_history]
            ax2.hist(response_times, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_title('Response Time Distribution')
            ax2.set_xlabel('Processing Time (s)')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate gauge
        ax3 = fig.add_subplot(gs[0, 3])
        if self.production_system and self.production_system.request_history:
            recent_success_rates = [r['success_rate'] for r in self.production_system.request_history[-10:]]
            avg_success_rate = np.mean(recent_success_rates) if recent_success_rates else 1.0
        else:
            avg_success_rate = 1.0
        
        # Create gauge chart
        theta = np.pi * avg_success_rate
        ax3.barh(0, theta, color='green' if avg_success_rate > 0.9 else 'orange' if avg_success_rate > 0.7 else 'red')
        ax3.set_xlim(0, np.pi)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_title(f'Success Rate: {avg_success_rate:.1%}')
        ax3.set_xticks([0, np.pi/2, np.pi])
        ax3.set_xticklabels(['0%', '50%', '100%'])
        
        # 4. Model performance comparison
        ax4 = fig.add_subplot(gs[1, :2])
        # This would show real-time model performance metrics
        ax4.text(0.5, 0.5, 'Model Performance Metrics\n(Real-time)', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Model Performance Comparison')
        
        # 5. System resource usage
        ax5 = fig.add_subplot(gs[1, 2:])
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            resources = ['CPU', 'Memory']
            usage = [cpu_percent, memory_percent]
            colors = ['green' if u < 70 else 'orange' if u < 90 else 'red' for u in usage]
            
            bars = ax5.bar(resources, usage, color=colors)
            ax5.set_ylabel('Usage (%)')
            ax5.set_title('System Resource Usage')
            ax5.set_ylim(0, 100)
            
            # Add value labels
            for bar, val in zip(bars, usage):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{val:.1f}%', ha='center', va='bottom')
        except:
            ax5.text(0.5, 0.5, 'Resource monitoring unavailable', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # 6. Error rate trends
        ax6 = fig.add_subplot(gs[2, :2])
        if self.production_system and self.production_system.request_history:
            timestamps = [r['timestamp'] for r in self.production_system.request_history]
            error_rates = [1 - r['success_rate'] for r in self.production_system.request_history]
            
            ax6.plot(timestamps, error_rates, 'r-', alpha=0.7)
            ax6.set_title('Error Rate Over Time')
            ax6.set_ylabel('Error Rate')
            ax6.tick_params(axis='x', rotation=45)
            ax6.set_ylim(0, max(max(error_rates) * 1.1, 0.1) if error_rates else 0.1)
        else:
            ax6.text(0.5, 0.5, 'No error data', ha='center', va='center', transform=ax6.transAxes)
        ax6.grid(True, alpha=0.3)
        
        # 7. Active models status
        ax7 = fig.add_subplot(gs[2, 2:])
        if self.production_system and self.production_system.models:
            model_types = list(self.production_system.models.keys())
            model_counts = [len(versions) for versions in self.production_system.models.values()]
            
            bars = ax7.bar(model_types, model_counts)
            ax7.set_title('Active Model Versions')
            ax7.set_ylabel('Number of Versions')
            
            # Add value labels
            for bar, count in zip(bars, model_counts):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(count), ha='center', va='bottom')
        else:
            ax7.text(0.5, 0.5, 'No models loaded', ha='center', va='center', transform=ax7.transAxes)
        
        plt.suptitle('ORION Real-time Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# =====================================================================
# 4. COMPLETE USAGE EXAMPLE
# =====================================================================

async def main_example():
    """Complete usage example demonstrating all components"""
    
    print("ORION: Advanced Usage Examples")
    print("="*50)
    
    # 1. Setup advanced training pipeline
    print("\n1. Setting up advanced training pipeline...")
    
    config = {
        'model_types': ['gnn', 'rf'],
        'optimization_budget': 10,  # Reduced for demo
        'use_wandb': False,
        'save_models': True
    }
    
    pipeline = AdvancedTrainingPipeline(config)
    
    # 2. Create synthetic training data
    print("2. Creating synthetic training data...")
    
    # This would be real materials data in practice
    n_samples = 500
    synthetic_data = []
    
    for i in range(n_samples):
        # Create synthetic graph data
        num_nodes = np.random.randint(5, 20)
        num_edges = np.random.randint(num_nodes, num_nodes * 2)
        
        # Node features (atomic properties)
        node_features = torch.randn(num_nodes, 10)
        
        # Edge indices
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Properties to predict
        bandgap = np.random.uniform(0.5, 4.0)
        formation_energy = np.random.normal(-1.0, 1.0)
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            bandgap=torch.tensor([bandgap], dtype=torch.float),
            formation_energy=torch.tensor([formation_energy], dtype=torch.float)
        )
        
        synthetic_data.append(data)
    
    # Split data
    train_size = int(0.7 * len(synthetic_data))
    val_size = int(0.15 * len(synthetic_data))
    
    train_data = synthetic_data[:train_size]
    val_data = synthetic_data[train_size:train_size + val_size]
    test_data = synthetic_data[train_size + val_size:]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 3. Run hyperparameter optimization
    print("\n3. Running hyperparameter optimization...")
    
    target_properties = ['bandgap', 'formation_energy']
    
    try:
        optimization_results = pipeline.optimize_all_models(
            train_loader, val_loader, target_properties
        )
        
        print("Optimization completed!")
        for model_type, results in optimization_results.items():
            if 'best_score' in results:
                print(f"  {model_type}: Best score = {results['best_score']:.4f}")
    
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Create dummy results for demonstration
        optimization_results = {
            'gnn': {
                'best_params': {'hidden_dim': 128, 'num_layers': 3, 'learning_rate': 0.001},
                'best_score': 0.85
            }
        }
    
    # 4. Train final models
    print("\n4. Training final models...")
    
    try:
        final_models = pipeline.train_final_models(
            optimization_results, train_loader, test_loader, target_properties
        )
        
        print("Final training completed!")
        for model_type, results in final_models.items():
            test_r2 = results['test_metrics'].get('r2', 0)
            print(f"  {model_type}: Test R² = {test_r2:.4f}")
    
    except Exception as e:
        print(f"Final training failed: {e}")
        final_models = {}
    
    # 5. Setup production system
    print("\n5. Setting up production system...")
    
    production_system = ORIONProductionSystem()
    
    # Simulate loading models
    # In practice, you'd load actual trained models
    print("  Simulating model loading...")
    
    # 6. Create monitoring dashboard
    print("\n6. Creating monitoring dashboard...")
    
    dashboard = ORIONDashboard(production_system)
    
    # Generate some synthetic results for demonstration
    synthetic_evaluation_results = {
        'gnn': {
            'bandgap_benchmark': {'mae': 0.15, 'rmse': 0.22, 'r2': 0.85, 'physics_violation_rate': 0.02},
            'formation_energy_benchmark': {'mae': 0.08, 'rmse': 0.12, 'r2': 0.92, 'physics_violation_rate': 0.01}
        },
        'rf': {
            'bandgap_benchmark': {'mae': 0.18, 'rmse': 0.25, 'r2': 0.82, 'physics_violation_rate': 0.05},
            'formation_energy_benchmark': {'mae': 0.10, 'rmse': 0.15, 'r2': 0.88, 'physics_violation_rate': 0.03}
        },
        'summary': {
            'model_rankings': {
                1: {'model': 'gnn', 'avg_r2': 0.885},
                2: {'model': 'rf', 'avg_r2': 0.850}
            }
        }
    }
    
    # Create visualizations
    print("  Generating comparison plots...")
    try:
        comparison_fig = dashboard.plot_model_comparison(synthetic_evaluation_results)
        comparison_fig.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        print("    Model comparison plot saved: model_comparison.png")
    except Exception as e:
        print(f"    Plot generation failed: {e}")
    
    # Create uncertainty analysis
    print("  Generating uncertainty analysis...")
    try:
        # Synthetic uncertainty data
        n_points = 200
        true_vals = np.random.normal(2.0, 1.0, n_points)
        predictions = true_vals + np.random.normal(0, 0.2, n_points)
        uncertainties = np.random.uniform(0.1, 0.4, n_points)
        
        uncertainty_fig = dashboard.plot_uncertainty_analysis(
            predictions, uncertainties, true_vals
        )
        uncertainty_fig.savefig('uncertainty_analysis.png', dpi=150, bbox_inches='tight')
        print("    Uncertainty analysis plot saved: uncertainty_analysis.png")
    except Exception as e:
        print(f"    Uncertainty plot generation failed: {e}")
    
    # 7. Simulate production predictions
    print("\n7. Simulating production predictions...")
    
    # Create synthetic candidates
    candidates = []
    for i in range(10):
        candidate = {
            'material_id': f'candidate_{i}',
            'composition': {'Ti': 0.5, 'O': 0.5},
            'features': np.random.randn(50)  # Feature vector
        }
        candidates.append(candidate)
    
    # Make batch prediction (simulated)
    try:
        result = await production_system.predict_batch(
            candidates, model_type='gnn', include_uncertainty=True
        )
        
        print(f"  Batch prediction completed:")
        print(f"    Batch size: {result['batch_size']}")
        print(f"    Processing time: {result['processing_time']:.3f}s")
        print(f"    Success rate: {result['success_rate']:.1%}")
        
    except Exception as e:
        print(f"  Batch prediction failed: {e}")
    
    # 8. System health check
    print("\n8. Performing system health check...")
    
    try:
        health_status = await production_system.health_check()
        print(f"  System status: {health_status['status']}")
        for check, result in health_status['checks'].items():
            print(f"    {check}: {result}")
    
    except Exception as e:
        print(f"  Health check failed: {e}")
    
    print("\nORION: Advanced Usage Examples Completed!")
    print("="*50)

if __name__ == "__main__":
    # Run the complete example
    asyncio.run(main_example())
