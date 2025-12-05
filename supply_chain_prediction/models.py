"""
Model training and evaluation for supply chain delay prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class DelayPredictionModel:
    """Train and manage delay prediction models."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model.
        
        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.is_trained = False
        self.feature_importance = None
    
    def _create_model(self):
        """Create the specified model."""
        if self.model_type == 'xgboost':
            # Use `verbosity` for newer xgboost versions; `verbose` may be ignored
            # and produce warnings like: Parameters: { "verbose" } are not used.
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                eval_metric='rmse'
            )
        elif self.model_type == 'lightgbm':
            return LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                random_state=42
            )
        else:
            return LinearRegression()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary with training metrics
        """
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # For XGBoost, pass `verbose=False` through the `fit` kwargs only
            # when supported; newer xgboost versions accept `verbose` in fit.
            try:
                self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            except TypeError:
                # Fall back if `verbose` is not accepted by this estimator's fit
                self.model.fit(X_train, y_train, eval_set=eval_set)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X_train)
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_r2': r2_score(y_train, y_pred_train)
        }
        
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            metrics.update({
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_pred_val)),
                'val_mae': mean_absolute_error(y_val, y_pred_val),
                'val_r2': r2_score(y_val, y_pred_val)
            })
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted delay days
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates (for tree-based models).
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, std_estimates)
        """
        predictions = self.predict(X)
        
        # Estimate uncertainty using tree variance
        if hasattr(self.model, 'estimators_'):
            predictions_per_tree = np.array([
                tree.predict(X) for tree in self.model.estimators_
            ])
            std_estimates = np.std(predictions_per_tree, axis=0)
        else:
            std_estimates = np.ones_like(predictions) * np.std(predictions)
        
        return predictions, std_estimates
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        joblib.dump(self.model, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Args:
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class EnsembleDelayPredictor:
    """Ensemble of multiple models for robust predictions."""
    
    def __init__(self, model_types: List[str] = None):
        """
        Initialize ensemble.
        
        Args:
            model_types: List of model types to use
        """
        if model_types is None:
            model_types = ['xgboost', 'lightgbm', 'random_forest']
        
        self.models = {model_type: DelayPredictionModel(model_type) 
                      for model_type in model_types}
        self.weights = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with all models' metrics
        """
        all_metrics = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            metrics = model.train(X_train, y_train, X_val, y_val)
            all_metrics[model_name] = metrics
        
        # Set equal weights initially
        self.weights = {name: 1.0 / len(self.models) 
                       for name in self.models.keys()}
        
        return all_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Weighted average predictions
        """
        predictions = np.zeros(len(X))
        
        for model_name, model in self.models.items():
            weight = self.weights.get(model_name, 1.0 / len(self.models))
            predictions += weight * model.predict(X)
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions with uncertainty.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, std)
        """
        all_predictions = []
        
        for model_name, model in self.models.items():
            pred, _ = model.predict_with_confidence(X)
            all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        predictions = np.mean(all_predictions, axis=0)
        std = np.std(all_predictions, axis=0)
        
        return predictions, std
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom model weights.
        
        Args:
            weights: Dictionary of model names to weights
        """
        self.weights = weights
