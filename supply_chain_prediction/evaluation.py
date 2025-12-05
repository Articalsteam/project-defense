"""
Evaluation and analysis utilities for supply chain prediction
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error
)
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate binary classification metrics (delayed vs on-time).
        
        Args:
            y_true: True delay values (continuous)
            y_pred: Predicted delay values (continuous)
            threshold: Delay threshold for classification
            
        Returns:
            Dictionary of classification metrics
        """
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * (tp / (tp + fp) if (tp + fp) > 0 else 0) * 
                       (tp / (tp + fn) if (tp + fn) > 0 else 0) / 
                       ((tp / (tp + fp) if (tp + fp) > 0 else 0) + 
                        (tp / (tp + fn) if (tp + fn) > 0 else 0) + 1e-10),
        }
        
        return metrics
    
    @staticmethod
    def get_residual_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate residual statistics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of residual statistics
        """
        residuals = y_true - y_pred
        
        stats = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals),
            'residual_median': np.median(residuals),
        }
        
        return stats


class DelayAnalyzer:
    """Analyze delay predictions and patterns."""
    
    @staticmethod
    def analyze_by_category(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
        """
        Analyze prediction errors by category.
        
        Args:
            df: DataFrame with categorical features
            y_pred: Predictions
            
        Returns:
            DataFrame with analysis by category
        """
        df_analysis = df.copy()
        df_analysis['predicted_delay'] = y_pred
        df_analysis['actual_delay'] = df['delay_days'].values
        df_analysis['error'] = abs(df_analysis['actual_delay'] - df_analysis['predicted_delay'])
        
        # Analyze by product category
        if 'product_category' in df.columns:
            category_stats = df_analysis.groupby('product_category').agg({
                'predicted_delay': 'mean',
                'actual_delay': 'mean',
                'error': ['mean', 'std', 'count']
            }).round(3)
            return category_stats
        
        return None
    
    @staticmethod
    def identify_high_risk_shipments(df: pd.DataFrame, y_pred: np.ndarray,
                                     threshold: float = 5.0) -> pd.DataFrame:
        """
        Identify shipments at high risk of delay.
        
        Args:
            df: DataFrame with shipment features
            y_pred: Predicted delays
            threshold: Delay threshold for high risk
            
        Returns:
            DataFrame of high-risk shipments
        """
        high_risk = df.copy()
        high_risk['predicted_delay'] = y_pred
        high_risk['risk_level'] = pd.cut(
            high_risk['predicted_delay'],
            bins=[0, 1, 3, 5, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return high_risk[high_risk['risk_level'].isin(['High', 'Critical'])].sort_values(
            'predicted_delay', ascending=False
        )
    
    @staticmethod
    def get_delay_distribution_stats(y_actual: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict:
        """
        Compare actual vs predicted delay distributions.
        
        Args:
            y_actual: Actual delays
            y_pred: Predicted delays
            
        Returns:
            Dictionary with distribution statistics
        """
        stats = {
            'actual': {
                'mean': np.mean(y_actual),
                'median': np.median(y_actual),
                'std': np.std(y_actual),
                'percentile_25': np.percentile(y_actual, 25),
                'percentile_75': np.percentile(y_actual, 75),
                'max': np.max(y_actual),
            },
            'predicted': {
                'mean': np.mean(y_pred),
                'median': np.median(y_pred),
                'std': np.std(y_pred),
                'percentile_25': np.percentile(y_pred, 25),
                'percentile_75': np.percentile(y_pred, 75),
                'max': np.max(y_pred),
            }
        }
        
        return stats
    
    @staticmethod
    def calculate_early_warning_metrics(df: pd.DataFrame, y_pred: np.ndarray,
                                       planning_horizon: int = 7) -> Dict:
        """
        Calculate metrics for early warning system.
        
        Args:
            df: DataFrame with shipment features
            y_pred: Predicted delays
            planning_horizon: Days ahead for warning
            
        Returns:
            Dictionary with early warning metrics
        """
        predicted_delayed = (y_pred > 0).astype(int)
        actual_delayed = (df['delay_days'].values > 0).astype(int)
        
        # Calculate how many delays we can catch before planning_horizon
        catchable_delays = (y_pred > (df['scheduled_delivery_days'].values - planning_horizon)).astype(int)
        
        metrics = {
            'total_shipments': len(df),
            'predicted_delayed_count': np.sum(predicted_delayed),
            'actual_delayed_count': np.sum(actual_delayed),
            'catchable_delays': np.sum(catchable_delays & actual_delayed),
            'false_alarms': np.sum(catchable_delays & (1 - actual_delayed)),
            'missed_delays': np.sum((1 - catchable_delays) & actual_delayed),
        }
        
        if np.sum(actual_delayed) > 0:
            metrics['early_warning_recall'] = metrics['catchable_delays'] / np.sum(actual_delayed)
        
        if np.sum(catchable_delays) > 0:
            metrics['early_warning_precision'] = metrics['catchable_delays'] / np.sum(catchable_delays)
        
        return metrics


class PerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self):
        self.history = []
    
    def record(self, epoch: int, metrics: Dict) -> None:
        """
        Record metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        record = {'epoch': epoch, **metrics}
        self.history.append(record)
    
    def get_history_df(self) -> pd.DataFrame:
        """Get history as DataFrame."""
        return pd.DataFrame(self.history)
    
    def get_best_epoch(self, metric: str = 'val_mae') -> int:
        """Get epoch with best metric value."""
        df = self.get_history_df()
        if metric in df.columns:
            if 'loss' in metric or 'error' in metric or 'mse' in metric:
                return df[metric].idxmin()
            else:
                return df[metric].idxmax()
        return -1
