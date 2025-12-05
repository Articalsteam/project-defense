"""
Main training and prediction pipeline for supply chain delay prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Use package-relative imports so the module can be imported when the
# package is installed or referenced as a package (e.g. via
# `from supply_chain_prediction import ...`). Absolute imports like
# `from data_loader import ...` fail when the package context isn't in
# sys.path.
from .data_loader import load_or_generate_data, split_temporal_data
from .feature_engineering import FeatureEngineer
from .models import DelayPredictionModel, EnsembleDelayPredictor
from .evaluation import ModelEvaluator, DelayAnalyzer, PerformanceTracker
from .visualization import PredictionVisualizer
import logging
import time
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# Module-level logger writes into `<project_root>/logs/pipeline.log` to help
# diagnose slow deployments. We only add handlers if none exist to avoid
# duplicating handlers when the module is reloaded.
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, 'pipeline.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class SupplyChainPredictionPipeline:
    """Complete pipeline for supply chain delay prediction."""
    
    def __init__(self, model_type: str = 'ensemble'):
        """
        Initialize the pipeline.
        
        Args:
            model_type: Type of model ('ensemble', 'xgboost', 'lightgbm', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = None
        self.evaluator = ModelEvaluator()
        self.analyzer = DelayAnalyzer()
        self.tracker = PerformanceTracker()
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        # Cache directory for trained models and artifacts
        self.cache_dir = os.path.join(os.getcwd(), 'models_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_data(self, filepath: str = None, n_samples: int = 1000) -> pd.DataFrame:
        """
        Load or generate data.
        
        Args:
            filepath: Path to CSV file (if None, generates synthetic data)
            n_samples: Number of samples to generate if filepath is None
            
        Returns:
            Loaded data
        """
        logger = logging.getLogger(__name__)
        start = time.perf_counter()
        logger.info("Loading data...")
        self.data = load_or_generate_data(filepath, n_samples)
        elapsed = time.perf_counter() - start
        logger.info(f"Data loaded: {len(self.data)} records (in {elapsed:.2f}s)")
        logger.debug(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def prepare_features(self) -> Tuple:
        """
        Prepare features for modeling.
        
        Returns:
            Tuple of (train_X, train_y, val_X, val_y, test_X, test_y)
        """
        logger = logging.getLogger(__name__)
        logger.info("Preparing features...")
        start = time.perf_counter()
        
        # Split data temporally
        self.train_data, self.val_data, self.test_data = split_temporal_data(self.data)
        logger.info(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        
        # Add temporal features
        self.train_data = self.feature_engineer.add_temporal_features(self.train_data)
        self.val_data = self.feature_engineer.add_temporal_features(self.val_data)
        self.test_data = self.feature_engineer.add_temporal_features(self.test_data)
        
        # Add interaction features
        self.train_data = self.feature_engineer.add_interaction_features(self.train_data)
        self.val_data = self.feature_engineer.add_interaction_features(self.val_data)
        self.test_data = self.feature_engineer.add_interaction_features(self.test_data)
        
        # Transform features
        train_X, feature_names = self.feature_engineer.fit_transform(self.train_data)
        val_X = self.feature_engineer.transform(self.val_data)
        test_X = self.feature_engineer.transform(self.test_data)
        
        train_y = self.train_data['delay_days'].values
        val_y = self.val_data['delay_days'].values
        test_y = self.test_data['delay_days'].values
        
        elapsed = time.perf_counter() - start
        logger.info(f"Features prepared: {len(feature_names)} features (in {elapsed:.2f}s)")
        
        return train_X, train_y, val_X, val_y, test_X, test_y, feature_names
    
    def train_model(self, train_X: np.ndarray, train_y: np.ndarray,
                   val_X: np.ndarray, val_y: np.ndarray) -> Dict:
        """
        Train the model.
        
        Args:
            train_X: Training features
            train_y: Training targets
            val_X: Validation features
            val_y: Validation targets
            
        Returns:
            Dictionary of training metrics
        """
        logger = logging.getLogger(__name__)
        logger.info("Training model...")
        start = time.perf_counter()
        
        if self.model_type == 'ensemble':
            self.model = EnsembleDelayPredictor()
            metrics = self.model.train(train_X, train_y, val_X, val_y)
            logger.info(f"Ensemble trained with {len(self.model.models)} models")
            for model_name, model_metrics in metrics.items():
                logger.info(f"{model_name}: {model_metrics}")
        else:
            self.model = DelayPredictionModel(self.model_type)
            metrics = self.model.train(train_X, train_y, val_X, val_y)
            logger.info(f"Model trained: {self.model_type}")
            logger.info(metrics)
        
        elapsed = time.perf_counter() - start
        logger.info(f"Training completed in {elapsed:.2f}s")

        # Save trained artifacts to cache for faster restarts
        try:
            self.save_artifacts()
            logger.info(f"Saved artifacts to {self.cache_dir}")
        except Exception:
            logger.exception("Failed to save artifacts")

        return metrics

    def save_artifacts(self, prefix: str = 'pipeline') -> None:
        """Save model and feature engineer to the pipeline cache directory."""
        model_path = os.path.join(self.cache_dir, f"{prefix}_model.joblib")
        engineer_path = os.path.join(self.cache_dir, f"{prefix}_engineer.joblib")
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_engineer, engineer_path)
        logger.info(f"Artifacts saved: {model_path}, {engineer_path}")

    def load_artifacts(self, prefix: str = 'pipeline') -> bool:
        """Attempt to load model and feature engineer from cache. Returns True if loaded."""
        model_path = os.path.join(self.cache_dir, f"{prefix}_model.joblib")
        engineer_path = os.path.join(self.cache_dir, f"{prefix}_engineer.joblib")
        if os.path.exists(model_path) and os.path.exists(engineer_path):
            try:
                self.model = joblib.load(model_path)
                self.feature_engineer = joblib.load(engineer_path)
                self.is_trained = True
                logger.info(f"Loaded artifacts from {self.cache_dir}")
                return True
            except Exception:
                logger.exception("Failed to load cached artifacts")
                return False
        else:
            logger.info("No cached artifacts found")
            return False
    
    def evaluate(self, test_X: np.ndarray, test_y: np.ndarray,
                test_data: pd.DataFrame = None) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_X: Test features
            test_y: Test targets
            test_data: Test dataframe (optional, for detailed analysis)
            
        Returns:
            Dictionary of evaluation results
        """
        print("\nEvaluating model...")
        
        # Make predictions
        predictions = self.model.predict(test_X)
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(test_y, predictions)
        class_metrics = self.evaluator.calculate_classification_metrics(test_y, predictions)
        residual_stats = self.evaluator.get_residual_statistics(test_y, predictions)
        
        print("\nRegression Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nClassification Metrics (Delayed vs On-time):")
        for key, value in class_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nResidual Statistics:")
        for key, value in residual_stats.items():
            print(f"  {key}: {value:.4f}")
        
        results = {
            'metrics': metrics,
            'class_metrics': class_metrics,
            'residual_stats': residual_stats,
            'predictions': predictions,
            'actual': test_y,
        }
        
        if test_data is not None:
            # Analyze by category
            print("\nAnalysis by Product Category:")
            category_analysis = self.analyzer.analyze_by_category(test_data, predictions)
            if category_analysis is not None:
                print(category_analysis)
            
            # Early warning metrics
            early_warning = self.analyzer.calculate_early_warning_metrics(test_data, predictions)
            print("\nEarly Warning Metrics:")
            for key, value in early_warning.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            results['early_warning'] = early_warning
        
        return results
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions, optionally with uncertainty
        """
        if return_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
            return self.model.predict_with_uncertainty(X)
        else:
            return self.model.predict(X)
    
    def identify_high_risk(self, test_X: np.ndarray, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify high-risk shipments.
        
        Args:
            test_X: Test features
            test_data: Test dataframe
            
        Returns:
            DataFrame of high-risk shipments
        """
        predictions = self.model.predict(test_X)
        high_risk = self.analyzer.identify_high_risk_shipments(test_data, predictions)
        
        print(f"\nHigh-risk shipments identified: {len(high_risk)}")
        print(high_risk[['supplier_id', 'warehouse_id', 'predicted_delay', 'risk_level']].head(10))
        
        return high_risk
    
    def visualize_results(self, test_X: np.ndarray, test_y: np.ndarray,
                         test_data: pd.DataFrame = None, feature_names: list = None) -> None:
        """
        Visualize prediction results.
        
        Args:
            test_X: Test features
            test_y: Test targets
            test_data: Test dataframe (optional)
            feature_names: Feature names (optional)
        """
        print("\nGenerating visualizations...")
        
        predictions = self.model.predict(test_X)
        
        # Predictions vs actual
        PredictionVisualizer.plot_predictions_vs_actual(test_y, predictions)
        
        # Error distribution
        PredictionVisualizer.plot_error_distribution(test_y, predictions)
        
        # Feature importance (if available)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'feature_importances_'):
            if feature_names:
                PredictionVisualizer.plot_feature_importance(
                    feature_names,
                    self.model.model.feature_importances_
                )
        
        # Category analysis
        if test_data is not None:
            if 'product_category' in test_data.columns:
                PredictionVisualizer.plot_prediction_by_category(test_data, predictions)
            
            if 'supplier_id' in test_data.columns:
                PredictionVisualizer.plot_delay_by_supplier(test_data, predictions)
            
            if 'transportation_mode' in test_data.columns and 'weather_condition' in test_data.columns:
                PredictionVisualizer.plot_delivery_risk_heatmap(test_data, predictions)
        
        # Confidence intervals (if uncertainty available)
        if hasattr(self.model, 'predict_with_uncertainty'):
            _, std = self.model.predict_with_uncertainty(test_X)
            PredictionVisualizer.plot_confidence_intervals(test_y, predictions, std)


def main():
    """Run the complete supply chain prediction pipeline."""
    
    print("=" * 80)
    print("SUPPLY CHAIN DELAY PREDICTION SYSTEM")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = SupplyChainPredictionPipeline(model_type='ensemble')
    
    # Load data
    data = pipeline.load_data(n_samples=1000)
    
    # Prepare features
    train_X, train_y, val_X, val_y, test_X, test_y, feature_names = pipeline.prepare_features()
    
    # Train model
    train_metrics = pipeline.train_model(train_X, train_y, val_X, val_y)
    
    # Evaluate
    results = pipeline.evaluate(test_X, test_y, pipeline.test_data)
    
    # Identify high-risk shipments
    high_risk = pipeline.identify_high_risk(test_X, pipeline.test_data)
    
    # Visualize results
    pipeline.visualize_results(test_X, test_y, pipeline.test_data, feature_names)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return pipeline, results


if __name__ == "__main__":
    from typing import Dict, Tuple
    pipeline, results = main()
