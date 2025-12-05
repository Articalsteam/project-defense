"""
Visualization utilities for supply chain prediction system
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PredictionVisualizer:
    """Visualize prediction results and analysis."""
    
    @staticmethod
    def plot_predictions_vs_actual(y_actual: np.ndarray, y_pred: np.ndarray,
                                   title: str = "Actual vs Predicted Delays") -> None:
        """
        Plot actual vs predicted delays.
        
        Args:
            y_actual: Actual delay values
            y_pred: Predicted delay values
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(y_actual, y_pred, alpha=0.5, s=30)
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0].set_xlabel('Actual Delays (days)')
        axes[0].set_ylabel('Predicted Delays (days)')
        axes[0].set_title('Scatter Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_actual - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=30)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Delays (days)')
        axes[1].set_ylabel('Residuals (days)')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_names: list, importances: np.ndarray,
                               top_n: int = 15) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_names: Names of features
            importances: Importance scores
            top_n: Number of top features to show
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color='steelblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_error_distribution(y_actual: np.ndarray, y_pred: np.ndarray,
                               title: str = "Prediction Error Distribution") -> None:
        """
        Plot distribution of prediction errors.
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            title: Plot title
        """
        errors = y_actual - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.2f}')
        axes[0].set_xlabel('Error (days)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        abs_errors = np.abs(errors)
        axes[1].boxplot(abs_errors, vert=True)
        axes[1].set_ylabel('Absolute Error (days)')
        axes[1].set_title('Absolute Error Box Plot')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_prediction_by_category(df: pd.DataFrame, y_pred: np.ndarray,
                                    category_col: str = 'product_category') -> None:
        """
        Plot predictions by category.
        
        Args:
            df: DataFrame with features
            y_pred: Predictions
            category_col: Name of category column
        """
        df_plot = df.copy()
        df_plot['predicted_delay'] = y_pred
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if category_col in df_plot.columns:
            categories = df_plot[category_col].unique()
            positions = range(len(categories))
            data_to_plot = [df_plot[df_plot[category_col] == cat]['predicted_delay'].values 
                           for cat in categories]
            
            bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            ax.set_xlabel('Category')
            ax.set_ylabel('Predicted Delay (days)')
            ax.set_title(f'Predicted Delays by {category_col}')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def plot_delay_by_supplier(df: pd.DataFrame, y_pred: np.ndarray,
                              top_n: int = 10) -> None:
        """
        Plot average predicted delays by supplier.
        
        Args:
            df: DataFrame with features
            y_pred: Predictions
            top_n: Number of top suppliers to show
        """
        df_plot = df.copy()
        df_plot['predicted_delay'] = y_pred
        
        supplier_delays = df_plot.groupby('supplier_id')['predicted_delay'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        top_suppliers = supplier_delays.head(top_n)
        ax.bar(range(len(top_suppliers)), top_suppliers['mean'], color='coral', edgecolor='black')
        ax.set_xticks(range(len(top_suppliers)))
        ax.set_xticklabels([f"Supplier {idx}" for idx in top_suppliers.index], rotation=45)
        ax.set_ylabel('Average Predicted Delay (days)')
        ax.set_title(f'Top {top_n} Suppliers with Highest Predicted Delays')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(metrics_dict: dict) -> None:
        """
        Compare metrics across multiple models.
        
        Args:
            metrics_dict: Dictionary of model names to metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(metrics_dict.keys())
        
        # RMSE comparison
        rmse_values = [metrics_dict[m].get('rmse', 0) for m in models]
        axes[0, 0].bar(models, rmse_values, color='steelblue', edgecolor='black')
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # MAE comparison
        mae_values = [metrics_dict[m].get('mae', 0) for m in models]
        axes[0, 1].bar(models, mae_values, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # R² comparison
        r2_values = [metrics_dict[m].get('r2', 0) for m in models]
        axes[1, 0].bar(models, r2_values, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('R² Score Comparison')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        # MAPE comparison
        mape_values = [metrics_dict[m].get('mape', 0) for m in models]
        axes[1, 1].bar(models, mape_values, color='gold', edgecolor='black')
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_delivery_risk_heatmap(df: pd.DataFrame, y_pred: np.ndarray) -> None:
        """
        Create heatmap of delivery risk factors.
        
        Args:
            df: DataFrame with features
            y_pred: Predictions
        """
        df_plot = df.copy()
        df_plot['predicted_delay'] = y_pred
        
        # Create risk categories
        if 'transportation_mode' in df_plot.columns and 'weather_condition' in df_plot.columns:
            pivot_data = df_plot.pivot_table(
                values='predicted_delay',
                index='transportation_mode',
                columns='weather_condition',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Avg Delay (days)'})
            ax.set_title('Predicted Delay Heatmap: Transportation Mode vs Weather')
            ax.set_xlabel('Weather Condition')
            ax.set_ylabel('Transportation Mode')
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def plot_confidence_intervals(y_actual: np.ndarray, y_pred: np.ndarray,
                                 std: np.ndarray, confidence: float = 0.95) -> None:
        """
        Plot predictions with confidence intervals.
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            std: Standard deviation estimates
            confidence: Confidence level (e.g., 0.95 for 95%)
        """
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z_score * std
        
        # Sort by predicted value for better visualization
        indices = np.argsort(y_pred)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(indices))
        ax.scatter(x, y_actual[indices], alpha=0.6, s=20, label='Actual', color='blue')
        ax.plot(x, y_pred[indices], 'r-', label='Predicted', linewidth=2)
        ax.fill_between(x, 
                        y_pred[indices] - margin[indices],
                        y_pred[indices] + margin[indices],
                        alpha=0.2, color='red', label=f'{int(confidence*100)}% Confidence Interval')
        
        ax.set_xlabel('Shipment Index (sorted by predicted delay)')
        ax.set_ylabel('Delay (days)')
        ax.set_title('Predictions with Confidence Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
