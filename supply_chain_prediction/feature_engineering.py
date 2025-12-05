"""
Feature engineering for supply chain prediction system
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List


class FeatureEngineer:
    """Handle feature engineering for supply chain data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.categorical_features = [
            'product_category', 'transportation_mode', 'weather_condition'
        ]
        self.numerical_features = [
            'order_quantity', 'order_value', 'supplier_reliability_score',
            'distance_km', 'fuel_price_index', 'port_congestion_score',
            'customs_clearance_hours', 'scheduled_delivery_days',
            'historical_delay_rate', 'supplier_inventory_level'
        ]
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit and transform features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (transformed features array, feature names)
        """
        df_processed = df.copy()
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
        
        # Select feature columns
        feature_cols = self.categorical_features + self.numerical_features
        feature_cols = [col for col in feature_cols if col in df_processed.columns]
        
        X = df_processed[feature_cols].values
        
        # Scale numerical features
        X = self.scaler.fit_transform(X)
        
        self.feature_names = feature_cols
        
        return X, feature_cols
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted encoders and scaler.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed features array
        """
        df_processed = df.copy()
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col in df_processed.columns and col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(
                    df_processed[col]
                )
        
        feature_cols = [col for col in self.feature_names 
                       if col in df_processed.columns]
        
        X = df_processed[feature_cols].values
        X = self.scaler.transform(X)
        
        return X
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features from date column.
        
        Args:
            df: Input dataframe with 'date' column
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Add to numerical features if not already there
        temporal_cols = ['day_of_week', 'month', 'quarter', 'is_weekend']
        for col in temporal_cols:
            if col not in self.numerical_features:
                self.numerical_features.append(col)
        
        return df
    
    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features between key variables.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Interaction: distance and transportation mode
        if 'distance_km' in df.columns and 'transportation_mode' in df.columns:
            transport_speed = {
                'Air': 500, 'Road': 80, 'Rail': 100, 'Sea': 30
            }
            df['estimated_transit_hours'] = df.apply(
                lambda row: row['distance_km'] / transport_speed.get(row['transportation_mode'], 100),
                axis=1
            )
            self.numerical_features.append('estimated_transit_hours')
        
        # Interaction: order value and quantity
        if 'order_value' in df.columns and 'order_quantity' in df.columns:
            df['value_per_unit'] = df['order_value'] / (df['order_quantity'] + 1)
            self.numerical_features.append('value_per_unit')
        
        # Interaction: supplier reliability and historical delay
        if 'supplier_reliability_score' in df.columns and 'historical_delay_rate' in df.columns:
            df['reliability_consistency'] = (df['supplier_reliability_score'] * 
                                            (1 - df['historical_delay_rate']))
            self.numerical_features.append('reliability_consistency')
        
        return df
    
    def get_feature_importance_preparation(self, feature_importance: Dict[str, float]):
        """
        Prepare feature importance scores for visualization.
        
        Args:
            feature_importance: Dictionary of feature names to importance scores
            
        Returns:
            Sorted DataFrame for visualization
        """
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        return importance_df
