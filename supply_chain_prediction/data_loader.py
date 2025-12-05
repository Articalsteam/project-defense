"""
Data generation and loading utilities for supply chain prediction system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple


class SupplyChainDataGenerator:
    """Generate synthetic supply chain data for model training and testing."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic supply chain dataset with features and delay labels.
        
        Args:
            n_samples: Number of records to generate
            
        Returns:
            DataFrame with supply chain features and delay target
        """
        dates = [datetime(2023, 1, 1) + timedelta(days=i) 
                 for i in range(n_samples)]
        
        data = {
            'date': dates,
            'supplier_id': np.random.choice(range(1, 21), n_samples),
            'warehouse_id': np.random.choice(range(1, 11), n_samples),
            'product_category': np.random.choice(
                ['Electronics', 'Textiles', 'Chemicals', 'Food', 'Machinery'], 
                n_samples
            ),
            'order_quantity': np.random.randint(100, 10000, n_samples),
            'order_value': np.random.uniform(1000, 100000, n_samples),
            'supplier_reliability_score': np.random.uniform(0.5, 1.0, n_samples),
            'distance_km': np.random.uniform(100, 5000, n_samples),
            'transportation_mode': np.random.choice(
                ['Air', 'Sea', 'Road', 'Rail'], 
                n_samples
            ),
            'weather_condition': np.random.choice(
                ['Clear', 'Rainy', 'Stormy', 'Foggy'], 
                n_samples
            ),
            'fuel_price_index': np.random.uniform(0.8, 1.5, n_samples),
            'port_congestion_score': np.random.uniform(0, 1, n_samples),
            'customs_clearance_hours': np.random.randint(1, 48, n_samples),
            'scheduled_delivery_days': np.random.randint(1, 30, n_samples),
            'historical_delay_rate': np.random.uniform(0, 0.3, n_samples),
            'supplier_inventory_level': np.random.uniform(0, 1, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate delay target based on features
        df['delay_days'] = self._calculate_delay(df)
        df['is_delayed'] = (df['delay_days'] > 0).astype(int)
        
        return df
    
    def _calculate_delay(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate realistic delays based on features.
        
        Args:
            df: DataFrame with supply chain features
            
        Returns:
            Array of delay days
        """
        delay = np.zeros(len(df))
        
        # Weather impact
        weather_impact = df['weather_condition'].map({
            'Clear': 0, 'Rainy': 0.5, 'Stormy': 2, 'Foggy': 1
        })
        
        # Supplier reliability impact
        supplier_impact = (1 - df['supplier_reliability_score']) * 3
        
        # Distance impact
        distance_impact = df['distance_km'] / 1000
        
        # Port congestion impact
        congestion_impact = df['port_congestion_score'] * 2
        
        # Fuel price impact
        fuel_impact = (df['fuel_price_index'] - 1) * 2
        
        # Transportation mode impact
        transport_impact = df['transportation_mode'].map({
            'Air': 0.2, 'Road': 0.5, 'Rail': 0.8, 'Sea': 1.5
        })
        
        # Combine impacts with some randomness
        delay = (weather_impact + supplier_impact + distance_impact + 
                congestion_impact + fuel_impact + transport_impact + 
                np.random.normal(0, 0.5, len(df)))
        
        return np.maximum(delay, 0)


def load_or_generate_data(filepath: str = None, 
                          n_samples: int = 1000) -> pd.DataFrame:
    """
    Load data from file or generate synthetic data.
    
    Args:
        filepath: Path to CSV file (if None, generates synthetic data)
        n_samples: Number of samples to generate if filepath is None
        
    Returns:
        DataFrame with supply chain data
    """
    if filepath:
        return pd.read_csv(filepath)
    else:
        generator = SupplyChainDataGenerator()
        return generator.generate_dataset(n_samples)


def split_temporal_data(df: pd.DataFrame, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for time-series validation.
    
    Args:
        df: Input dataframe with 'date' column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.sort_values('date').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    return train_df, val_df, test_df
