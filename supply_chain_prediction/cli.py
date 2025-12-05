#!/usr/bin/env python
"""
Command-line interface for supply chain prediction system
"""
import argparse
import sys
import joblib
import pandas as pd
import numpy as np
from .pipeline import SupplyChainPredictionPipeline
from .data_loader import load_or_generate_data
from .feature_engineering import FeatureEngineer
import warnings

warnings.filterwarnings('ignore')


def train_command(args):
    """Train a new model."""
    print("Starting training pipeline...")
    
    pipeline = SupplyChainPredictionPipeline(model_type=args.model_type)
    
    # Load data
    data = pipeline.load_data(filepath=args.data_file, n_samples=args.samples)
    
    # Prepare features
    train_X, train_y, val_X, val_y, test_X, test_y, feature_names = pipeline.prepare_features()
    
    # Train model
    pipeline.train_model(train_X, train_y, val_X, val_y)
    
    # Evaluate
    results = pipeline.evaluate(test_X, test_y, pipeline.test_data)
    
    # Save model and feature engineer
    model_path = args.output_model or 'supply_chain_model.pkl'
    engineer_path = args.output_engineer or 'feature_engineer.pkl'
    
    if isinstance(pipeline.model, type(pipeline.model)):
        joblib.dump(pipeline.model, model_path)
    joblib.dump(pipeline.feature_engineer, engineer_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Feature engineer saved to: {engineer_path}")
    
    # Save results
    if args.output_results:
        with open(args.output_results, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {args.output_results}")


def predict_command(args):
    """Make predictions on new data."""
    print("Loading model and feature engineer...")
    
    # Load model and feature engineer
    model = joblib.load(args.model)
    feature_engineer = joblib.load(args.engineer)
    
    print("Loading data...")
    df = pd.read_csv(args.input_file)
    
    print(f"Making predictions for {len(df)} shipments...")
    
    # Transform features
    X = feature_engineer.transform(df)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Add predictions to dataframe
    df['predicted_delay_days'] = predictions
    df['is_delayed'] = (predictions > 0).astype(int)
    df['risk_level'] = pd.cut(
        predictions,
        bins=[0, 1, 3, 5, float('inf')],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # Save results
    output_file = args.output or 'predictions.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total shipments: {len(df)}")
    print(f"  Delayed (>0 days): {np.sum(predictions > 0)}")
    print(f"  Mean predicted delay: {np.mean(predictions):.2f} days")
    print(f"  Max predicted delay: {np.max(predictions):.2f} days")


def analyze_command(args):
    """Analyze prediction results."""
    print("Loading results...")
    
    df = pd.read_csv(args.file)
    
    if 'predicted_delay_days' not in df.columns:
        print("Error: File must contain 'predicted_delay_days' column")
        sys.exit(1)
    
    predictions = df['predicted_delay_days'].values
    
    print("\n" + "="*60)
    print("PREDICTION ANALYSIS")
    print("="*60)
    
    print("\nDelay Statistics:")
    print(f"  Mean: {np.mean(predictions):.2f} days")
    print(f"  Median: {np.median(predictions):.2f} days")
    print(f"  Std Dev: {np.std(predictions):.2f} days")
    print(f"  Min: {np.min(predictions):.2f} days")
    print(f"  Max: {np.max(predictions):.2f} days")
    print(f"  25th percentile: {np.percentile(predictions, 25):.2f} days")
    print(f"  75th percentile: {np.percentile(predictions, 75):.2f} days")
    
    print("\nDelay Classification:")
    on_time = np.sum(predictions <= 0)
    low = np.sum((predictions > 0) & (predictions <= 1))
    medium = np.sum((predictions > 1) & (predictions <= 3))
    high = np.sum((predictions > 3) & (predictions <= 5))
    critical = np.sum(predictions > 5)
    
    total = len(predictions)
    print(f"  On-time (0 days): {on_time} ({100*on_time/total:.1f}%)")
    print(f"  Low risk (0-1 days): {low} ({100*low/total:.1f}%)")
    print(f"  Medium risk (1-3 days): {medium} ({100*medium/total:.1f}%)")
    print(f"  High risk (3-5 days): {high} ({100*high/total:.1f}%)")
    print(f"  Critical (>5 days): {critical} ({100*critical/total:.1f}%)")
    
    # Category analysis if available
    if 'product_category' in df.columns:
        print("\nDelay by Product Category:")
        for category in df['product_category'].unique():
            mask = df['product_category'] == category
            cat_delays = predictions[mask]
            print(f"  {category}: mean={np.mean(cat_delays):.2f}, median={np.median(cat_delays):.2f}")
    
    # Supplier analysis if available
    if 'supplier_id' in df.columns:
        print("\nTop 5 Suppliers with Highest Average Delays:")
        supplier_delays = df.groupby('supplier_id')['predicted_delay_days'].mean().sort_values(ascending=False)
        for i, (supplier, delay) in enumerate(supplier_delays.head(5).items(), 1):
            print(f"  {i}. Supplier {supplier}: {delay:.2f} days")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Supply Chain Delay Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python cli.py train --model-type ensemble --samples 1000
  
  # Make predictions
  python cli.py predict --model supply_chain_model.pkl --input data.csv
  
  # Analyze results
  python cli.py analyze predictions.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--model-type', default='ensemble',
                             choices=['ensemble', 'xgboost', 'lightgbm', 'random_forest'],
                             help='Type of model to train')
    train_parser.add_argument('--data-file', help='Path to training data CSV file')
    train_parser.add_argument('--samples', type=int, default=1000,
                             help='Number of synthetic samples to generate if no data file')
    train_parser.add_argument('--output-model', help='Path to save model')
    train_parser.add_argument('--output-engineer', help='Path to save feature engineer')
    train_parser.add_argument('--output-results', help='Path to save results JSON')
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--engineer', required=True, help='Path to feature engineer')
    predict_parser.add_argument('--input-file', required=True, help='Input CSV file with shipment data')
    predict_parser.add_argument('--output', help='Output CSV file for predictions')
    predict_parser.set_defaults(func=predict_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze prediction results')
    analyze_parser.add_argument('file', help='CSV file with predictions')
    analyze_parser.set_defaults(func=analyze_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == '__main__':
    main()
