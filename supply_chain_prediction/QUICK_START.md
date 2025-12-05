# Quick Start Guide - Supply Chain Delay Prediction

## 5-Minute Quick Start

### 1. Install Dependencies
```bash
cd /workspaces/project-defense/supply_chain_prediction
pip install -r requirements.txt
```

### 2. Run the Tutorial Notebook
```bash
# Open tutorial.ipynb in VS Code and run all cells
# This demonstrates the complete pipeline
```

### 3. Train Your Own Model
```python
from pipeline import SupplyChainPredictionPipeline

# Initialize
pipeline = SupplyChainPredictionPipeline(model_type='ensemble')

# Load data
data = pipeline.load_data(n_samples=1000)

# Prepare features
train_X, train_y, val_X, val_y, test_X, test_y, features = pipeline.prepare_features()

# Train
pipeline.train_model(train_X, train_y, val_X, val_y)

# Evaluate
results = pipeline.evaluate(test_X, test_y, pipeline.test_data)

# Visualize
pipeline.visualize_results(test_X, test_y, pipeline.test_data, features)
```

### 4. Make Predictions
```python
# Single prediction
predictions = pipeline.predict(X_test)

# With uncertainty
predictions, uncertainty = pipeline.predict(X_test, return_uncertainty=True)

# High-risk identification
high_risk = pipeline.identify_high_risk(X_test, pipeline.test_data)
```

### 5. Save and Load Models
```python
import joblib

# Save
joblib.dump(pipeline.model, 'my_model.pkl')
joblib.dump(pipeline.feature_engineer, 'feature_engineer.pkl')

# Load
model = joblib.load('my_model.pkl')
engineer = joblib.load('feature_engineer.pkl')
predictions = model.predict(engineer.transform(new_data))
```

## Using the CLI

### Train from Command Line
```bash
python cli.py train --model-type ensemble --samples 1000 \
    --output-model model.pkl \
    --output-engineer engineer.pkl
```

### Make Predictions
```bash
python cli.py predict \
    --model model.pkl \
    --engineer engineer.pkl \
    --input-file shipments.csv \
    --output predictions.csv
```

### Analyze Results
```bash
python cli.py analyze predictions.csv
```

## Using the REST API

### Start Server
```bash
python api.py model.pkl engineer.pkl
```

### Make Predictions via API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "shipments": [{
      "supplier_id": 1,
      "warehouse_id": 1,
      "product_category": "Electronics",
      "order_quantity": 500,
      "order_value": 50000,
      "supplier_reliability_score": 0.9,
      "distance_km": 1000,
      "transportation_mode": "Road",
      "weather_condition": "Clear",
      "fuel_price_index": 1.1,
      "port_congestion_score": 0.3,
      "customs_clearance_hours": 8,
      "scheduled_delivery_days": 7,
      "historical_delay_rate": 0.1,
      "supplier_inventory_level": 0.8
    }]
  }'
```

### Check Health
```bash
curl http://localhost:5000/api/health
```

## File Structure

```
supply_chain_prediction/
├── data_loader.py              # Data generation and loading
├── feature_engineering.py      # Feature engineering utilities
├── models.py                   # Model implementations
├── evaluation.py               # Evaluation and analysis
├── visualization.py            # Visualization tools
├── pipeline.py                 # Main pipeline
├── api.py                      # REST API
├── cli.py                      # Command-line interface
├── tutorial.ipynb              # Complete tutorial notebook
├── README.md                   # Full documentation
├── requirements.txt            # Dependencies
└── QUICK_START.md              # This file
```

## Common Use Cases

### Use Case 1: Identify High-Risk Shipments
```python
high_risk = pipeline.identify_high_risk(X_test, pipeline.test_data)
print(high_risk[['supplier_id', 'predicted_delay', 'risk_level']])
```

### Use Case 2: Analyze by Category
```python
analyzer = DelayAnalyzer()
category_analysis = analyzer.analyze_by_category(test_data, predictions)
print(category_analysis)
```

### Use Case 3: Get Delay Forecasts
```python
predictions, uncertainty = model.predict_with_uncertainty(X_new)
for pred, unc in zip(predictions, uncertainty):
    print(f"Predicted delay: {pred:.1f} ± {1.96*unc:.1f} days")
```

### Use Case 4: Early Warning System
```python
early_warning = analyzer.calculate_early_warning_metrics(df, predictions)
print(f"Early Warning Recall: {early_warning['early_warning_recall']:.2%}")
```

## Performance Tips

1. **Data Quality**: Ensure consistent data formats and complete records
2. **Feature Engineering**: Add domain-specific features for better predictions
3. **Model Selection**: Use Ensemble for best accuracy, XGBoost for speed
4. **Hyperparameter Tuning**: Adjust learning rates and tree depths
5. **Regular Retraining**: Update model monthly or when performance drops

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Run `pip install -r requirements.txt` |
| Model not training | Check data format matches expected columns |
| Poor predictions | Increase training data or add relevant features |
| API connection refused | Ensure API server is running on port 5000 |
| Memory errors | Reduce batch size or training data size |

## Next Steps

1. **Explore the Data**: Run the tutorial notebook to understand the dataset
2. **Experiment with Features**: Try adding new features in `feature_engineering.py`
3. **Hyperparameter Tuning**: Adjust model parameters for your specific use case
4. **Deployment**: Use the API or CLI tools for production deployment
5. **Monitoring**: Set up logging and alerts for model performance

## Resources

- **Full Documentation**: See `README.md` for comprehensive guide
- **API Documentation**: Run API and visit `/api/health`
- **Examples**: Check `tutorial.ipynb` for complete examples
- **Performance**: See `README.md` for benchmark results

## Support

For issues or questions:
1. Check the README.md troubleshooting section
2. Review the tutorial notebook for examples
3. Examine the source code docstrings
4. Create an issue with error details

---

**Ready to predict supply chain delays? Start with the tutorial notebook!**
