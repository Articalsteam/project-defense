"""
REST API for supply chain delay prediction system
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global model and feature engineer
model = None
feature_engineer = None


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict delay for shipments.
    
    Expected JSON format:
    {
        "shipments": [
            {
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
            }
        ]
    }
    """
    try:
        if model is None or feature_engineer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        shipments = data.get('shipments', [])
        
        if not shipments:
            return jsonify({'error': 'No shipments provided'}), 400
        
        # Create dataframe
        df = pd.DataFrame(shipments)
        
        # Transform features
        X = feature_engineer.transform(df)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get uncertainty if available
        uncertainty = None
        if hasattr(model, 'predict_with_uncertainty'):
            _, uncertainty = model.predict_with_uncertainty(X)
        
        # Prepare response
        results = []
        for i, shipment in enumerate(shipments):
            result = {
                'shipment_id': i,
                'predicted_delay_days': float(predictions[i]),
                'is_delayed': bool(predictions[i] > 0),
                'risk_level': get_risk_level(predictions[i]),
            }
            
            if uncertainty is not None:
                result['uncertainty'] = float(uncertainty[i])
                result['confidence_lower'] = float(predictions[i] - 1.96 * uncertainty[i])
                result['confidence_upper'] = float(predictions[i] + 1.96 * uncertainty[i])
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'predictions': results,
            'summary': {
                'total_shipments': len(results),
                'delayed_count': sum(1 for r in results if r['is_delayed']),
                'high_risk_count': sum(1 for r in results if r['risk_level'] in ['High', 'Critical']),
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for CSV data.
    """
    try:
        if model is None or feature_engineer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Transform features
        X = feature_engineer.transform(df)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to dataframe
        df['predicted_delay_days'] = predictions
        df['is_delayed'] = (predictions > 0).astype(int)
        df['risk_level'] = df['predicted_delay_days'].apply(get_risk_level)
        
        # Save results
        output_path = '/tmp/predictions.csv'
        df.to_csv(output_path, index=False)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'delayed_count': int(np.sum(predictions > 0)),
            'statistics': {
                'mean_delay': float(np.mean(predictions)),
                'median_delay': float(np.median(predictions)),
                'max_delay': float(np.max(predictions)),
                'std_delay': float(np.std(predictions)),
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """Get feature importance scores."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            features = feature_engineer.feature_names
            
            importance_list = [
                {'feature': f, 'importance': float(i)}
                for f, i in zip(features, importances)
            ]
            importance_list.sort(key=lambda x: x['importance'], reverse=True)
            
            return jsonify({
                'success': True,
                'feature_importance': importance_list[:20]  # Top 20
            })
        else:
            return jsonify({'error': 'Model does not support feature importance'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about loaded model."""
    try:
        if model is None:
            return jsonify({'error': 'No model loaded'}), 500
        
        info = {
            'model_type': str(type(model).__name__),
            'is_trained': getattr(model, 'is_trained', False),
            'timestamp': datetime.now().isoformat(),
        }
        
        if hasattr(model, 'model_type'):
            info['model_type'] = model.model_type
        
        if hasattr(model, 'models'):
            info['ensemble_models'] = list(model.models.keys())
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/high-risk-analysis', methods=['POST'])
def high_risk_analysis():
    """Analyze high-risk shipments."""
    try:
        if model is None or feature_engineer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        shipments = data.get('shipments', [])
        threshold = data.get('threshold', 5.0)
        
        if not shipments:
            return jsonify({'error': 'No shipments provided'}), 400
        
        # Create dataframe
        df = pd.DataFrame(shipments)
        
        # Transform features
        X = feature_engineer.transform(df)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Identify high-risk
        high_risk_indices = np.where(predictions > threshold)[0]
        
        high_risk_shipments = []
        for idx in high_risk_indices:
            high_risk_shipments.append({
                'shipment_id': int(idx),
                'predicted_delay_days': float(predictions[idx]),
                'risk_level': get_risk_level(predictions[idx]),
                'details': shipments[idx] if idx < len(shipments) else {}
            })
        
        high_risk_shipments.sort(key=lambda x: x['predicted_delay_days'], reverse=True)
        
        return jsonify({
            'success': True,
            'high_risk_count': len(high_risk_shipments),
            'high_risk_shipments': high_risk_shipments
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def get_risk_level(delay_days: float) -> str:
    """Classify risk level based on predicted delay."""
    if delay_days <= 0:
        return 'On-time'
    elif delay_days <= 1:
        return 'Low'
    elif delay_days <= 3:
        return 'Medium'
    elif delay_days <= 5:
        return 'High'
    else:
        return 'Critical'


def load_model_and_engineer(model_path: str, engineer_path: str):
    """Load model and feature engineer from disk."""
    global model, feature_engineer
    
    try:
        model = joblib.load(model_path)
        feature_engineer = joblib.load(engineer_path)
        print(f"Model loaded from {model_path}")
        print(f"Feature engineer loaded from {engineer_path}")
    except Exception as e:
        print(f"Error loading model or feature engineer: {e}")


if __name__ == '__main__':
    import sys
    
    # Load model if path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        engineer_path = sys.argv[2] if len(sys.argv) > 2 else model_path.replace('model', 'engineer')
        load_model_and_engineer(model_path, engineer_path)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
