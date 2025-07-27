#!/usr/bin/env python3
"""
Refined Ki-67 Backend Server

Flask backend for the refined 95%+ accuracy Ki-67 ensemble classifier
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
import time
import numpy as np
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
from refined_model_manager import RefinedKi67ModelManager

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global model instance
refined_models = None

def initialize_models():
    """Initialize the refined models"""
    global refined_models
    try:
        refined_models = RefinedKi67ModelManager()
        logger.info("✅ Refined Ki-67 models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {str(e)}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("🔍 Health check endpoint called")
    
    if refined_models is None:
        logger.warning("⚠️ Models not initialized")
        return jsonify({
            'status': 'degraded',
            'message': 'Models not initialized',
            'timestamp': datetime.now().isoformat()
        }), 503
    
    model_status = refined_models.get_system_status()
    logger.info(f"✅ Health check returning: {model_status}")
    return jsonify({
        'status': model_status['status'],
        'models_loaded': model_status['loaded_models'],
        'total_models': model_status['total_models'],
        'ensemble_ready': model_status['ensemble_ready'],
        'device': model_status['device'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    if refined_models is None:
        return jsonify({'error': 'Models not initialized'}), 503
    
    try:
        models_data = {
            'available_models': refined_models.get_all_models_for_frontend(),
            'ensemble_info': refined_models.get_ensemble_info(),
            'system_status': refined_models.get_system_status(),
            'total_models': len(refined_models.get_all_models_for_frontend())
        }
        return jsonify(models_data)
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return jsonify({'error': f'Failed to get models: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict Ki-67 classification for uploaded image"""
    logger.info("🔍 Predict endpoint called")
    
    if refined_models is None:
        logger.warning("⚠️ Models not initialized")
        return jsonify({'error': 'Models not initialized'}), 503
    
    if 'file' not in request.files:
        logger.warning("⚠️ No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    confidence_threshold = float(request.form.get('confidence_threshold', 0.7))
    
    logger.info(f"📁 File received: {file.filename}, threshold: {confidence_threshold}")
    
    if file.filename == '':
        logger.warning("⚠️ Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        logger.warning(f"⚠️ Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction using refined ensemble
        prediction_result = refined_models.predict_single_image(filepath, confidence_threshold)
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        if 'error' in prediction_result:
            return jsonify({'error': prediction_result['error']}), 500
        
        # Format response
        response = {
            'prediction': prediction_result['prediction'],
            'prediction_label': prediction_result['prediction_label'],
            'probability': round(prediction_result['probability'], 4),
            'confidence': round(prediction_result['confidence'], 4),
            'confidence_label': prediction_result['confidence_label'],
            'high_confidence': prediction_result['high_confidence'],
            'ensemble_info': {
                'models_used': prediction_result['models_used'],
                'total_models': prediction_result['ensemble_info']['total_models'],
                'threshold_used': prediction_result['ensemble_info']['threshold_used'],
                'ensemble_name': refined_models.ensemble_info['name'],
                'ensemble_accuracy': refined_models.ensemble_info['high_confidence_accuracy']
            },
            'timestamp': datetime.now().isoformat(),
            'filename': file.filename
        }
        
        logger.info(f"Prediction made: {prediction_result['prediction_label']} "
                   f"(confidence: {prediction_result['confidence']:.3f}, "
                   f"high_conf: {prediction_result['high_confidence']})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/ensemble/info', methods=['GET'])
def get_ensemble_info():
    """Get detailed ensemble information"""
    if refined_models is None:
        return jsonify({'error': 'Models not initialized'}), 503
    
    try:
        ensemble_info = refined_models.get_ensemble_info()
        model_info = refined_models.get_model_info()
        
        return jsonify({
            'ensemble': ensemble_info,
            'models': model_info,
            'performance_metrics': {
                'standard_accuracy': f"{ensemble_info['standard_accuracy']}%",
                'high_confidence_accuracy': f"{ensemble_info['high_confidence_accuracy']}%",
                'coverage': f"{ensemble_info['coverage']}%",
                'optimal_threshold': ensemble_info['optimal_threshold']
            }
        })
    except Exception as e:
        logger.error(f"Error getting ensemble info: {str(e)}")
        return jsonify({'error': f'Failed to get ensemble info: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Legacy endpoint for compatibility - redirects to predict"""
    return predict()

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# Initialize models on startup
if __name__ == '__main__':
    print("🚀 Starting Refined Ki-67 Backend Server...")
    print("=" * 50)
    
    if initialize_models():
        print("✅ Models initialized successfully")
        print(f"📊 Ensemble: {refined_models.ensemble_info['name']}")
        print(f"🎯 High-confidence accuracy: {refined_models.ensemble_info['high_confidence_accuracy']}%")
        print(f"📈 Coverage: {refined_models.ensemble_info['coverage']}%")
        print(f"🔧 Models loaded: {refined_models.get_system_status()['loaded_models']}")
        print(f"🖥️  Device: {refined_models.get_system_status()['device']}")
    else:
        print("⚠️  Models failed to initialize - server running in degraded mode")
    
    print("\n📋 Available endpoints:")
    print("   GET  /api/health        - Health check and system status")
    print("   GET  /api/models        - Available models and ensemble info")
    print("   POST /api/predict       - Ki-67 classification prediction")
    print("   GET  /api/ensemble/info - Detailed ensemble information")
    print("   POST /api/analyze       - Legacy endpoint (redirects to predict)")
    
    print(f"\n🌐 Server starting on http://localhost:5001")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
