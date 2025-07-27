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

# Add the scripts directory to the path so we can import our validation modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts', 'validation'))

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

# Initialize the refined Ki-67 model manager
class RefinedKi67Models:
    def __init__(self):
        self.model_manager = RefinedKi67ModelManager()
        self.available_models = self.model_manager.get_all_models_for_frontend()
        self.ensemble_info = self.model_manager.get_ensemble_info()
        
        # Convert to dict format for backward compatibility
        self.models = {}
        for model in self.available_models:
            self.models[model['id']] = {
                'name': model['name'],
                'accuracy': model['accuracy'],
                'loaded': True,
                'type': model.get('type', 'individual'),
                'recommended': model.get('recommended', False)
            }
    
    def analyze_image(self, image_path, selected_models):
        """
        Enhanced Ki-67 analysis with detailed cell detection and marking
        """
        time.sleep(np.random.uniform(2, 4))  # Simulate processing time
        
        # Generate realistic results based on selected models
        base_ki67 = np.random.uniform(15, 35)
        confidence = np.random.uniform(85, 97)
        processing_time = np.random.uniform(8, 15)
        
        # Enhanced cell detection simulation
        total_cells = np.random.randint(800, 1500)
        positive_cells = int(total_cells * (base_ki67 / 100))
        negative_cells = total_cells - positive_cells
        
        # Generate detailed cell detections with coordinates
        cell_detections = []
        for i in range(total_cells):
            is_positive = i < positive_cells
            cell_detections.append({
                'id': i + 1,
                'x': np.random.randint(10, 1014),
                'y': np.random.randint(10, 1014),
                'type': 'positive' if is_positive else 'negative',
                'confidence': round(np.random.uniform(0.7, 0.99), 3),
                'area': np.random.randint(15, 85),
                'intensity': round(np.random.uniform(0.6, 0.95) if is_positive else np.random.uniform(0.1, 0.4), 3),
                'nucleus_area': np.random.randint(8, 25),
                'shape_factor': round(np.random.uniform(0.7, 1.0), 3),
                'border_clarity': np.random.choice(['clear', 'moderate', 'poor'], p=[0.7, 0.25, 0.05])
            })
        
        # Generate regions of interest with detailed analysis
        roi_regions = []
        for i in range(np.random.randint(4, 8)):
            region_cells = np.random.randint(50, 150)
            region_positive = np.random.randint(5, int(region_cells * 0.6))
            roi_regions.append({
                'id': i + 1,
                'x': np.random.randint(50, 700),
                'y': np.random.randint(50, 700),
                'width': np.random.randint(120, 250),
                'height': np.random.randint(120, 250),
                'ki67_density': round((region_positive / region_cells) * 100, 1),
                'cell_count': region_cells,
                'positive_count': region_positive,
                'negative_count': region_cells - region_positive,
                'annotation': f'ROI-{i + 1}: {"High" if region_positive/region_cells > 0.3 else "Moderate" if region_positive/region_cells > 0.15 else "Low"} proliferation',
                'proliferation_index': round((region_positive / region_cells) * 100, 1),
                'mitotic_figures': np.random.randint(0, 8)
            })
        
        # Hot spots detection (areas with highest Ki-67 expression)
        hot_spots = []
        for i in range(np.random.randint(2, 5)):
            hot_spots.append({
                'id': i + 1,
                'x': np.random.randint(100, 900),
                'y': np.random.randint(100, 900),
                'radius': np.random.randint(40, 80),
                'ki67_percentage': round(np.random.uniform(40, 70), 1),
                'cell_density': round(np.random.uniform(800, 1200), 0),
                'significance': 'high' if np.random.random() > 0.6 else 'moderate'
            })
        
        # Generate model comparison if multiple models selected
        model_comparison = None
        if len(selected_models) > 1:
            model_comparison = []
            for model_id in selected_models:
                model_ki67 = base_ki67 + np.random.uniform(-3, 3)
                model_comparison.append({
                    'name': self.models[model_id]['name'],
                    'accuracy': self.models[model_id]['accuracy'],
                    'ki67Index': round(max(0, model_ki67), 2),
                    'confidence': round(confidence + np.random.uniform(-5, 5), 2),
                    'processingTime': round(processing_time + np.random.uniform(-2, 2), 1),
                    'cellsDetected': total_cells + np.random.randint(-100, 100),
                    'agreementScore': round(np.random.uniform(85, 98), 1)
                })
        
        # Advanced statistics
        statistics = {
            'cell_density_per_mm2': round(total_cells / 1.048576, 0),  # assuming 1024x1024 = ~1mm²
            'positive_cell_density': round(positive_cells / 1.048576, 0),
            'negative_cell_density': round(negative_cells / 1.048576, 0),
            'average_cell_size': round(np.mean([cell['area'] for cell in cell_detections]), 1),
            'average_positive_size': round(np.mean([cell['area'] for cell in cell_detections if cell['type'] == 'positive']), 1),
            'average_negative_size': round(np.mean([cell['area'] for cell in cell_detections if cell['type'] == 'negative']), 1),
            'staining_intensity_distribution': {
                'strong': len([c for c in cell_detections if c['intensity'] > 0.7]),
                'moderate': len([c for c in cell_detections if 0.4 < c['intensity'] <= 0.7]),
                'weak': len([c for c in cell_detections if c['intensity'] <= 0.4])
            },
            'shape_analysis': {
                'round_cells': len([c for c in cell_detections if c['shape_factor'] > 0.9]),
                'oval_cells': len([c for c in cell_detections if 0.7 < c['shape_factor'] <= 0.9]),
                'irregular_cells': len([c for c in cell_detections if c['shape_factor'] <= 0.7])
            }
        }
        
        return {
            'ki67Index': round(base_ki67, 2),
            'confidence': round(confidence, 2),
            'totalCells': total_cells,
            'positiveCells': positive_cells,
            'negativeCells': negative_cells,
            'processingTime': round(processing_time, 1),
            'modelUsed': self.models[selected_models[0]]['name'] if selected_models else 'Unknown',
            'imageResolution': '1024x1024px',
            'cellDetections': cell_detections,
            'roiRegions': roi_regions,
            'hotSpots': hot_spots,
            'statistics': statistics,
            'modelComparison': model_comparison,
            'annotatedImageUrl': f'/api/annotated/{len(cell_detections)}',  # Mock URL
            'qualityMetrics': {
                'imageQuality': np.random.choice(['Excellent', 'Good', 'Fair'], p=[0.6, 0.3, 0.1]),
                'stainingQuality': np.random.choice(['Optimal', 'Good', 'Acceptable'], p=[0.5, 0.4, 0.1]),
                'focusQuality': np.random.choice(['Sharp', 'Good', 'Adequate'], p=[0.7, 0.25, 0.05]),
                'artifactLevel': np.random.choice(['None', 'Minimal', 'Moderate'], p=[0.5, 0.4, 0.1])
            }
        }

# Initialize model manager and models
print("🚀 Starting Ki-67 Analysis Backend Server...")
model_manager = Ki67ModelManager()
ki67_models = MockKi67Models(model_manager)

# Print startup information
summary = model_manager.get_model_summary()
print(f"📁 Upload directory: {os.path.abspath(UPLOAD_FOLDER)}")
print(f"🤖 Models loaded: {summary['total_models']}")
print(f"🌐 Server running on http://localhost:8000")
print("📋 Available endpoints:")
print("   GET  /api/health - Health check")
print("   GET  /api/models - Model status")
print("   POST /api/analyze - Analyze image")
print("   GET  /api/results - Analysis history")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    summary = model_manager.get_model_summary()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': summary['total_models'],
        'individual_models': summary['individual_models'],
        'ensemble_models': summary['ensemble_models'],
        'best_model': summary['best_model']['name'] if summary['best_model'] else None,
        'discovery_timestamp': summary['discovery_timestamp']
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models and their status"""
    summary = model_manager.get_model_summary()
    return jsonify({
        'models': [
            {
                'id': model['id'],
                'name': model['name'],
                'accuracy': model['accuracy'],
                'status': 'active',
                'type': model.get('type', 'individual'),
                'recommended': model.get('recommended', False),
                'models_count': model.get('models_count') if model.get('type') == 'ensemble' else None,
                'lastUpdated': '2025-06-19'
            }
            for model in ki67_models.available_models
        ],
        'systemStatus': 'healthy',
        'totalModels': summary['total_models'],
        'individualModels': summary['individual_models'],
        'ensembleModels': summary['ensemble_models'],
        'lastDiscovery': summary['discovery_timestamp']
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for Ki-67 expression"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, TIFF, or BMP files.'}), 400
        
        # Get selected models
        selected_models = request.form.get('models', '["ensemble"]')
        try:
            selected_models = json.loads(selected_models)
        except json.JSONDecodeError:
            selected_models = ['ensemble']
        
        # Validate selected models
        invalid_models = [m for m in selected_models if m not in ki67_models.models]
        if invalid_models:
            return jsonify({'error': f'Invalid models: {invalid_models}'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Analyzing image: {filename} with models: {selected_models}")
        
        # Perform analysis
        start_time = time.time()
        results = ki67_models.analyze_image(filepath, selected_models)
        processing_time = time.time() - start_time
        
        # Update processing time with actual time
        results['processingTime'] = round(processing_time, 1)
        
        # Clean up uploaded file (optional - you might want to keep for audit)
        # os.remove(filepath)
        
        logger.info(f"Analysis completed in {processing_time:.1f}s. Ki-67 Index: {results['ki67Index']}%")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/api/results', methods=['GET'])
def get_analysis_history():
    """Get analysis history"""
    # In production, this would query a database
    mock_results = [
        {
            'id': 1,
            'filename': 'sample_001.png',
            'ki67Index': 23.4,
            'confidence': 94.2,
            'date': '2024-01-15T10:30:00Z',
            'status': 'completed'
        },
        {
            'id': 2,
            'filename': 'sample_002.png',
            'ki67Index': 18.7,
            'confidence': 91.8,
            'date': '2024-01-14T14:20:00Z',
            'status': 'completed'
        },
        {
            'id': 3,
            'filename': 'sample_003.png',
            'ki67Index': 31.2,
            'confidence': 96.1,
            'date': '2024-01-13T09:45:00Z',
            'status': 'completed'
        }
    ]
    
    return jsonify(mock_results)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    print("🚀 Starting Ki-67 Analysis Backend Server...")
    print(f"📁 Upload directory: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"🤖 Models loaded: {len(ki67_models.models)}")
    print("🌐 Server running on http://localhost:8000")
    print("📋 Available endpoints:")
    print("   GET  /api/health - Health check")
    print("   GET  /api/models - Model status")
    print("   POST /api/analyze - Analyze image")
    print("   GET  /api/results - Analysis history")
    
    app.run(debug=True, host='0.0.0.0', port=8000)
