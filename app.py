from flask import Flask, render_template, jsonify, request
from datetime import datetime
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Sample data structure for WAF ML metrics
dashboard_data = {
    'project': 'Swavlamban WAF ML',
    'last_updated': datetime.utcnow().isoformat(),
    'metrics': {
        'total_threats_detected': 1247,
        'false_positives': 45,
        'detection_accuracy': 97.8,
        'average_response_time_ms': 2.3
    },
    'recent_alerts': [
        {
            'id': 1,
            'threat_type': 'SQL Injection',
            'severity': 'high',
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'blocked'
        },
        {
            'id': 2,
            'threat_type': 'XSS Attack',
            'severity': 'medium',
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'blocked'
        },
        {
            'id': 3,
            'threat_type': 'Path Traversal',
            'severity': 'high',
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'blocked'
        }
    ],
    'model_performance': {
        'precision': 0.985,
        'recall': 0.972,
        'f1_score': 0.978,
        'training_samples': 50000
    }
}

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', data=dashboard_data)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint to get current metrics"""
    dashboard_data['last_updated'] = datetime.utcnow().isoformat()
    return jsonify(dashboard_data['metrics'])

@app.route('/api/alerts')
def get_alerts():
    """API endpoint to get recent alerts"""
    return jsonify(dashboard_data['recent_alerts'])

@app.route('/api/model-performance')
def get_model_performance():
    """API endpoint to get model performance metrics"""
    return jsonify(dashboard_data['model_performance'])

@app.route('/api/dashboard')
def get_dashboard_data():
    """API endpoint to get all dashboard data"""
    dashboard_data['last_updated'] = datetime.utcnow().isoformat()
    return jsonify(dashboard_data)

@app.route('/api/threat-analysis', methods=['POST'])
def analyze_threat():
    """Endpoint to analyze a potential threat"""
    data = request.get_json()
    
    if not data or 'payload' not in data:
        return jsonify({'error': 'Missing payload'}), 400
    
    # Placeholder for threat analysis logic
    analysis_result = {
        'payload': data.get('payload'),
        'threat_detected': True,
        'threat_type': 'Potential Attack Vector',
        'confidence': 0.89,
        'timestamp': datetime.utcnow().isoformat(),
        'recommendation': 'Block and log'
    }
    
    return jsonify(analysis_result)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Development configuration
    debug_mode = os.environ.get('FLASK_DEBUG', True)
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=debug_mode
    )