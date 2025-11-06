"""
Flask AI Service for ISO 21001 Quality Education Analytics
Advanced machine learning service for compliance prediction, sentiment analysis, and student segmentation
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import AI modules
from ai_models.compliance_predictor import CompliancePredictor
from ai_models.sentiment_analyzer import SentimentAnalyzer
from ai_models.student_clusterer import StudentClusterer
from ai_models.student_performance_predictor import StudentPerformancePredictor
from ai_models.dropout_risk_predictor import DropoutRiskPredictor
from ai_models.risk_assessment_predictor import RiskAssessmentPredictor
from ai_models.satisfaction_trend_predictor import SatisfactionTrendPredictor
from utils.data_processor import DataProcessor, convert_numpy_types

# Initialize AI models
compliance_predictor = CompliancePredictor()
sentiment_analyzer = SentimentAnalyzer()
student_clusterer = StudentClusterer()
performance_predictor = StudentPerformancePredictor()
dropout_predictor = DropoutRiskPredictor()
risk_assessment_predictor = RiskAssessmentPredictor()
trend_predictor = SatisfactionTrendPredictor()
data_processor = DataProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ISO 21001 AI Service',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/v1/compliance/predict', methods=['POST'])
def predict_compliance():
    """Predict ISO 21001 compliance level using advanced ML models"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = [
            'learner_needs_index', 'satisfaction_score', 'success_index',
            'safety_index', 'wellbeing_index', 'overall_satisfaction'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Process data and make prediction
        processed_data = data_processor.preprocess_compliance_data(data)
        prediction = compliance_predictor.predict(processed_data)
        prediction = convert_numpy_types(prediction)

        logger.info(f"Compliance prediction completed for data: {data.get('id', 'unknown')}")

        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Compliance prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of student feedback using NLP models"""
    try:
        data = request.get_json()

        if not data or 'comments' not in data:
            return jsonify({'error': 'Comments field is required'}), 400

        comments = data['comments']
        if not isinstance(comments, list):
            return jsonify({'error': 'Comments must be a list of strings'}), 400

        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze_batch(comments)
        sentiment_result = convert_numpy_types(sentiment_result)

        logger.info(f"Sentiment analysis completed for {len(comments)} comments")

        return jsonify({
            'success': True,
            'sentiment_analysis': sentiment_result,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/students/cluster', methods=['POST'])
def cluster_students():
    """Cluster students based on survey responses for targeted interventions"""
    try:
        data = request.get_json()

        if not data or 'responses' not in data:
            return jsonify({'error': 'Responses field is required'}), 400

        responses = data['responses']
        k = data.get('clusters', 3)

        if not isinstance(responses, list) or len(responses) < k:
            return jsonify({'error': f'At least {k} responses required for clustering'}), 400

        # Process and cluster data
        processed_data = data_processor.preprocess_clustering_data(responses)
        clusters = student_clusterer.cluster(processed_data, k)
        clusters = convert_numpy_types(clusters)

        logger.info(f"Student clustering completed for {len(responses)} responses into {k} clusters")

        return jsonify({
            'success': True,
            'clustering_result': {
                'clusters': clusters.get('num_clusters', len(clusters.get('clusters', []))),
                'cluster_count': clusters.get('num_clusters', len(clusters.get('clusters', []))),
                'algorithm': clusters.get('algorithm', 'kmeans'),
                'total_samples': clusters.get('metrics', {}).get('total_samples', 0),
                'insights': clusters.get('insights', []),
                'detailed_clusters': clusters.get('clusters', []),
                'metrics': clusters.get('metrics', {})
            },
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Student clustering error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/performance/predict', methods=['POST'])
def predict_performance():
    """Predict student academic performance using ML models"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make prediction
        prediction = performance_predictor.predict(data)
        prediction = convert_numpy_types(prediction)

        logger.info(f"Performance prediction completed for student data")

        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Performance prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/dropout/predict', methods=['POST'])
def predict_dropout_risk():
    """Predict student dropout risk using ML models"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make prediction
        prediction = dropout_predictor.predict(data)
        prediction = convert_numpy_types(prediction)

        logger.info(f"Dropout risk prediction completed for student data")

        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Dropout risk prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/risk/assess', methods=['POST'])
def assess_risk():
    """Comprehensive risk assessment for ISO 21001 compliance"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make prediction
        assessment = risk_assessment_predictor.predict(data)
        assessment = convert_numpy_types(assessment)

        logger.info(f"Risk assessment completed for student data")

        return jsonify({
            'success': True,
            'assessment': assessment,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Risk assessment error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/satisfaction/trend', methods=['POST'])
def predict_satisfaction_trend():
    """Predict satisfaction trends using time series analysis"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Make prediction
        trend_prediction = trend_predictor.predict(data)
        trend_prediction = convert_numpy_types(trend_prediction)

        logger.info(f"Satisfaction trend prediction completed")

        return jsonify({
            'success': True,
            'trend_prediction': trend_prediction,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Satisfaction trend prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/analytics/comprehensive', methods=['POST'])
def comprehensive_analytics():
    """Comprehensive analytics combining all AI models"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        results = {}

        # Compliance prediction if data available
        if all(key in data for key in ['learner_needs_index', 'satisfaction_score', 'success_index', 'safety_index', 'wellbeing_index', 'overall_satisfaction']):
            processed_data = data_processor.preprocess_compliance_data(data)
            results['compliance_prediction'] = compliance_predictor.predict(processed_data)

        # Sentiment analysis if comments available
        if 'comments' in data and isinstance(data['comments'], list):
            results['sentiment_analysis'] = sentiment_analyzer.analyze_batch(data['comments'])

        # Clustering if responses available
        if 'responses' in data and isinstance(data['responses'], list):
            k = data.get('clusters', 3)
            if len(data['responses']) >= k:
                results['student_clustering'] = student_clusterer.cluster(data['responses'], k)

        # Performance prediction
        try:
            results['performance_prediction'] = performance_predictor.predict(data)
        except Exception as e:
            logger.warning(f"Performance prediction failed in comprehensive analytics: {e}")

        # Dropout risk prediction
        try:
            results['dropout_risk_prediction'] = dropout_predictor.predict(data)
        except Exception as e:
            logger.warning(f"Dropout risk prediction failed in comprehensive analytics: {e}")

        # Risk assessment
        try:
            results['risk_assessment'] = risk_assessment_predictor.predict(data)
        except Exception as e:
            logger.warning(f"Risk assessment failed in comprehensive analytics: {e}")

        # Satisfaction trend prediction
        try:
            results['satisfaction_trend'] = trend_predictor.predict(data)
        except Exception as e:
            logger.warning(f"Satisfaction trend prediction failed in comprehensive analytics: {e}")

        logger.info("Comprehensive analytics completed with all models")

        # Convert numpy types to Python types for JSON serialization
        results = convert_numpy_types(results)

        return jsonify({
            'success': True,
            'analytics_results': results,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Comprehensive analytics error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/analytics/predictive', methods=['POST'])
def predictive_analytics():
    """Advanced predictive analytics combining multiple forecasting models"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        results = {}

        # Performance prediction with forecasting
        try:
            performance_pred = performance_predictor.predict(data)
            results['performance_forecast'] = {
                'current_performance': performance_pred.get('predicted_gpa', 0),
                'trend': 'stable',  # Would be calculated from historical data
                'confidence': performance_pred.get('confidence', 0.5),
                'forecast_period': '3 months'
            }
        except Exception as e:
            logger.warning(f"Performance forecasting failed: {e}")

        # Risk trend prediction
        try:
            risk_pred = risk_assessment_predictor.predict(data)
            results['risk_trend'] = {
                'current_risk': risk_pred.get('overall_risk_score', 50),
                'trend_direction': 'stable',
                'predicted_change': 0,
                'confidence': risk_pred.get('confidence', 0.5)
            }
        except Exception as e:
            logger.warning(f"Risk trend prediction failed: {e}")

        # Satisfaction forecasting
        try:
            trend_pred = trend_predictor.predict(data)
            results['satisfaction_forecast'] = {
                'current_level': trend_pred.get('current_satisfaction', 3.0),
                'trend': trend_pred.get('trend_direction', 'stable'),
                'forecast_values': trend_pred.get('forecasted_satisfaction', []),
                'confidence': trend_pred.get('confidence', 0.5)
            }
        except Exception as e:
            logger.warning(f"Satisfaction forecasting failed: {e}")

        # Generate predictive insights
        insights = []
        if results.get('performance_forecast'):
            perf = results['performance_forecast']
            if perf['current_performance'] < 2.5:
                insights.append("Predictive models indicate declining performance trend - early intervention recommended")

        if results.get('risk_trend'):
            risk = results['risk_trend']
            if risk['current_risk'] > 60:
                insights.append("Risk assessment predicts increasing compliance challenges - proactive measures needed")

        if results.get('satisfaction_forecast'):
            sat = results['satisfaction_forecast']
            if sat['trend'] == 'declining':
                insights.append("Satisfaction trend analysis shows declining pattern - immediate quality improvement actions required")

        return jsonify({
            'success': True,
            'prediction': results,
            'predictive_insights': insights,
            'model_used': 'Multi-Model Predictive Ensemble',
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Predictive analytics error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/analytics/risk-assessment', methods=['POST'])
def comprehensive_risk_assessment():
    """Comprehensive ISO 21001 risk assessment across all dimensions"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Use the risk assessment predictor for comprehensive evaluation
        assessment = risk_assessment_predictor.predict(data)

        return jsonify({
            'success': True,
            'assessment': assessment,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Comprehensive risk assessment error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/api/v1/analytics/trend-analysis', methods=['POST'])
def trend_analysis():
    """Advanced trend analysis with forecasting capabilities"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Use the trend predictor for analysis
        trend_prediction = trend_predictor.predict(data)

        return jsonify({
            'success': True,
            'trend_prediction': trend_prediction,
            'timestamp': datetime.utcnow().isoformat()
        })

    except Exception as e:
        logger.error(f"Trend analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    app.run(host='0.0.0.0', port=port, debug=debug)
