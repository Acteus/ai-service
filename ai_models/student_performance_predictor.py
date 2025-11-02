"""
Advanced Student Performance Predictor using Machine Learning
Implements academic performance prediction for ISO 21001 compliance monitoring
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StudentPerformancePredictor:
    def __init__(self, model_path='models/performance_model.pkl', scaler_path='models/performance_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = 'models/performance_encoder.pkl'
        self.model = None
        self.scaler = None
        self.encoder = None
        self.is_trained = False

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load pre-trained model and scaler if they exist"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Loaded existing performance prediction model")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing scaler")

            if os.path.exists(self.encoder_path):
                self.encoder = joblib.load(self.encoder_path)
                logger.info("Loaded existing encoder")

            self.is_trained = self.model is not None and self.scaler is not None

        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            self.is_trained = False

    def preprocess_data(self, data):
        """Preprocess student data for performance prediction"""
        # Convert single dict to list
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a dictionary, list of dictionaries, or pandas DataFrame")

        # Handle categorical variables
        categorical_cols = ['track', 'grade_level', 'semester']
        for col in categorical_cols:
            if col in df.columns:
                if self.encoder is None:
                    self.encoder = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = self.encoder.fit_transform(df[col])

        # Select features for prediction
        feature_columns = [
            'curriculum_relevance_rating', 'learning_pace_appropriateness',
            'individual_support_availability', 'learning_style_accommodation',
            'teaching_quality_rating', 'learning_environment_rating',
            'peer_interaction_satisfaction', 'academic_progress_rating',
            'skill_development_rating', 'critical_thinking_improvement',
            'problem_solving_confidence', 'attendance_rate',
            'participation_score', 'overall_satisfaction'
        ]

        # Add encoded categorical features
        feature_columns.extend([col for col in categorical_cols if col in df.columns])

        # Filter to available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        if not available_columns:
            raise ValueError("No suitable features found for performance prediction")

        # Extract features - ensure all expected features are present
        # Fill missing features with default values (0 or median)
        X = pd.DataFrame()
        for col in feature_columns:
            if col in df.columns:
                X[col] = df[col]
            else:
                # Use default value of 3.0 for rating fields (neutral), 80 for rates
                if 'rate' in col.lower() and col not in ['learning_pace_appropriateness']:
                    X[col] = 80.0
                elif 'score' in col.lower():
                    X[col] = 3.0
                else:
                    X[col] = 3.0  # Default to neutral rating

        X = X.fillna(3.0)

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Use transform() for already fitted scaler
            X_scaled = self.scaler.transform(X)

        return X_scaled, df

    def train(self, X_train, y_train, X_val=None, y_val=None, model_type='random_forest'):
        """Train the performance prediction model"""
        try:
            # Prepare data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
                    X_train_scaled, y_train, test_size=0.2, random_state=42
                )

            # Choose model
            if model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_val_scaled)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.encoder:
                joblib.dump(self.encoder, self.encoder_path)

            self.is_trained = True

            logger.info(f"Performance prediction model trained successfully with R² = {r2:.3f}")

            return {
                'success': True,
                'model_type': model_type,
                'mse': round(mse, 4),
                'r2_score': round(r2, 4),
                'feature_importance': self._get_feature_importance()
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, data):
        """Predict student performance"""
        try:
            if not self.is_trained:
                # Fallback to rule-based prediction
                return self._rule_based_prediction(data)

            # Preprocess input data
            X_scaled, _ = self.preprocess_data(data)

            # Make prediction
            prediction = self.model.predict(X_scaled)[0]

            # Ensure prediction is within valid range (0-4.0 GPA scale)
            prediction = np.clip(prediction, 0, 4.0)

            # Determine performance level
            if prediction >= 3.5:
                performance_level = 'Excellent'
                risk_level = 'Low'
            elif prediction >= 3.0:
                performance_level = 'Good'
                risk_level = 'Low'
            elif prediction >= 2.5:
                performance_level = 'Satisfactory'
                risk_level = 'Medium'
            elif prediction >= 2.0:
                performance_level = 'Needs Improvement'
                risk_level = 'High'
            else:
                performance_level = 'Critical'
                risk_level = 'High'

            # Calculate confidence based on feature consistency
            confidence = self._calculate_prediction_confidence(data)

            return {
                'prediction': performance_level,
                'predicted_gpa': round(float(prediction), 2),
                'risk_level': risk_level,
                'confidence': round(confidence, 2),
                'model_used': 'Random Forest Regressor',
                'iso_21001_insights': {
                    'performance_indicator': performance_level,
                    'intervention_required': risk_level == 'High',
                    'monitoring_frequency': 'Weekly' if risk_level == 'High' else 'Monthly'
                },
                'recommendations': self._generate_performance_recommendations(prediction, data)
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._rule_based_prediction(data)

    def _rule_based_prediction(self, data):
        """Fallback rule-based performance prediction"""
        if isinstance(data, dict):
            # Simple weighted calculation based on key factors
            attendance = data.get('attendance_rate', 0) / 100  # Convert to 0-1
            participation = data.get('participation_score', 0) / 5  # Convert to 0-1
            satisfaction = data.get('overall_satisfaction', 0) / 5
            progress = data.get('academic_progress_rating', 0) / 5

            # Weighted prediction
            prediction = (
                attendance * 0.3 +
                participation * 0.25 +
                satisfaction * 0.25 +
                progress * 0.2
            ) * 4.0  # Scale to 4.0 GPA

        else:
            prediction = 2.5  # Default

        prediction = np.clip(prediction, 0, 4.0)

        if prediction >= 3.5:
            performance_level = 'Excellent'
            risk_level = 'Low'
        elif prediction >= 3.0:
            performance_level = 'Good'
            risk_level = 'Low'
        elif prediction >= 2.5:
            performance_level = 'Satisfactory'
            risk_level = 'Medium'
        elif prediction >= 2.0:
            performance_level = 'Needs Improvement'
            risk_level = 'High'
        else:
            performance_level = 'Critical'
            risk_level = 'High'

        return {
            'prediction': performance_level,
            'predicted_gpa': round(float(prediction), 2),
            'risk_level': risk_level,
            'confidence': 0.6,
            'model_used': 'Rule-based (Fallback)',
            'iso_21001_insights': {
                'performance_indicator': performance_level,
                'intervention_required': risk_level == 'High',
                'monitoring_frequency': 'Weekly' if risk_level == 'High' else 'Monthly'
            },
            'recommendations': self._generate_performance_recommendations(prediction, data)
        }

    def _calculate_prediction_confidence(self, data):
        """Calculate confidence in the prediction based on data consistency"""
        if isinstance(data, dict):
            # Check for missing or extreme values
            key_fields = ['attendance_rate', 'participation_score', 'overall_satisfaction']
            available_fields = sum(1 for field in key_fields if field in data and data[field] is not None)
            confidence = min(0.9, 0.5 + (available_fields / len(key_fields)) * 0.4)
        else:
            confidence = 0.7

        return confidence

    def _get_feature_importance(self):
        """Get feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            # For tree-based models
            importance_dict = {}
            feature_names = ['curriculum_relevance', 'learning_pace', 'support_availability',
                           'learning_style', 'teaching_quality', 'environment', 'peer_interaction',
                           'academic_progress', 'skill_development', 'critical_thinking',
                           'problem_solving', 'attendance', 'participation', 'satisfaction']

            for i, importance in enumerate(self.model.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = round(importance, 4)

            return importance_dict
        return {}

    def _generate_performance_recommendations(self, prediction, data):
        """Generate ISO 21001 specific recommendations"""
        recommendations = []

        if prediction < 2.5:
            recommendations.append("URGENT: Implement immediate academic intervention program (ISO 21001:7.1)")
            recommendations.append("Conduct comprehensive learning needs assessment")

        if isinstance(data, dict):
            attendance = data.get('attendance_rate', 0)
            if attendance < 75:
                recommendations.append("CRITICAL: Address attendance issues through counseling and support services")

            satisfaction = data.get('overall_satisfaction', 0)
            if satisfaction < 3.0:
                recommendations.append("PRIORITY: Improve learner satisfaction through curriculum and teaching enhancements")

            participation = data.get('participation_score', 0)
            if participation < 3.0:
                recommendations.append("Increase student engagement through interactive learning activities")

        if not recommendations:
            recommendations.append("Continue monitoring performance and maintain current support levels")

        return recommendations
