"""
Advanced Dropout Risk Predictor using Machine Learning
Implements student dropout risk prediction for ISO 21001 early intervention
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import logging
from datetime import datetime
from utils.data_processor import convert_numpy_types

logger = logging.getLogger(__name__)

class DropoutRiskPredictor:
    def __init__(self, model_path='models/dropout_model.pkl', scaler_path='models/dropout_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = 'models/dropout_encoder.pkl'
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
                logger.info("Loaded existing dropout risk prediction model")

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
        """Preprocess student data for dropout risk prediction"""
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

        # Select features for prediction - MUST match training features exactly
        feature_columns = [
            'curriculum_relevance_rating', 'learning_pace_appropriateness',
            'individual_support_availability', 'teaching_quality_rating',
            'learning_environment_rating', 'peer_interaction_satisfaction',
            'academic_progress_rating', 'skill_development_rating',
            'physical_safety_rating', 'psychological_safety_rating',
            'mental_health_support_rating', 'stress_management_support',
            'attendance_rate', 'participation_score', 'overall_satisfaction'
        ]

        # Extract features - ensure all expected features are present
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
        """Train the dropout risk prediction model"""
        try:
            # Prepare data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
            else:
                X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
                    X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
                )

            # Choose model
            if model_type == 'random_forest':
                base_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
                self.model = CalibratedClassifierCV(base_model, cv=3)
            elif model_type == 'gradient_boosting':
                base_model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                self.model = CalibratedClassifierCV(base_model, cv=3)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_val_scaled)
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]

            report = classification_report(y_val, y_pred, output_dict=True)
            auc_score = roc_auc_score(y_val, y_pred_proba)

            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.encoder:
                joblib.dump(self.encoder, self.encoder_path)

            self.is_trained = True

            logger.info(f"Dropout risk prediction model trained successfully with AUC = {auc_score:.3f}")

            return {
                'success': True,
                'model_type': model_type,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'auc_score': round(auc_score, 4),
                'classification_report': report
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, data):
        """Predict dropout risk for a student"""
        try:
            if not self.is_trained:
                # Fallback to rule-based prediction
                return self._rule_based_prediction(data)

            # Preprocess input data
            X_scaled, _ = self.preprocess_data(data)

            # Make prediction
            risk_probability = self.model.predict_proba(X_scaled)[0][1]
            risk_prediction = self.model.predict(X_scaled)[0]

            # Determine risk level
            if risk_probability >= 0.8:
                risk_level = 'Very High Risk'
                intervention_urgency = 'Immediate'
            elif risk_probability >= 0.6:
                risk_level = 'High Risk'
                intervention_urgency = 'Urgent'
            elif risk_probability >= 0.4:
                risk_level = 'Moderate Risk'
                intervention_urgency = 'Priority'
            elif risk_probability >= 0.2:
                risk_level = 'Low Risk'
                intervention_urgency = 'Monitor'
            else:
                risk_level = 'Very Low Risk'
                intervention_urgency = 'Routine'

            # Calculate confidence
            confidence = self._calculate_prediction_confidence(data, risk_probability)

            result = {
                'dropout_risk': risk_level,
                'risk_probability': round(float(risk_probability), 4),
                'intervention_urgency': intervention_urgency,
                'confidence': round(confidence, 2),
                'model_used': 'Random Forest Classifier (Calibrated)',
                'iso_21001_insights': {
                    'risk_indicator': risk_level,
                    'requires_intervention': bool(risk_probability >= 0.4),
                    'monitoring_frequency': self._get_monitoring_frequency(risk_probability),
                    'compliance_impact': 'High' if risk_probability >= 0.6 else 'Medium' if risk_probability >= 0.4 else 'Low'
                },
                'risk_factors': self._identify_risk_factors(data),
                'recommendations': self._generate_risk_recommendations(risk_probability, data)
            }

            # Convert numpy types to Python native types
            return convert_numpy_types(result)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._rule_based_prediction(data)

    def _rule_based_prediction(self, data):
        """Fallback rule-based dropout risk prediction"""
        risk_score = 0
        risk_factors = []

        if isinstance(data, dict):
            # Helper function to safely convert to float
            def to_float(value, default):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default

            # Attendance factor
            attendance = to_float(data.get('attendance_rate'), 100)
            if attendance < 75:
                risk_score += 0.3
                risk_factors.append('Low attendance')

            # Satisfaction factor
            satisfaction = to_float(data.get('overall_satisfaction'), 5)
            if satisfaction < 3:
                risk_score += 0.25
                risk_factors.append('Low satisfaction')

            # Academic progress factor
            progress = to_float(data.get('academic_progress_rating'), 5)
            if progress < 3:
                risk_score += 0.2
                risk_factors.append('Poor academic progress')

            # Safety factors
            safety = to_float(data.get('physical_safety_rating'), 5)
            psych_safety = to_float(data.get('psychological_safety_rating'), 5)
            if safety < 3 or psych_safety < 3:
                risk_score += 0.15
                risk_factors.append('Safety concerns')

            # Participation factor
            participation = to_float(data.get('participation_score'), 5)
            if participation < 3:
                risk_score += 0.1
                risk_factors.append('Low participation')

        risk_probability = min(risk_score, 1.0)

        if risk_probability >= 0.8:
            risk_level = 'Very High Risk'
            intervention_urgency = 'Immediate'
        elif risk_probability >= 0.6:
            risk_level = 'High Risk'
            intervention_urgency = 'Urgent'
        elif risk_probability >= 0.4:
            risk_level = 'Moderate Risk'
            intervention_urgency = 'Priority'
        elif risk_probability >= 0.2:
            risk_level = 'Low Risk'
            intervention_urgency = 'Monitor'
        else:
            risk_level = 'Very Low Risk'
            intervention_urgency = 'Routine'

        return {
            'dropout_risk': risk_level,
            'risk_probability': round(float(risk_probability), 4),
            'intervention_urgency': intervention_urgency,
            'confidence': 0.6,
            'model_used': 'Rule-based (Fallback)',
            'iso_21001_insights': {
                'risk_indicator': risk_level,
                'requires_intervention': risk_probability >= 0.4,
                'monitoring_frequency': self._get_monitoring_frequency(risk_probability),
                'compliance_impact': 'High' if risk_probability >= 0.6 else 'Medium' if risk_probability >= 0.4 else 'Low'
            },
            'risk_factors': risk_factors,
            'recommendations': self._generate_risk_recommendations(risk_probability, data)
        }

    def _calculate_prediction_confidence(self, data, risk_probability):
        """Calculate confidence in the prediction"""
        base_confidence = 0.7

        # Adjust based on data completeness
        if isinstance(data, dict):
            key_fields = ['attendance_rate', 'overall_satisfaction', 'academic_progress_rating']
            available_fields = sum(1 for field in key_fields if field in data and data[field] is not None)
            completeness_factor = available_fields / len(key_fields)
            base_confidence += completeness_factor * 0.2

        # Adjust based on risk probability extremes
        if risk_probability < 0.1 or risk_probability > 0.9:
            base_confidence += 0.1  # More confident in extreme predictions

        return min(base_confidence, 0.95)

    def _get_monitoring_frequency(self, risk_probability):
        """Determine monitoring frequency based on risk level"""
        if risk_probability >= 0.8:
            return 'Daily'
        elif risk_probability >= 0.6:
            return 'Weekly'
        elif risk_probability >= 0.4:
            return 'Bi-weekly'
        elif risk_probability >= 0.2:
            return 'Monthly'
        else:
            return 'Quarterly'

    def _identify_risk_factors(self, data):
        """Identify specific risk factors from the data"""
        risk_factors = []

        if isinstance(data, dict):
            # Helper function to safely convert to float
            def to_float(value, default):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default

            if to_float(data.get('attendance_rate'), 100) < 75:
                risk_factors.append('Low attendance rate')

            if to_float(data.get('overall_satisfaction'), 5) < 3:
                risk_factors.append('Low overall satisfaction')

            if to_float(data.get('academic_progress_rating'), 5) < 3:
                risk_factors.append('Poor academic progress')

            if to_float(data.get('physical_safety_rating'), 5) < 3:
                risk_factors.append('Physical safety concerns')

            if to_float(data.get('psychological_safety_rating'), 5) < 3:
                risk_factors.append('Psychological safety concerns')

            if to_float(data.get('mental_health_support_rating'), 5) < 3:
                risk_factors.append('Mental health concerns')

            if to_float(data.get('participation_score'), 5) < 3:
                risk_factors.append('Low class participation')

        return risk_factors

    def _generate_risk_recommendations(self, risk_probability, data):
        """Generate ISO 21001 specific recommendations"""
        recommendations = []

        if risk_probability >= 0.8:
            recommendations.append("URGENT: Immediate intervention required - contact student and family (ISO 21001:7.2)")
            recommendations.append("Conduct comprehensive risk assessment and develop personalized support plan")
            recommendations.append("Consider academic counseling and mental health support services")

        elif risk_probability >= 0.6:
            recommendations.append("HIGH PRIORITY: Schedule counseling session within one week (ISO 21001:7.3)")
            recommendations.append("Monitor attendance and academic performance closely")
            recommendations.append("Connect with peer support groups or mentors")

        elif risk_probability >= 0.4:
            recommendations.append("MODERATE PRIORITY: Increase monitoring and provide additional support (ISO 21001:7.1)")
            recommendations.append("Review curriculum relevance and learning pace appropriateness")
            recommendations.append("Offer tutoring or academic assistance programs")

        elif risk_probability >= 0.2:
            recommendations.append("MONITOR: Regular check-ins and early intervention opportunities")
            recommendations.append("Encourage participation in extracurricular activities")

        else:
            recommendations.append("LOW RISK: Continue standard support and monitoring")

        # Add specific recommendations based on identified risk factors
        if isinstance(data, dict):
            # Helper function to safely convert to float
            def to_float(value, default):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default

            if to_float(data.get('attendance_rate'), 100) < 75:
                recommendations.append("Address attendance issues through flexible scheduling or transportation support")

            if to_float(data.get('overall_satisfaction'), 5) < 3:
                recommendations.append("Conduct satisfaction survey and implement improvements based on feedback")

            if to_float(data.get('physical_safety_rating'), 5) < 3 or to_float(data.get('psychological_safety_rating'), 5) < 3:
                recommendations.append("Review and enhance safety protocols and support services")

        return recommendations
