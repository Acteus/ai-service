"""
Advanced Compliance Predictor using Deep Learning
Implements ISO 21001 compliance prediction with TensorFlow/Keras
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CompliancePredictor:
    def __init__(self, model_path='models/compliance_model.h5', scaler_path='models/compliance_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.is_trained = False

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load pre-trained model and scaler if they exist"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info("Loaded existing compliance prediction model")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing scaler")

            self.is_trained = self.model is not None and self.scaler is not None

        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            self.is_trained = False

    def _build_model(self, input_shape):
        """Build deep learning model for compliance prediction"""
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification: compliant/non-compliant
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the compliance prediction model"""
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

            # Build model
            input_shape = (X_train_scaled.shape[1],)
            self.model = self._build_model(input_shape)

            # Train model
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )

            # Save model and scaler
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

            self.is_trained = True

            logger.info("Compliance prediction model trained successfully")

            return {
                'success': True,
                'training_history': history.history,
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, data):
        """Predict compliance level for given data"""
        try:
            if not self.is_trained:
                # Fallback to rule-based prediction if model not trained
                return self._rule_based_prediction(data)

            # Prepare input data
            if isinstance(data, dict):
                # Convert dict to array
                features = [
                    data.get('learner_needs_index', 0),
                    data.get('satisfaction_score', 0),
                    data.get('success_index', 0),
                    data.get('safety_index', 0),
                    data.get('wellbeing_index', 0),
                    data.get('overall_satisfaction', 0)
                ]
                X = np.array([features])
            elif isinstance(data, (list, np.ndarray)):
                X = np.array([data] if len(np.array(data).shape) == 1 else data)
            else:
                raise ValueError("Invalid input data format")

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make prediction
            prediction_prob = self.model.predict(X_scaled, verbose=0)[0][0]

            # Determine compliance level
            if prediction_prob >= 0.9:
                compliance_level = 'High ISO 21001 Compliance'
                risk_level = 'Low'
                confidence = 0.95
            elif prediction_prob >= 0.6:
                compliance_level = 'Moderate ISO 21001 Compliance'
                risk_level = 'Medium'
                confidence = 0.75
            else:
                compliance_level = 'Low ISO 21001 Compliance'
                risk_level = 'High'
                confidence = 0.55

            # Calculate weighted score for backward compatibility
            weighted_score = prediction_prob * 5  # Scale to 0-5 range

            return {
                'prediction': compliance_level,
                'risk_level': risk_level,
                'confidence': round(confidence, 2),
                'weighted_score': round(weighted_score, 2),
                'prediction_probability': round(float(prediction_prob), 4),
                'indices_used': data if isinstance(data, dict) else {},
                'model_used': 'Deep Learning (TensorFlow)',
                'analysis': {
                    'score_variance': self._calculate_score_variance(data),
                    'recommended_actions': self._generate_recommendations(weighted_score, data)
                }
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to rule-based prediction
            return self._rule_based_prediction(data)

    def _rule_based_prediction(self, data):
        """Fallback rule-based prediction when model is not available"""
        # Extract values (same logic as original PHP implementation)
        learner_needs = float(data.get('learner_needs_index', 0) if isinstance(data, dict) else data[0])
        satisfaction = float(data.get('satisfaction_score', 0) if isinstance(data, dict) else data[1])
        success = float(data.get('success_index', 0) if isinstance(data, dict) else data[2])
        safety = float(data.get('safety_index', 0) if isinstance(data, dict) else data[3])
        wellbeing = float(data.get('wellbeing_index', 0) if isinstance(data, dict) else data[4])
        overall = float(data.get('overall_satisfaction', 0) if isinstance(data, dict) else data[5])

        # Calculate weighted compliance score
        weighted_score = (
            learner_needs * 0.15 +
            satisfaction * 0.25 +
            success * 0.20 +
            safety * 0.20 +
            wellbeing * 0.15 +
            overall * 0.05
        )

        # Determine compliance prediction
        if weighted_score >= 4.5:
            prediction = 'High ISO 21001 Compliance'
            risk_level = 'Low'
            confidence = 0.95
        elif weighted_score >= 3.0:
            prediction = 'Moderate ISO 21001 Compliance'
            risk_level = 'Medium'
            confidence = 0.75
        else:
            prediction = 'Low ISO 21001 Compliance'
            risk_level = 'High'
            confidence = 0.55

        return {
            'prediction': prediction,
            'risk_level': risk_level,
            'confidence': round(confidence, 2),
            'weighted_score': round(weighted_score, 2),
            'model_used': 'Rule-based (Fallback)',
            'indices_used': data if isinstance(data, dict) else {},
            'analysis': {
                'score_variance': self._calculate_score_variance(data),
                'recommended_actions': self._generate_recommendations(weighted_score, data)
            }
        }

    def _calculate_score_variance(self, data):
        """Calculate variance across all indices"""
        if isinstance(data, dict):
            values = [float(v) for v in data.values()]
        else:
            values = [float(v) for v in data]

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return round(variance, 2)

    def _generate_recommendations(self, weighted_score, data):
        """Generate ISO 21001 specific recommendations"""
        recommendations = []

        if isinstance(data, dict):
            safety = data.get('safety_index', 0)
            wellbeing = data.get('wellbeing_index', 0)
            satisfaction = data.get('satisfaction_score', 0)
        else:
            safety = data[3] if len(data) > 3 else 0
            wellbeing = data[4] if len(data) > 4 else 0
            satisfaction = data[1] if len(data) > 1 else 0

        if safety < 3.5:
            recommendations.append('URGENT: Review and enhance safety protocols and emergency preparedness (ISO 21001:7.2)')

        if wellbeing < 3.5:
            recommendations.append('PRIORITY: Implement comprehensive wellbeing support programs (ISO 21001:7.3)')

        if satisfaction < 3.5:
            recommendations.append('IMMEDIATE: Address learner satisfaction issues through curriculum and teaching improvements (ISO 21001:7.1)')

        if weighted_score < 3.5:
            recommendations.append('CRITICAL: Conduct full ISO 21001 compliance audit and develop improvement plan')

        return recommendations
