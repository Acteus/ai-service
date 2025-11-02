"""
Advanced Satisfaction Trend Predictor using Time Series Analysis
Implements satisfaction level prediction and trend analysis for ISO 21001 monitoring
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import os
import logging
from datetime import datetime, timedelta
from utils.data_processor import convert_numpy_types

logger = logging.getLogger(__name__)

class SatisfactionTrendPredictor:
    def __init__(self, model_path='models/satisfaction_trend_model.pkl', scaler_path='models/satisfaction_trend_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.time_series_model_path = 'models/satisfaction_ts_model.pkl'
        self.model = None
        self.scaler = None
        self.ts_model = None  # Time series model
        self.is_trained = False

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Loaded existing satisfaction trend model")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing scaler")

            if os.path.exists(self.time_series_model_path):
                self.ts_model = joblib.load(self.time_series_model_path)
                logger.info("Loaded existing time series model")

            self.is_trained = self.model is not None

        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            self.is_trained = False

    def preprocess_data(self, data, include_time_features=True):
        """Preprocess data for satisfaction trend prediction"""
        # Convert single dict to list
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a dictionary, list of dictionaries, or pandas DataFrame")

        # Handle time-based features if available
        if include_time_features and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['week_of_year'] = df['timestamp'].dt.isocalendar().week

        # Select features for trend prediction - MUST match training features exactly
        feature_columns = [
            'curriculum_relevance_rating', 'learning_pace_appropriateness',
            'individual_support_availability', 'teaching_quality_rating',
            'learning_environment_rating', 'peer_interaction_satisfaction',
            'academic_progress_rating', 'skill_development_rating',
            'physical_safety_rating', 'mental_health_support_rating',
            'attendance_rate', 'participation_score'
        ]

        # NOTE: Time-based features were NOT used in training, so don't add them during prediction
        # if include_time_features:
        #     time_features = ['month', 'quarter', 'day_of_year', 'week_of_year']
        #     feature_columns.extend([col for col in time_features if col in df.columns])

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

        y = df.get('overall_satisfaction', pd.Series([3.0] * len(df)))  # Default satisfaction

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            # Use transform() for already fitted scaler
            X_scaled = self.scaler.transform(X)

        return X_scaled, y, df

    def train(self, X_train, y_train, X_val=None, y_val=None, model_type='gradient_boosting'):
        """Train the satisfaction trend prediction model"""
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
            if model_type == 'gradient_boosting':
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
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

            self.is_trained = True

            logger.info(f"Satisfaction trend model trained successfully with R² = {r2:.3f}")

            return {
                'success': True,
                'model_type': model_type,
                'mae': round(mae, 4),
                'mse': round(mse, 4),
                'r2_score': round(r2, 4),
                'feature_importance': self._get_feature_importance()
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_time_series(self, time_series_data, periods=12):
        """Train time series model for trend forecasting"""
        try:
            if not isinstance(time_series_data, pd.Series):
                time_series_data = pd.Series(time_series_data)

            # Ensure we have enough data
            if len(time_series_data) < periods:
                logger.warning("Insufficient data for time series training")
                return {'success': False, 'error': 'Insufficient time series data'}

            # Try different time series models
            models_to_try = [
                ('ARIMA', lambda: ARIMA(time_series_data, order=(1, 1, 1))),
                ('SARIMA', lambda: SARIMAX(time_series_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))),
                ('ExponentialSmoothing', lambda: ExponentialSmoothing(time_series_data, seasonal_periods=12, trend='add', seasonal='add'))
            ]

            best_model = None
            best_aic = float('inf')

            for model_name, model_func in models_to_try:
                try:
                    model = model_func()
                    result = model.fit(disp=False)

                    if hasattr(result, 'aic') and result.aic < best_aic:
                        best_model = result
                        best_aic = result.aic

                except Exception as e:
                    logger.warning(f"Failed to fit {model_name}: {e}")
                    continue

            if best_model is None:
                return {'success': False, 'error': 'Could not fit any time series model'}

            # Save the best model
            self.ts_model = best_model
            joblib.dump(self.ts_model, self.time_series_model_path)

            return {
                'success': True,
                'model_type': type(best_model.model).__name__,
                'aic': round(best_aic, 2) if best_aic != float('inf') else None
            }

        except Exception as e:
            logger.error(f"Time series training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, data, forecast_periods=3):
        """Predict satisfaction trends"""
        try:
            if not self.is_trained:
                # Fallback to rule-based prediction
                result = self._rule_based_prediction(data, forecast_periods)
                return convert_numpy_types(result)

            # Preprocess input data
            X_scaled, _, _ = self.preprocess_data(data, include_time_features=False)

            # Make current prediction
            current_prediction = self.model.predict(X_scaled)[0]

            # Ensure prediction is within valid range (1-5 scale)
            current_prediction = np.clip(current_prediction, 1, 5)

            # Generate trend analysis
            trend_analysis = self._analyze_trends(data, current_prediction)

            # Forecast future satisfaction if time series model is available
            forecast = None
            if self.ts_model is not None and hasattr(self.ts_model, 'forecast'):
                try:
                    forecast = self.ts_model.forecast(steps=forecast_periods)
                    forecast = np.clip(forecast, 1, 5).tolist()
                except Exception as e:
                    logger.warning(f"Forecasting failed: {e}")

            # Calculate confidence
            confidence = self._calculate_trend_confidence(data, current_prediction)

            result = {
                'current_satisfaction': round(float(current_prediction), 2),
                'trend_direction': trend_analysis['direction'],
                'trend_strength': trend_analysis['strength'],
                'forecasted_satisfaction': forecast,
                'confidence': round(confidence, 2),
                'model_used': 'Gradient Boosting Regressor',
                'iso_21001_insights': {
                    'satisfaction_trend': self._interpret_trend(trend_analysis),
                    'intervention_needed': self._assess_intervention_need(trend_analysis, current_prediction),
                    'monitoring_recommendation': self._get_monitoring_recommendation(trend_analysis),
                    'predictive_stability': self._assess_stability(data)
                },
                'trend_analysis': trend_analysis,
                'recommendations': self._generate_trend_recommendations(trend_analysis, current_prediction)
            }

            # Convert all numpy types to Python native types for JSON serialization
            return convert_numpy_types(result)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            result = self._rule_based_prediction(data, forecast_periods)
            return convert_numpy_types(result)

    def _rule_based_prediction(self, data, forecast_periods=3):
        """Fallback rule-based satisfaction trend prediction"""
        if isinstance(data, dict):
            # Calculate current satisfaction based on key indicators
            satisfaction_factors = [
                data.get('overall_satisfaction', 3),
                data.get('teaching_quality_rating', 3),
                data.get('learning_environment_rating', 3),
                data.get('peer_interaction_satisfaction', 3),
                data.get('overall_wellbeing_rating', 3)
            ]
            current_prediction = np.mean(satisfaction_factors)
        else:
            current_prediction = 3.0

        current_prediction = np.clip(current_prediction, 1, 5)

        # Simple trend analysis
        trend_analysis = {
            'direction': 'stable',
            'strength': 'moderate',
            'change_rate': 0.0,
            'volatility': 0.5
        }

        # Generate simple forecast (assume slight improvement or decline)
        if current_prediction > 3.5:
            forecast_trend = -0.1  # Slight decline
        elif current_prediction < 2.5:
            forecast_trend = 0.2  # Improvement trend
        else:
            forecast_trend = 0.0  # Stable

        forecast = []
        for i in range(forecast_periods):
            next_val = current_prediction + (forecast_trend * (i + 1) * 0.1)
            forecast.append(round(np.clip(next_val, 1, 5), 2))

        return {
            'current_satisfaction': round(float(current_prediction), 2),
            'trend_direction': trend_analysis['direction'],
            'trend_strength': trend_analysis['strength'],
            'forecasted_satisfaction': forecast,
            'confidence': 0.6,
            'model_used': 'Rule-based (Fallback)',
            'iso_21001_insights': {
                'satisfaction_trend': 'Stable with moderate confidence',
                'intervention_needed': current_prediction < 3.0,
                'monitoring_recommendation': 'Monthly monitoring recommended',
                'predictive_stability': 'Moderate stability'
            },
            'trend_analysis': trend_analysis,
            'recommendations': self._generate_trend_recommendations(trend_analysis, current_prediction)
        }

    def _analyze_trends(self, data, current_prediction):
        """Analyze satisfaction trends from historical data"""
        trend_analysis = {
            'direction': 'stable',
            'strength': 'weak',
            'change_rate': 0.0,
            'volatility': 0.0,
            'seasonal_pattern': False
        }

        if isinstance(data, list) and len(data) > 1:
            # Extract satisfaction values over time
            satisfaction_values = []
            for item in data:
                if isinstance(item, dict):
                    sat = item.get('overall_satisfaction', current_prediction)
                    satisfaction_values.append(sat)

            if len(satisfaction_values) > 1:
                # Calculate trend direction
                values_array = np.array(satisfaction_values)
                slope = np.polyfit(range(len(values_array)), values_array, 1)[0]

                if slope > 0.05:
                    trend_analysis['direction'] = 'improving'
                elif slope < -0.05:
                    trend_analysis['direction'] = 'declining'
                else:
                    trend_analysis['direction'] = 'stable'

                # Calculate trend strength
                slope_abs = abs(slope)
                if slope_abs > 0.15:
                    trend_analysis['strength'] = 'strong'
                elif slope_abs > 0.08:
                    trend_analysis['strength'] = 'moderate'
                else:
                    trend_analysis['strength'] = 'weak'

                trend_analysis['change_rate'] = round(float(slope), 4)
                trend_analysis['volatility'] = round(float(np.std(values_array)), 4)

                # Check for seasonal patterns (simplified)
                if len(values_array) >= 12:
                    # Look for monthly patterns
                    monthly_avg = []
                    for i in range(0, len(values_array), 4):  # Assuming quarterly data
                        monthly_avg.append(np.mean(values_array[i:i+4]))

                    if len(monthly_avg) >= 3:
                        seasonal_variation = np.std(monthly_avg)
                        trend_analysis['seasonal_pattern'] = bool(seasonal_variation > 0.3)

        # Convert all numpy types to Python native types
        return convert_numpy_types(trend_analysis)

    def _calculate_trend_confidence(self, data, prediction):
        """Calculate confidence in trend prediction"""
        base_confidence = 0.7

        # Adjust based on data availability
        if isinstance(data, list):
            data_points = len(data)
            if data_points > 10:
                base_confidence += 0.1
            elif data_points < 3:
                base_confidence -= 0.2

        # Adjust based on prediction extremity
        if prediction < 2.0 or prediction > 4.5:
            base_confidence += 0.05  # More confident in extreme predictions

        return min(base_confidence, 0.9)

    def _interpret_trend(self, trend_analysis):
        """Interpret trend analysis results"""
        direction = trend_analysis['direction']
        strength = trend_analysis['strength']

        if direction == 'improving':
            if strength == 'strong':
                return 'Strong upward trend in satisfaction levels'
            elif strength == 'moderate':
                return 'Moderate improvement in satisfaction'
            else:
                return 'Slight improvement in satisfaction levels'
        elif direction == 'declining':
            if strength == 'strong':
                return 'Strong downward trend in satisfaction levels'
            elif strength == 'moderate':
                return 'Moderate decline in satisfaction'
            else:
                return 'Slight decline in satisfaction levels'
        else:
            return 'Stable satisfaction levels with no significant trend'

    def _assess_intervention_need(self, trend_analysis, current_satisfaction):
        """Assess if intervention is needed based on trend and current level"""
        if current_satisfaction < 2.5:
            return True  # Low satisfaction requires intervention

        if trend_analysis['direction'] == 'declining' and trend_analysis['strength'] in ['strong', 'moderate']:
            return True  # Declining trend requires attention

        return False

    def _get_monitoring_recommendation(self, trend_analysis):
        """Get monitoring recommendations based on trend"""
        if trend_analysis['direction'] == 'declining' and trend_analysis['strength'] == 'strong':
            return 'Weekly monitoring required'
        elif trend_analysis['direction'] == 'declining' or trend_analysis['strength'] == 'moderate':
            return 'Bi-weekly monitoring recommended'
        elif trend_analysis['volatility'] > 0.8:
            return 'Monthly monitoring with volatility tracking'
        else:
            return 'Monthly routine monitoring'

    def _assess_stability(self, data):
        """Assess predictive stability"""
        if isinstance(data, list) and len(data) > 1:
            satisfaction_values = [item.get('overall_satisfaction', 3) if isinstance(item, dict) else 3 for item in data]
            volatility = np.std(satisfaction_values)

            if volatility < 0.3:
                return 'High stability'
            elif volatility < 0.6:
                return 'Moderate stability'
            else:
                return 'Low stability - high variability'
        else:
            return 'Unknown stability - limited data'

    def _generate_trend_recommendations(self, trend_analysis, current_satisfaction):
        """Generate recommendations based on trend analysis"""
        recommendations = []

        if trend_analysis['direction'] == 'declining':
            if trend_analysis['strength'] == 'strong':
                recommendations.extend([
                    'URGENT: Implement immediate satisfaction improvement measures (ISO 21001:7.1.2)',
                    'Conduct comprehensive satisfaction survey to identify root causes',
                    'Develop urgent action plan with stakeholder involvement'
                ])
            else:
                recommendations.extend([
                    'PRIORITY: Monitor satisfaction trends closely and identify decline factors',
                    'Implement targeted improvements in identified weak areas',
                    'Increase communication and feedback mechanisms'
                ])

        elif trend_analysis['direction'] == 'improving':
            recommendations.extend([
                'Continue current successful practices and identify best practices',
                'Monitor to ensure trend sustainability',
                'Share successful strategies across the organization'
            ])

        else:  # stable
            if current_satisfaction < 3.5:
                recommendations.extend([
                    'Focus on incremental improvements to increase satisfaction',
                    'Regular monitoring and periodic satisfaction assessments',
                    'Implement continuous improvement initiatives'
                ])
            else:
                recommendations.extend([
                    'Maintain current high satisfaction levels',
                    'Regular monitoring to ensure continued stability',
                    'Focus on excellence and innovation'
                ])

        # Add volatility-based recommendations
        if trend_analysis.get('volatility', 0) > 0.8:
            recommendations.append('Address satisfaction volatility through consistent service delivery')

        return recommendations

    def _get_feature_importance(self):
        """Get feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = {}
            feature_names = [
                'curriculum_relevance', 'learning_pace', 'support_availability',
                'learning_style', 'teaching_quality', 'environment', 'peer_interaction',
                'extracurricular', 'academic_progress', 'skill_development',
                'critical_thinking', 'problem_solving', 'physical_safety',
                'psychological_safety', 'bullying_prevention', 'emergency_preparedness',
                'mental_health', 'stress_management', 'physical_health', 'wellbeing',
                'attendance', 'participation', 'month', 'quarter', 'day_of_year', 'week_of_year'
            ]

            for i, importance in enumerate(self.model.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = round(importance, 4)

            return importance_dict
        return {}
