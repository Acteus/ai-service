"""
Data Processing Utilities for AI Service
Handles data preprocessing, validation, and transformation for ML models
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class DataProcessor:
    def __init__(self):
        pass

    def preprocess_compliance_data(self, data):
        """Preprocess data for compliance prediction"""
        try:
            if isinstance(data, dict):
                # Extract and validate required fields
                processed = {
                    'learner_needs_index': float(data.get('learner_needs_index', 0)),
                    'satisfaction_score': float(data.get('satisfaction_score', 0)),
                    'success_index': float(data.get('success_index', 0)),
                    'safety_index': float(data.get('safety_index', 0)),
                    'wellbeing_index': float(data.get('wellbeing_index', 0)),
                    'overall_satisfaction': float(data.get('overall_satisfaction', 0))
                }
            else:
                # Assume array-like input
                processed = [float(x) for x in data]

            # Validate ranges (1-5 scale)
            for key, value in processed.items():
                if not (0 <= value <= 5):
                    logger.warning(f"Value {value} for {key} is outside expected range [0,5]")

            return processed

        except Exception as e:
            logger.error(f"Error preprocessing compliance data: {e}")
            raise ValueError(f"Invalid compliance data format: {e}")

    def preprocess_clustering_data(self, responses):
        """Preprocess survey responses for clustering"""
        try:
            if not isinstance(responses, list):
                raise ValueError("Responses must be a list")

            processed_data = []

            for response in responses:
                if not isinstance(response, dict):
                    continue

                # Extract relevant features for clustering
                features = {
                    'curriculum_relevance_rating': float(response.get('curriculum_relevance_rating', 0)),
                    'learning_pace_appropriateness': float(response.get('learning_pace_appropriateness', 0)),
                    'individual_support_availability': float(response.get('individual_support_availability', 0)),
                    'learning_style_accommodation': float(response.get('learning_style_accommodation', 0)),
                    'teaching_quality_rating': float(response.get('teaching_quality_rating', 0)),
                    'learning_environment_rating': float(response.get('learning_environment_rating', 0)),
                    'peer_interaction_satisfaction': float(response.get('peer_interaction_satisfaction', 0)),
                    'extracurricular_satisfaction': float(response.get('extracurricular_satisfaction', 0)),
                    'academic_progress_rating': float(response.get('academic_progress_rating', 0)),
                    'skill_development_rating': float(response.get('skill_development_rating', 0)),
                    'critical_thinking_improvement': float(response.get('critical_thinking_improvement', 0)),
                    'problem_solving_confidence': float(response.get('problem_solving_confidence', 0)),
                    'physical_safety_rating': float(response.get('physical_safety_rating', 0)),
                    'psychological_safety_rating': float(response.get('psychological_safety_rating', 0)),
                    'bullying_prevention_effectiveness': float(response.get('bullying_prevention_effectiveness', 0)),
                    'emergency_preparedness_rating': float(response.get('emergency_preparedness_rating', 0)),
                    'mental_health_support_rating': float(response.get('mental_health_support_rating', 0)),
                    'stress_management_support': float(response.get('stress_management_support', 0)),
                    'physical_health_support': float(response.get('physical_health_support', 0)),
                    'overall_wellbeing_rating': float(response.get('overall_wellbeing_rating', 0)),
                    'overall_satisfaction': float(response.get('overall_satisfaction', 0)),
                    'grade_average': float(response.get('grade_average', 0)),
                    'attendance_rate': float(response.get('attendance_rate', 0)),
                    'participation_score': float(response.get('participation_score', 0))
                }

                processed_data.append(features)

            if not processed_data:
                raise ValueError("No valid responses found for clustering")

            return processed_data

        except Exception as e:
            logger.error(f"Error preprocessing clustering data: {e}")
            raise ValueError(f"Invalid clustering data format: {e}")

    def calculate_composite_indices(self, response):
        """Calculate composite indices from individual ratings"""
        try:
            indices = {}

            # Learner Needs Index (Curriculum, Pace, Support, Accommodation)
            learner_needs_fields = [
                'curriculum_relevance_rating',
                'learning_pace_appropriateness',
                'individual_support_availability',
                'learning_style_accommodation'
            ]
            learner_values = [float(response.get(field, 0)) for field in learner_needs_fields]
            indices['learner_needs_index'] = round(np.mean(learner_values), 2) if learner_values else 0.0

            # Satisfaction Score (Teaching, Environment, Peer Interaction, Extracurricular)
            satisfaction_fields = [
                'teaching_quality_rating',
                'learning_environment_rating',
                'peer_interaction_satisfaction',
                'extracurricular_satisfaction'
            ]
            satisfaction_values = [float(response.get(field, 0)) for field in satisfaction_fields]
            indices['satisfaction_score'] = round(np.mean(satisfaction_values), 2) if satisfaction_values else 0.0

            # Success Index (Academic Progress, Skills, Critical Thinking, Problem Solving)
            success_fields = [
                'academic_progress_rating',
                'skill_development_rating',
                'critical_thinking_improvement',
                'problem_solving_confidence'
            ]
            success_values = [float(response.get(field, 0)) for field in success_fields]
            indices['success_index'] = round(np.mean(success_values), 2) if success_values else 0.0

            # Safety Index (Physical, Psychological, Bullying Prevention, Emergency)
            safety_fields = [
                'physical_safety_rating',
                'psychological_safety_rating',
                'bullying_prevention_effectiveness',
                'emergency_preparedness_rating'
            ]
            safety_values = [float(response.get(field, 0)) for field in safety_fields]
            indices['safety_index'] = round(np.mean(safety_values), 2) if safety_values else 0.0

            # Wellbeing Index (Mental Health, Stress Management, Physical Health, Overall Wellbeing)
            wellbeing_fields = [
                'mental_health_support_rating',
                'stress_management_support',
                'physical_health_support',
                'overall_wellbeing_rating'
            ]
            wellbeing_values = [float(response.get(field, 0)) for field in wellbeing_fields]
            indices['wellbeing_index'] = round(np.mean(wellbeing_values), 2) if wellbeing_values else 0.0

            # Overall Satisfaction
            indices['overall_satisfaction'] = float(response.get('overall_satisfaction', 0))

            return indices

        except Exception as e:
            logger.error(f"Error calculating composite indices: {e}")
            return {
                'learner_needs_index': 0.0,
                'satisfaction_score': 0.0,
                'success_index': 0.0,
                'safety_index': 0.0,
                'wellbeing_index': 0.0,
                'overall_satisfaction': 0.0
            }

    def validate_survey_response(self, response):
        """Validate survey response data structure"""
        required_fields = [
            'track', 'grade_level', 'academic_year', 'semester',
            'overall_satisfaction'
        ]

        missing_fields = []
        for field in required_fields:
            if field not in response or response[field] is None:
                missing_fields.append(field)

        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

        # Validate rating ranges (1-5)
        rating_fields = [
            'curriculum_relevance_rating', 'learning_pace_appropriateness',
            'individual_support_availability', 'learning_style_accommodation',
            'teaching_quality_rating', 'learning_environment_rating',
            'peer_interaction_satisfaction', 'extracurricular_satisfaction',
            'academic_progress_rating', 'skill_development_rating',
            'critical_thinking_improvement', 'problem_solving_confidence',
            'physical_safety_rating', 'psychological_safety_rating',
            'bullying_prevention_effectiveness', 'emergency_preparedness_rating',
            'mental_health_support_rating', 'stress_management_support',
            'physical_health_support', 'overall_wellbeing_rating',
            'overall_satisfaction'
        ]

        invalid_ratings = []
        for field in rating_fields:
            if field in response:
                value = response[field]
                if not isinstance(value, (int, float)) or not (1 <= value <= 5):
                    invalid_ratings.append(f"{field}: {value}")

        if invalid_ratings:
            return False, f"Invalid rating values: {', '.join(invalid_ratings)}"

        return True, "Valid survey response"

    def extract_text_feedback(self, response):
        """Extract text feedback from survey response"""
        text_fields = ['positive_aspects', 'improvement_suggestions', 'additional_comments']
        feedback_texts = []

        for field in text_fields:
            if field in response and response[field]:
                text = str(response[field]).strip()
                if text and len(text) > 2:  # Filter out very short responses
                    feedback_texts.append(text)

        return feedback_texts

    def normalize_data(self, data, method='standard'):
        """Normalize data using specified method"""
        try:
            if isinstance(data, list):
                data = np.array(data)

            if method == 'standard':
                # Standard normalization (z-score)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                std = np.where(std == 0, 1, std)  # Avoid division by zero
                normalized = (data - mean) / std

            elif method == 'minmax':
                # Min-max normalization (0-1)
                min_vals = np.min(data, axis=0)
                max_vals = np.max(data, axis=0)
                range_vals = max_vals - min_vals
                range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
                normalized = (data - min_vals) / range_vals

            else:
                raise ValueError(f"Unsupported normalization method: {method}")

            return normalized.tolist() if isinstance(data, list) else normalized

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return data

    def preprocess_performance_data(self, data):
        """Preprocess data for student performance prediction"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

            # Select features for performance prediction
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'learning_style_accommodation',
                'teaching_quality_rating', 'learning_environment_rating',
                'peer_interaction_satisfaction', 'academic_progress_rating',
                'skill_development_rating', 'critical_thinking_improvement',
                'problem_solving_confidence', 'attendance_rate',
                'participation_score', 'overall_satisfaction'
            ]

            # Filter to available columns
            available_columns = [col for col in feature_columns if col in df.columns]
            if not available_columns:
                raise ValueError("No suitable features found for performance prediction")

            # Extract features and target
            X = df[available_columns].fillna(0)
            y = df.get('grade_average', pd.Series([2.5] * len(df)))  # Default GPA

            return X.values.tolist(), y.tolist()

        except Exception as e:
            logger.error(f"Error preprocessing performance data: {e}")
            raise ValueError(f"Invalid performance data format: {e}")

    def preprocess_dropout_data(self, data):
        """Preprocess data for dropout risk prediction"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

            # Select features for dropout prediction
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'learning_style_accommodation',
                'teaching_quality_rating', 'learning_environment_rating',
                'peer_interaction_satisfaction', 'academic_progress_rating',
                'skill_development_rating', 'critical_thinking_improvement',
                'problem_solving_confidence', 'physical_safety_rating',
                'psychological_safety_rating', 'mental_health_support_rating',
                'stress_management_support', 'attendance_rate',
                'participation_score', 'overall_satisfaction',
                'bullying_prevention_effectiveness', 'emergency_preparedness_rating'
            ]

            # Filter to available columns
            available_columns = [col for col in feature_columns if col in df.columns]
            if not available_columns:
                raise ValueError("No suitable features found for dropout prediction")

            # Extract features and target
            X = df[available_columns].fillna(0)
            y = df.get('dropout_risk', pd.Series([0] * len(df)))  # Default no risk

            return X.values.tolist(), y.tolist()

        except Exception as e:
            logger.error(f"Error preprocessing dropout data: {e}")
            raise ValueError(f"Invalid dropout data format: {e}")

    def preprocess_risk_assessment_data(self, data):
        """Preprocess data for comprehensive risk assessment"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

            # Select comprehensive features for risk assessment
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'learning_style_accommodation',
                'teaching_quality_rating', 'learning_environment_rating',
                'peer_interaction_satisfaction', 'extracurricular_satisfaction',
                'academic_progress_rating', 'skill_development_rating',
                'critical_thinking_improvement', 'problem_solving_confidence',
                'physical_safety_rating', 'psychological_safety_rating',
                'bullying_prevention_effectiveness', 'emergency_preparedness_rating',
                'mental_health_support_rating', 'stress_management_support',
                'physical_health_support', 'overall_wellbeing_rating',
                'attendance_rate', 'participation_score', 'overall_satisfaction',
                'grade_average'
            ]

            # Filter to available columns
            available_columns = [col for col in feature_columns if col in df.columns]
            if not available_columns:
                raise ValueError("No suitable features found for risk assessment")

            # Extract features and target
            X = df[available_columns].fillna(0)
            y = df.get('risk_score', pd.Series([50.0] * len(df)))  # Default moderate risk

            return X.values.tolist(), y.tolist()

        except Exception as e:
            logger.error(f"Error preprocessing risk assessment data: {e}")
            raise ValueError(f"Invalid risk assessment data format: {e}")

    def preprocess_trend_data(self, data):
        """Preprocess time series data for satisfaction trend prediction"""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

            # Handle time-based features if available
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')  # Sort by time
                df['month'] = df['timestamp'].dt.month
                df['quarter'] = df['timestamp'].dt.quarter
                df['day_of_year'] = df['timestamp'].dt.dayofyear
                df['week_of_year'] = df['timestamp'].dt.isocalendar().week

            # Select features for trend prediction
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'learning_style_accommodation',
                'teaching_quality_rating', 'learning_environment_rating',
                'peer_interaction_satisfaction', 'extracurricular_satisfaction',
                'academic_progress_rating', 'skill_development_rating',
                'critical_thinking_improvement', 'problem_solving_confidence',
                'physical_safety_rating', 'psychological_safety_rating',
                'bullying_prevention_effectiveness', 'emergency_preparedness_rating',
                'mental_health_support_rating', 'stress_management_support',
                'physical_health_support', 'overall_wellbeing_rating',
                'attendance_rate', 'participation_score'
            ]

            # Add time-based features if available
            time_features = ['month', 'quarter', 'day_of_year', 'week_of_year']
            feature_columns.extend([col for col in time_features if col in df.columns])

            # Filter to available columns
            available_columns = [col for col in feature_columns if col in df.columns]
            if not available_columns:
                raise ValueError("No suitable features found for trend prediction")

            # Extract features and target
            X = df[available_columns].fillna(0)
            y = df.get('overall_satisfaction', pd.Series([3.0] * len(df)))

            return X.values.tolist(), y.tolist(), df

        except Exception as e:
            logger.error(f"Error preprocessing trend data: {e}")
            raise ValueError(f"Invalid trend data format: {e}")

    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """Detect outliers in data"""
        try:
            if isinstance(data, list):
                data = np.array(data)

            if method == 'iqr':
                # Interquartile range method
                q1 = np.percentile(data, 25, axis=0)
                q3 = np.percentile(data, 75, axis=0)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)

            elif method == 'zscore':
                # Z-score method
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                z_scores = np.abs((data - mean) / std)
                outliers = (z_scores > threshold).any(axis=1)

            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")

            return outliers.tolist() if isinstance(data, list) else outliers

        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return [False] * len(data) if isinstance(data, list) else np.zeros(len(data), dtype=bool)
