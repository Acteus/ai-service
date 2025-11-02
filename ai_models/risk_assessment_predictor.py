"""
Advanced Risk Assessment Predictor using Machine Learning
Implements comprehensive risk scoring for ISO 21001 compliance across multiple dimensions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class RiskAssessmentPredictor:
    def __init__(self, model_path='models/risk_assessment_model.pkl', scaler_path='models/risk_assessment_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = 'models/risk_assessment_encoder.pkl'
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
                logger.info("Loaded existing risk assessment model")

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
        """Preprocess data for comprehensive risk assessment"""
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
        categorical_cols = ['track', 'grade_level', 'semester', 'academic_year']
        for col in categorical_cols:
            if col in df.columns:
                if self.encoder is None:
                    self.encoder = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = self.encoder.fit_transform(df[col])

        # Select comprehensive features for risk assessment
        feature_columns = [
            # Learning Environment
            'curriculum_relevance_rating', 'learning_pace_appropriateness',
            'individual_support_availability', 'learning_style_accommodation',
            'teaching_quality_rating', 'learning_environment_rating',

            # Social Factors
            'peer_interaction_satisfaction', 'extracurricular_satisfaction',

            # Academic Performance
            'academic_progress_rating', 'skill_development_rating',
            'critical_thinking_improvement', 'problem_solving_confidence',

            # Safety and Wellbeing
            'physical_safety_rating', 'psychological_safety_rating',
            'bullying_prevention_effectiveness', 'emergency_preparedness_rating',
            'mental_health_support_rating', 'stress_management_support',
            'physical_health_support', 'overall_wellbeing_rating',

            # Engagement Metrics
            'attendance_rate', 'participation_score', 'overall_satisfaction',

            # Demographic/Administrative
            'grade_average'
        ]

        # Add encoded categorical features
        feature_columns.extend([col for col in categorical_cols if col in df.columns])

        # Filter to available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        if not available_columns:
            raise ValueError("No suitable features found for risk assessment")

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
                elif 'average' in col.lower():
                    X[col] = 2.5
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

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the comprehensive risk assessment model"""
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

            # Use Random Forest for comprehensive risk assessment
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_val_scaled)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

            # Save model and scaler
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            if self.encoder:
                joblib.dump(self.encoder, self.encoder_path)

            self.is_trained = True

            logger.info(f"Risk assessment model trained successfully with R² = {r2:.3f}")

            return {
                'success': True,
                'mae': round(mae, 4),
                'r2_score': round(r2, 4),
                'feature_importance': self._get_feature_importance()
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def predict(self, data):
        """Predict comprehensive risk score"""
        try:
            if not self.is_trained:
                # Fallback to rule-based assessment
                return self._rule_based_assessment(data)

            # Preprocess input data
            X_scaled, _ = self.preprocess_data(data)

            # Make prediction
            risk_score = self.model.predict(X_scaled)[0]

            # Ensure prediction is within valid range (0-100 scale)
            risk_score = np.clip(risk_score, 0, 100)

            # Determine risk level and category
            risk_level, risk_category, compliance_impact = self._categorize_risk(risk_score)

            # Calculate confidence
            confidence = self._calculate_assessment_confidence(data, risk_score)

            # Get detailed risk breakdown
            risk_breakdown = self._calculate_risk_breakdown(data)

            return {
                'overall_risk_score': round(float(risk_score), 2),
                'risk_level': risk_level,
                'risk_category': risk_category,
                'compliance_impact': compliance_impact,
                'confidence': round(confidence, 2),
                'model_used': 'Random Forest Regressor',
                'iso_21001_insights': {
                    'primary_concerns': self._identify_primary_concerns(risk_breakdown),
                    'recommended_actions': self._generate_action_plan(risk_score, risk_breakdown),
                    'monitoring_priority': self._get_monitoring_priority(risk_score),
                    'compliance_gap': self._assess_compliance_gap(risk_score)
                },
                'risk_breakdown': risk_breakdown,
                'action_plan': self._generate_detailed_action_plan(risk_score, risk_breakdown)
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._rule_based_assessment(data)

    def _rule_based_assessment(self, data):
        """Fallback rule-based comprehensive risk assessment"""
        risk_components = {
            'academic_risk': 0,
            'engagement_risk': 0,
            'safety_risk': 0,
            'wellbeing_risk': 0,
            'satisfaction_risk': 0
        }

        if isinstance(data, dict):
            # Academic Risk (30% weight)
            academic_factors = [
                data.get('academic_progress_rating', 5),
                data.get('skill_development_rating', 5),
                data.get('critical_thinking_improvement', 5),
                data.get('grade_average', 2.5) * 2  # Convert GPA to 1-5 scale
            ]
            academic_score = np.mean(academic_factors)
            risk_components['academic_risk'] = max(0, (5 - academic_score) / 5) * 100

            # Engagement Risk (25% weight)
            engagement_factors = [
                data.get('attendance_rate', 100) / 20,  # Convert to 1-5 scale
                data.get('participation_score', 5),
                data.get('peer_interaction_satisfaction', 5),
                data.get('extracurricular_satisfaction', 5)
            ]
            engagement_score = np.mean(engagement_factors)
            risk_components['engagement_risk'] = max(0, (5 - engagement_score) / 5) * 100

            # Safety Risk (20% weight)
            safety_factors = [
                data.get('physical_safety_rating', 5),
                data.get('psychological_safety_rating', 5),
                data.get('bullying_prevention_effectiveness', 5),
                data.get('emergency_preparedness_rating', 5)
            ]
            safety_score = np.mean(safety_factors)
            risk_components['safety_risk'] = max(0, (5 - safety_score) / 5) * 100

            # Wellbeing Risk (15% weight)
            wellbeing_factors = [
                data.get('mental_health_support_rating', 5),
                data.get('stress_management_support', 5),
                data.get('physical_health_support', 5),
                data.get('overall_wellbeing_rating', 5)
            ]
            wellbeing_score = np.mean(wellbeing_factors)
            risk_components['wellbeing_risk'] = max(0, (5 - wellbeing_score) / 5) * 100

            # Satisfaction Risk (10% weight)
            satisfaction_factors = [
                data.get('overall_satisfaction', 5),
                data.get('teaching_quality_rating', 5),
                data.get('learning_environment_rating', 5)
            ]
            satisfaction_score = np.mean(satisfaction_factors)
            risk_components['satisfaction_risk'] = max(0, (5 - satisfaction_score) / 5) * 100

        # Calculate weighted overall risk score
        weights = {'academic_risk': 0.3, 'engagement_risk': 0.25, 'safety_risk': 0.2,
                  'wellbeing_risk': 0.15, 'satisfaction_risk': 0.1}

        overall_risk = sum(risk_components[comp] * weights[comp] for comp in risk_components)

        risk_level, risk_category, compliance_impact = self._categorize_risk(overall_risk)

        return {
            'overall_risk_score': round(float(overall_risk), 2),
            'risk_level': risk_level,
            'risk_category': risk_category,
            'compliance_impact': compliance_impact,
            'confidence': 0.65,
            'model_used': 'Rule-based (Fallback)',
            'iso_21001_insights': {
                'primary_concerns': self._identify_primary_concerns(risk_components),
                'recommended_actions': self._generate_action_plan(overall_risk, risk_components),
                'monitoring_priority': self._get_monitoring_priority(overall_risk),
                'compliance_gap': self._assess_compliance_gap(overall_risk)
            },
            'risk_breakdown': risk_components,
            'action_plan': self._generate_detailed_action_plan(overall_risk, risk_components)
        }

    def _categorize_risk(self, risk_score):
        """Categorize risk level based on score"""
        if risk_score >= 70:
            return 'High', 'High Risk', 'Severe'
        elif risk_score >= 30:
            return 'Medium', 'Medium Risk', 'Moderate'
        else:
            return 'Low', 'Low Risk', 'Minor'

    def _calculate_assessment_confidence(self, data, risk_score):
        """Calculate confidence in the risk assessment"""
        base_confidence = 0.75

        # Adjust based on data completeness
        if isinstance(data, dict):
            essential_fields = [
                'overall_satisfaction', 'attendance_rate', 'academic_progress_rating',
                'physical_safety_rating', 'mental_health_support_rating'
            ]
            available_fields = sum(1 for field in essential_fields if field in data and data[field] is not None)
            completeness_factor = available_fields / len(essential_fields)
            base_confidence += completeness_factor * 0.15

        # Adjust based on risk score extremes (more confident in clear cases)
        if risk_score < 10 or risk_score > 90:
            base_confidence += 0.1

        return min(base_confidence, 0.95)

    def _calculate_risk_breakdown(self, data):
        """Calculate detailed risk breakdown by category"""
        breakdown = {}

        if isinstance(data, dict):
            # Learning Environment Risk
            learning_factors = [
                data.get('curriculum_relevance_rating', 3),
                data.get('learning_pace_appropriateness', 3),
                data.get('teaching_quality_rating', 3),
                data.get('learning_environment_rating', 3)
            ]
            breakdown['learning_environment'] = round(max(0, (5 - np.mean(learning_factors)) / 5) * 100, 2)

            # Academic Performance Risk
            academic_factors = [
                data.get('academic_progress_rating', 3),
                data.get('skill_development_rating', 3),
                data.get('critical_thinking_improvement', 3),
                data.get('grade_average', 2.5) * 2
            ]
            breakdown['academic_performance'] = round(max(0, (5 - np.mean(academic_factors)) / 5) * 100, 2)

            # Safety Risk
            safety_factors = [
                data.get('physical_safety_rating', 3),
                data.get('psychological_safety_rating', 3),
                data.get('bullying_prevention_effectiveness', 3),
                data.get('emergency_preparedness_rating', 3)
            ]
            breakdown['safety'] = round(max(0, (5 - np.mean(safety_factors)) / 5) * 100, 2)

            # Wellbeing Risk
            wellbeing_factors = [
                data.get('mental_health_support_rating', 3),
                data.get('stress_management_support', 3),
                data.get('physical_health_support', 3),
                data.get('overall_wellbeing_rating', 3)
            ]
            breakdown['wellbeing'] = round(max(0, (5 - np.mean(wellbeing_factors)) / 5) * 100, 2)

            # Engagement Risk
            engagement_factors = [
                data.get('attendance_rate', 75) / 15,
                data.get('participation_score', 3),
                data.get('peer_interaction_satisfaction', 3),
                data.get('overall_satisfaction', 3)
            ]
            breakdown['engagement'] = round(max(0, (5 - np.mean(engagement_factors)) / 5) * 100, 2)

        return breakdown

    def _identify_primary_concerns(self, risk_breakdown):
        """Identify primary areas of concern"""
        concerns = []
        threshold = 60  # Risk score threshold for concern

        for category, score in risk_breakdown.items():
            if score >= threshold:
                category_names = {
                    'learning_environment': 'Learning Environment',
                    'academic_performance': 'Academic Performance',
                    'safety': 'Safety and Security',
                    'wellbeing': 'Student Wellbeing',
                    'engagement': 'Student Engagement'
                }
                concerns.append({
                    'category': category_names.get(category, category),
                    'risk_score': score,
                    'severity': 'Critical' if score >= 80 else 'High' if score >= 70 else 'Moderate'
                })

        return sorted(concerns, key=lambda x: x['risk_score'], reverse=True)

    def _generate_action_plan(self, overall_risk, risk_breakdown):
        """Generate high-level action plan"""
        actions = []

        if overall_risk >= 75:
            actions.extend([
                "URGENT: Implement comprehensive intervention program immediately",
                "Conduct full ISO 21001 compliance audit within 30 days",
                "Establish emergency response team for at-risk students"
            ])
        elif overall_risk >= 60:
            actions.extend([
                "HIGH PRIORITY: Develop targeted improvement plans for high-risk areas",
                "Increase monitoring frequency and student support services",
                "Schedule stakeholder meetings to address critical concerns"
            ])
        elif overall_risk >= 40:
            actions.extend([
                "MODERATE PRIORITY: Implement preventive measures and early interventions",
                "Review and update policies in identified risk areas",
                "Enhance staff training and resource allocation"
            ])
        else:
            actions.extend([
                "LOW PRIORITY: Maintain current standards and monitor trends",
                "Conduct regular assessments and implement continuous improvements",
                "Focus on best practice sharing and staff development"
            ])

        return actions

    def _get_monitoring_priority(self, risk_score):
        """Determine monitoring priority level"""
        if risk_score >= 75:
            return 'Immediate Daily Monitoring'
        elif risk_score >= 60:
            return 'Weekly Intensive Monitoring'
        elif risk_score >= 40:
            return 'Bi-weekly Focused Monitoring'
        elif risk_score >= 25:
            return 'Monthly Regular Monitoring'
        else:
            return 'Quarterly Routine Monitoring'

    def _assess_compliance_gap(self, risk_score):
        """Assess compliance gap with ISO 21001 standards"""
        if risk_score >= 75:
            return 'Major compliance gaps requiring immediate corrective action'
        elif risk_score >= 60:
            return 'Significant compliance gaps needing urgent attention'
        elif risk_score >= 40:
            return 'Moderate compliance gaps requiring improvement plans'
        elif risk_score >= 25:
            return 'Minor compliance gaps with room for enhancement'
        else:
            return 'Good compliance with opportunities for excellence'

    def _generate_detailed_action_plan(self, overall_risk, risk_breakdown):
        """Generate detailed action plan by risk category"""
        action_plan = {}

        # Define actions for each risk category
        category_actions = {
            'learning_environment': {
                'high': [
                    'Review and update curriculum to improve relevance',
                    'Adjust learning pace and provide differentiated instruction',
                    'Enhance teaching quality through professional development',
                    'Improve physical learning environment and resources'
                ],
                'medium': [
                    'Conduct curriculum relevance assessment with students',
                    'Provide additional support for diverse learning needs',
                    'Implement peer mentoring programs'
                ]
            },
            'academic_performance': {
                'high': [
                    'Implement intensive academic support programs',
                    'Provide individualized tutoring and remediation',
                    'Review assessment methods and provide timely feedback',
                    'Develop skill-building workshops and interventions'
                ],
                'medium': [
                    'Establish academic early warning systems',
                    'Create study skills and time management workshops',
                    'Strengthen academic advising and counseling services'
                ]
            },
            'safety': {
                'high': [
                    'Conduct comprehensive safety audit and risk assessment',
                    'Strengthen bullying prevention and intervention programs',
                    'Improve emergency preparedness and response procedures',
                    'Enhance security measures and monitoring systems'
                ],
                'medium': [
                    'Review and update safety policies and procedures',
                    'Provide safety education and awareness training',
                    'Strengthen reporting mechanisms for safety concerns'
                ]
            },
            'wellbeing': {
                'high': [
                    'Expand mental health support services and counseling',
                    'Implement comprehensive stress management programs',
                    'Develop peer support and wellness initiatives',
                    'Strengthen physical health promotion and support'
                ],
                'medium': [
                    'Increase access to counseling and mental health resources',
                    'Create wellness workshops and stress reduction programs',
                    'Foster positive school climate and relationships'
                ]
            },
            'engagement': {
                'high': [
                    'Implement attendance improvement strategies',
                    'Create engaging extracurricular and co-curricular programs',
                    'Strengthen student voice and participation mechanisms',
                    'Develop personalized engagement plans for at-risk students'
                ],
                'medium': [
                    'Enhance student participation opportunities',
                    'Improve communication and feedback mechanisms',
                    'Create mentorship and buddy programs'
                ]
            }
        }

        # Generate actions based on risk levels
        for category, score in risk_breakdown.items():
            if score >= 70:
                action_plan[category] = category_actions.get(category, {}).get('high', [])
            elif score >= 40:
                action_plan[category] = category_actions.get(category, {}).get('medium', [])
            else:
                action_plan[category] = ['Monitor and maintain current standards']

        return action_plan

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
                'attendance', 'participation', 'satisfaction', 'grade_average'
            ]

            for i, importance in enumerate(self.model.feature_importances_):
                if i < len(feature_names):
                    importance_dict[feature_names[i]] = round(importance, 4)

            return importance_dict
        return {}
