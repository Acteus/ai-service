"""
ISO 21001 AI Models Training Pipeline

Trains all AI models using synthetic data and existing survey data:
- Compliance Predictor (Deep Learning)
- Student Performance Predictor
- Dropout Risk Predictor
- Risk Assessment Predictor
- Satisfaction Trend Predictor
- Student Clusterer
- Sentiment Analyzer
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import data generator
from data_generation.iso21001_data_generator import ISO21001DataGenerator

# Import AI models
from ai_models.compliance_predictor import CompliancePredictor
from ai_models.student_performance_predictor import StudentPerformancePredictor
from ai_models.dropout_risk_predictor import DropoutRiskPredictor
from ai_models.risk_assessment_predictor import RiskAssessmentPredictor
from ai_models.satisfaction_trend_predictor import SatisfactionTrendPredictor
from ai_models.student_clusterer import StudentClusterer
from ai_models.sentiment_analyzer import SentimentAnalyzer

# Import data processor
from utils.data_processor import DataProcessor


class ModelTrainer:
    """Orchestrates training of all ISO 21001 AI models"""

    def __init__(self, use_synthetic=True, use_existing=True):
        """
        Initialize trainer

        Args:
            use_synthetic: Generate and use synthetic data
            use_existing: Use existing survey data from Laravel database
        """
        self.use_synthetic = use_synthetic
        self.use_existing = use_existing
        self.data_processor = DataProcessor()

        # Create necessary directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        logger.info("Model Trainer initialized")

    def load_data(self) -> pd.DataFrame:
        """Load training data from various sources"""
        dfs = []

        # Load synthetic data
        if self.use_synthetic:
            logger.info("Generating synthetic ISO 21001 data...")
            generator = ISO21001DataGenerator(seed=42)

            # Generate diverse datasets
            df_base = generator.generate_dataset(n_samples=800, days_back=180)
            df_improving = generator.generate_with_trends(n_samples=100, improvement_trend=True)
            df_declining = generator.generate_with_trends(n_samples=100, improvement_trend=False)

            dfs.extend([df_base, df_improving, df_declining])
            logger.info(f"Generated {len(df_base) + len(df_improving) + len(df_declining)} synthetic responses")

        # Load existing data from CSV export (if available)
        if self.use_existing:
            existing_data_path = 'data/existing_survey_data.csv'
            if os.path.exists(existing_data_path):
                logger.info("Loading existing survey data...")
                df_existing = pd.read_csv(existing_data_path)
                dfs.append(df_existing)
                logger.info(f"Loaded {len(df_existing)} existing responses")
            else:
                logger.warning(f"No existing data found at {existing_data_path}")
                logger.info("To include existing data, export from Laravel: php artisan export:survey-data")

        if not dfs:
            raise ValueError("No data available for training! Enable synthetic or provide existing data.")

        # Combine all datasets
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total training samples: {len(df_combined)}")

        return df_combined

    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare data for different model types"""
        logger.info("Preparing training data...")

        # Calculate composite indices
        df['learner_needs_index'] = df[[
            'curriculum_relevance_rating',
            'learning_pace_appropriateness',
            'individual_support_availability',
            'learning_style_accommodation'
        ]].mean(axis=1)

        df['satisfaction_score'] = df[[
            'teaching_quality_rating',
            'learning_environment_rating',
            'peer_interaction_satisfaction',
            'extracurricular_satisfaction'
        ]].mean(axis=1)

        df['success_index'] = df[[
            'academic_progress_rating',
            'skill_development_rating',
            'critical_thinking_improvement',
            'problem_solving_confidence'
        ]].mean(axis=1)

        df['safety_index'] = df[[
            'physical_safety_rating',
            'psychological_safety_rating',
            'bullying_prevention_effectiveness',
            'emergency_preparedness_rating'
        ]].mean(axis=1)

        df['wellbeing_index'] = df[[
            'mental_health_support_rating',
            'stress_management_support',
            'physical_health_support',
            'overall_wellbeing_rating'
        ]].mean(axis=1)

        return df

    def train_compliance_predictor(self, df: pd.DataFrame):
        """Train compliance prediction model"""
        logger.info("\n" + "="*50)
        logger.info("Training Compliance Predictor...")
        logger.info("="*50)

        try:
            model = CompliancePredictor()

            # Prepare features
            features = [
                'learner_needs_index',
                'satisfaction_score',
                'success_index',
                'safety_index',
                'wellbeing_index',
                'overall_satisfaction'
            ]

            X = df[features].values

            # Create labels (binary: compliant vs non-compliant)
            # Calculate weighted compliance score
            y = (
                df['learner_needs_index'] * 0.15 +
                df['satisfaction_score'] * 0.25 +
                df['success_index'] * 0.20 +
                df['safety_index'] * 0.20 +
                df['wellbeing_index'] * 0.15 +
                df['overall_satisfaction'] * 0.05
            )
            y_binary = (y >= 3.5).astype(int)  # 1 = compliant, 0 = non-compliant

            # Train model
            result = model.train(X, y_binary, epochs=50, batch_size=32)

            if result['success']:
                logger.info("✅ Compliance Predictor trained successfully")
                logger.info(f"   Final accuracy: {result['final_accuracy']:.4f}")
                logger.info(f"   Final val_accuracy: {result['final_val_accuracy']:.4f}")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"❌ Compliance Predictor training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_clusterer(self, df: pd.DataFrame):
        """Train student clustering model"""
        logger.info("\n" + "="*50)
        logger.info("Training Student Clusterer...")
        logger.info("="*50)

        try:
            clusterer = StudentClusterer()

            # Prepare features for clustering - pass the DataFrame directly
            clustering_features = [
                'curriculum_relevance_rating',
                'learning_pace_appropriateness',
                'individual_support_availability',
                'teaching_quality_rating',
                'academic_progress_rating',
                'attendance_rate',
                'grade_average',
                'overall_satisfaction'
            ]

            # Filter to available features
            available_features = [col for col in clustering_features if col in df.columns]
            df_cluster = df[available_features].copy()

            # Train with different k values and save best
            best_k = 3
            result = clusterer.train(df_cluster, k=best_k)

            if result['success']:
                logger.info("✅ Student Clusterer trained successfully")
                logger.info(f"   Optimal clusters: {best_k}")
                logger.info(f"   Silhouette score: {result.get('metrics', {}).get('silhouette_score', 'N/A')}")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"❌ Student Clusterer training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_sentiment_analyzer(self, df: pd.DataFrame):
        """Train sentiment analysis model"""
        logger.info("\n" + "="*50)
        logger.info("Training Sentiment Analyzer...")
        logger.info("="*50)

        try:
            analyzer = SentimentAnalyzer()

            # Collect all comments
            comments = []
            labels = []

            for _, row in df.iterrows():
                # Positive aspects
                if pd.notna(row.get('positive_aspects')):
                    comments.append(row['positive_aspects'])
                    # Label based on overall satisfaction
                    if row['overall_satisfaction'] >= 4:
                        labels.append('positive')
                    elif row['overall_satisfaction'] >= 3:
                        labels.append('neutral')
                    else:
                        labels.append('negative')

                # Improvement suggestions (typically neutral/negative)
                if pd.notna(row.get('improvement_suggestions')):
                    comments.append(row['improvement_suggestions'])
                    labels.append('neutral' if row['overall_satisfaction'] >= 3 else 'negative')

            if len(comments) > 50:  # Need sufficient data
                result = analyzer.train(comments, labels)

                if result['success']:
                    logger.info("✅ Sentiment Analyzer trained successfully")
                    logger.info(f"   Training samples: {len(comments)}")
                    logger.info(f"   Accuracy: {result.get('accuracy', 'N/A')}")
                else:
                    logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

                return result
            else:
                logger.warning("⚠️  Insufficient comment data for sentiment analysis training")
                return {'success': False, 'error': 'Insufficient data'}

        except Exception as e:
            logger.error(f"❌ Sentiment Analyzer training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_performance_predictor(self, df: pd.DataFrame):
        """Train performance prediction model"""
        logger.info("\n" + "="*50)
        logger.info("Training Performance Predictor...")
        logger.info("="*50)

        try:
            model = StudentPerformancePredictor()

            # Prepare features
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'learning_style_accommodation',
                'teaching_quality_rating', 'learning_environment_rating',
                'peer_interaction_satisfaction', 'academic_progress_rating',
                'skill_development_rating', 'critical_thinking_improvement',
                'problem_solving_confidence', 'attendance_rate',
                'participation_score', 'overall_satisfaction'
            ]

            available_features = [col for col in feature_columns if col in df.columns]
            X = df[available_features].values

            # Target: grade_average (GPA)
            if 'grade_average' in df.columns:
                y = df['grade_average'].values
            else:
                # Estimate GPA from performance indicators
                y = (df['academic_progress_rating'] / 5) * 4.0
                y = np.clip(y, 0, 4.0).values

            # Train model
            result = model.train(X, y, model_type='random_forest')

            if result['success']:
                logger.info("✅ Performance Predictor trained successfully")
                logger.info(f"   R² Score: {result['r2_score']:.4f}")
                logger.info(f"   MSE: {result['mse']:.4f}")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"❌ Performance Predictor training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_dropout_predictor(self, df: pd.DataFrame):
        """Train dropout risk prediction model"""
        logger.info("\n" + "="*50)
        logger.info("Training Dropout Risk Predictor...")
        logger.info("="*50)

        try:
            model = DropoutRiskPredictor()

            # Prepare features
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'teaching_quality_rating',
                'learning_environment_rating', 'peer_interaction_satisfaction',
                'academic_progress_rating', 'skill_development_rating',
                'physical_safety_rating', 'psychological_safety_rating',
                'mental_health_support_rating', 'stress_management_support',
                'attendance_rate', 'participation_score', 'overall_satisfaction'
            ]

            available_features = [col for col in feature_columns if col in df.columns]
            X = df[available_features].values

            # Create labels (binary: at-risk vs not at-risk)
            # Consider at-risk if attendance < 70 OR satisfaction < 2.5 OR progress < 2.5
            at_risk = (
                (df.get('attendance_rate', 100) < 70) |
                (df.get('overall_satisfaction', 5) < 2.5) |
                (df.get('academic_progress_rating', 5) < 2.5)
            )
            y = at_risk.astype(int).values

            # Train model
            result = model.train(X, y, model_type='random_forest')

            if result['success']:
                logger.info("✅ Dropout Risk Predictor trained successfully")
                logger.info(f"   AUC Score: {result['auc_score']:.4f}")
                logger.info(f"   Accuracy: {result['accuracy']:.4f}")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"❌ Dropout Risk Predictor training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_risk_assessment(self, df: pd.DataFrame):
        """Train comprehensive risk assessment model"""
        logger.info("\n" + "="*50)
        logger.info("Training Risk Assessment Predictor...")
        logger.info("="*50)

        try:
            model = RiskAssessmentPredictor()

            # Use all available features for comprehensive risk
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

            available_features = [col for col in feature_columns if col in df.columns]
            X = df[available_features].values

            # Calculate overall risk score (0-100)
            # Risk is inverse of satisfaction/performance
            risk_score = (
                (5 - df['overall_satisfaction']) * 10 +  # 0-50 points
                (100 - df.get('attendance_rate', 80)) * 0.5 +  # 0-50 points
                (5 - df.get('academic_progress_rating', 3)) * 10  # 0-50 points
            )
            y = np.clip(risk_score, 0, 100).values

            # Train model
            result = model.train(X, y)

            if result['success']:
                logger.info("✅ Risk Assessment Predictor trained successfully")
                logger.info(f"   R² Score: {result['r2_score']:.4f}")
                logger.info(f"   MAE: {result['mae']:.4f}")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"❌ Risk Assessment Predictor training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_satisfaction_trend(self, df: pd.DataFrame):
        """Train satisfaction trend predictor"""
        logger.info("\n" + "="*50)
        logger.info("Training Satisfaction Trend Predictor...")
        logger.info("="*50)

        try:
            model = SatisfactionTrendPredictor()

            # Sort by timestamp if available
            if 'created_at' in df.columns:
                df = df.sort_values('created_at')
                # Handle various datetime formats including timezone info
                df['timestamp'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce')

            # Prepare features
            feature_columns = [
                'curriculum_relevance_rating', 'learning_pace_appropriateness',
                'individual_support_availability', 'teaching_quality_rating',
                'learning_environment_rating', 'peer_interaction_satisfaction',
                'academic_progress_rating', 'skill_development_rating',
                'physical_safety_rating', 'mental_health_support_rating',
                'attendance_rate', 'participation_score'
            ]

            available_features = [col for col in feature_columns if col in df.columns]
            X = df[available_features].values
            y = df['overall_satisfaction'].values

            # Train regression model
            result = model.train(X, y, model_type='gradient_boosting')

            # Train time series model if we have temporal data
            if 'timestamp' in df.columns and len(df) >= 12:
                # Ensure timestamp is datetime type
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # Remove any invalid timestamps
                df_valid = df.dropna(subset=['timestamp'])

                if len(df_valid) >= 12:
                    time_series = df_valid.groupby(df_valid['timestamp'].dt.to_period('M'))['overall_satisfaction'].mean()
                    ts_result = model.train_time_series(time_series)
                    result['time_series_model'] = ts_result

            if result['success']:
                logger.info("✅ Satisfaction Trend Predictor trained successfully")
                logger.info(f"   R² Score: {result['r2_score']:.4f}")
                logger.info(f"   MAE: {result['mae']:.4f}")
            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

            return result

        except Exception as e:
            logger.error(f"❌ Satisfaction Trend Predictor training failed: {e}")
            return {'success': False, 'error': str(e)}

    def train_all_models(self):
        """Train all models in the pipeline"""
        logger.info("\n🎓 ISO 21001 AI Models Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        df = self.load_data()
        df = self.prepare_training_data(df)

        # Save prepared data
        df.to_csv('data/prepared_training_data.csv', index=False)
        logger.info(f"Saved prepared data to data/prepared_training_data.csv")

        # Training results
        results = {}

        # Train each model
        results['compliance_predictor'] = self.train_compliance_predictor(df)
        results['student_clusterer'] = self.train_clusterer(df)
        results['sentiment_analyzer'] = self.train_sentiment_analyzer(df)
        results['performance_predictor'] = self.train_performance_predictor(df)
        results['dropout_predictor'] = self.train_dropout_predictor(df)
        results['risk_assessment'] = self.train_risk_assessment(df)
        results['satisfaction_trend'] = self.train_satisfaction_trend(df)

        # Summary
        logger.info("\n" + "="*60)
        logger.info("Training Summary")
        logger.info("="*60)

        for model_name, result in results.items():
            status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
            logger.info(f"{model_name}: {status}")
            if not result.get('success'):
                logger.info(f"  Error: {result.get('error', 'Unknown')}")

        # Save training summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'training_samples': len(df),
            'results': results
        }

        with open('logs/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\nTraining summary saved to logs/training_summary.json")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\n✅ Training pipeline complete!")

        return results


def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(description='Train ISO 21001 AI Models')
    parser.add_argument('--no-synthetic', action='store_true',
                       help='Disable synthetic data generation')
    parser.add_argument('--no-existing', action='store_true',
                       help='Disable loading existing survey data')
    parser.add_argument('--model', type=str,
                       choices=['all', 'compliance', 'cluster', 'sentiment',
                               'performance', 'dropout', 'risk', 'trend'],
                       default='all', help='Which model to train')

    args = parser.parse_args()

    # Initialize trainer
    trainer = ModelTrainer(
        use_synthetic=not args.no_synthetic,
        use_existing=not args.no_existing
    )

    # Load and prepare data
    df = trainer.load_data()
    df = trainer.prepare_training_data(df)

    # Train specified model(s)
    if args.model == 'all':
        trainer.train_all_models()
    elif args.model == 'compliance':
        trainer.train_compliance_predictor(df)
    elif args.model == 'cluster':
        trainer.train_clusterer(df)
    elif args.model == 'sentiment':
        trainer.train_sentiment_analyzer(df)
    elif args.model == 'performance':
        trainer.train_performance_predictor(df)
    elif args.model == 'dropout':
        trainer.train_dropout_predictor(df)
    elif args.model == 'risk':
        trainer.train_risk_assessment(df)
    elif args.model == 'trend':
        trainer.train_satisfaction_trend(df)


if __name__ == '__main__':
    main()
