"""
ISO 21001 Synthetic Data Generator

Generates realistic survey response data based on ISO 21001:2018 standards
for Educational Organizations Management Systems.

This generator creates diverse, correlated data that reflects real-world
patterns in educational quality metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Tuple

class ISO21001DataGenerator:
    """
    Generates synthetic ISO 21001 compliance survey data with realistic patterns
    """

    def __init__(self, seed=42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)

        # ISO 21001 Clause weights (how they influence each other)
        self.clause_correlations = {
            'leadership': 0.3,  # Strong leadership improves everything
            'planning': 0.25,
            'support': 0.2,
            'operation': 0.15,
            'evaluation': 0.1
        }

        # Student personas with different compliance patterns
        self.student_personas = [
            {'name': 'high_achiever', 'base_satisfaction': 4.2, 'variance': 0.3, 'weight': 0.15},
            {'name': 'average_student', 'base_satisfaction': 3.5, 'variance': 0.5, 'weight': 0.50},
            {'name': 'struggling_student', 'base_satisfaction': 2.8, 'variance': 0.6, 'weight': 0.20},
            {'name': 'at_risk', 'base_satisfaction': 2.2, 'variance': 0.7, 'weight': 0.10},
            {'name': 'exceptional', 'base_satisfaction': 4.7, 'variance': 0.2, 'weight': 0.05}
        ]

        # Comment templates for sentiment analysis
        self.positive_comments = [
            "The teaching quality is excellent and very engaging",
            "I feel supported in my learning journey",
            "Great learning environment and helpful instructors",
            "The curriculum is relevant and well-structured",
            "Teachers are very supportive and understanding",
            "Love the hands-on learning approach",
            "Excellent facilities and resources available",
            "The school provides great support services",
            "Very satisfied with the overall learning experience",
            "Good balance between theory and practical work"
        ]

        self.neutral_comments = [
            "The curriculum could be more relevant to real-world applications",
            "Some improvements needed in teaching methods",
            "Learning pace is okay but could be better",
            "Average experience overall",
            "Resources are adequate but not exceptional",
            "Some areas need improvement",
            "It's fine but could be more engaging",
            "Satisfactory learning environment",
            "The program meets basic expectations",
            "Some good points, some areas for improvement"
        ]

        self.negative_comments = [
            "Need more individual support and attention",
            "The learning pace is too fast for me",
            "Not enough practical hands-on experience",
            "Curriculum feels outdated and irrelevant",
            "Limited support for struggling students",
            "Teaching methods need significant improvement",
            "Feel overwhelmed and unsupported",
            "Resources and facilities need upgrading",
            "Not satisfied with the learning environment",
            "Need better mental health support services"
        ]

    def _select_persona(self) -> Dict:
        """Select student persona based on weighted probability"""
        weights = [p['weight'] for p in self.student_personas]
        return random.choices(self.student_personas, weights=weights)[0]

    def _generate_correlated_ratings(self, base_value: float, variance: float, count: int = 4) -> List[float]:
        """Generate correlated ratings around a base value"""
        ratings = []
        for _ in range(count):
            # Add correlation effect
            correlation_factor = np.random.uniform(-variance, variance)
            rating = np.clip(base_value + correlation_factor, 1.0, 5.0)
            ratings.append(round(rating, 1))
        return ratings

    def _calculate_attendance_from_engagement(self, engagement_level: float) -> float:
        """Calculate attendance rate based on engagement (1-5 scale)"""
        # Higher engagement = higher attendance
        base_attendance = 60 + (engagement_level - 1) * 10
        variance = np.random.uniform(-5, 5)
        return round(np.clip(base_attendance + variance, 50, 100), 2)

    def _calculate_gpa_from_performance(self, performance_level: float) -> float:
        """Calculate GPA from performance indicators (1-5 scale to 0-4 GPA)"""
        # Convert 1-5 scale to 0-4 GPA scale
        base_gpa = (performance_level - 1) * 1.0
        variance = np.random.uniform(-0.2, 0.2)
        return round(np.clip(base_gpa + variance, 0.0, 4.0), 2)

    def _generate_comment(self, overall_satisfaction: float) -> Tuple[str, str, str]:
        """Generate realistic comments based on satisfaction level"""
        if overall_satisfaction >= 4.0:
            positive = random.choice(self.positive_comments)
            improvement = random.choice(self.neutral_comments[:5])
            additional = random.choice(self.positive_comments)
        elif overall_satisfaction >= 3.0:
            positive = random.choice(self.neutral_comments)
            improvement = random.choice(self.neutral_comments[5:])
            additional = random.choice(self.neutral_comments)
        else:
            positive = random.choice(self.neutral_comments[-3:])
            improvement = random.choice(self.negative_comments)
            additional = random.choice(self.negative_comments)

        return positive, improvement, additional

    def generate_single_response(self, persona: Dict = None,
                                timestamp: datetime = None) -> Dict:
        """Generate a single realistic survey response"""
        if persona is None:
            persona = self._select_persona()

        if timestamp is None:
            timestamp = datetime.now()

        base_satisfaction = persona['base_satisfaction']
        variance = persona['variance']

        # ISO 21001 Clause 4-6: Context and Leadership
        # These affect all other metrics
        leadership_quality = base_satisfaction + np.random.uniform(-variance, variance)
        leadership_quality = np.clip(leadership_quality, 1.0, 5.0)

        # ISO 21001 Clause 7: Support (affects learner needs and resources)
        support_ratings = self._generate_correlated_ratings(base_satisfaction, variance, 4)

        # ISO 21001 Clause 8: Operation (teaching and learning environment)
        operation_ratings = self._generate_correlated_ratings(base_satisfaction, variance, 4)

        # ISO 21001 Clause 9: Performance evaluation (learner success)
        performance_ratings = self._generate_correlated_ratings(base_satisfaction, variance, 4)

        # ISO 21001 Clause 7.3: Infrastructure and safety
        safety_ratings = self._generate_correlated_ratings(
            base_satisfaction + 0.3,  # Safety typically rated higher
            variance * 0.8,
            4
        )

        # ISO 21001 Clause 7.1.4: Learner wellbeing
        wellbeing_ratings = self._generate_correlated_ratings(base_satisfaction, variance, 4)

        # Calculate indirect metrics
        avg_engagement = np.mean(operation_ratings)
        avg_performance = np.mean(performance_ratings)

        attendance = self._calculate_attendance_from_engagement(avg_engagement)
        gpa = self._calculate_gpa_from_performance(avg_performance)
        participation = int(np.clip(avg_engagement, 1, 5))

        # Generate realistic comments
        overall_sat = base_satisfaction
        pos_comment, imp_comment, add_comment = self._generate_comment(overall_sat)

        # Student demographics
        grade_levels = [11, 12]  # Senior High School
        grade = random.choice(grade_levels)

        semesters = ['1st Semester', '2nd Semester']
        semester = random.choice(semesters)

        genders = ['Male', 'Female', 'Other', 'Prefer not to say']
        gender_weights = [0.45, 0.45, 0.05, 0.05]
        gender = random.choices(genders, weights=gender_weights)[0]

        # Build response dictionary
        response = {
            # Student Information
            'student_id': f'CSS-{random.randint(1000, 9999)}',
            'track': 'CSS',  # Computer System Servicing
            'grade_level': grade,
            'academic_year': f'{timestamp.year}-{timestamp.year + 1}',
            'semester': semester,
            'gender': gender,

            # ISO 21001 Learner Needs Assessment (Clause 7.1)
            'curriculum_relevance_rating': int(round(support_ratings[0])),
            'learning_pace_appropriateness': int(round(support_ratings[1])),
            'individual_support_availability': int(round(support_ratings[2])),
            'learning_style_accommodation': int(round(support_ratings[3])),

            # ISO 21001 Learner Satisfaction Metrics (Clause 8.2)
            'teaching_quality_rating': int(round(operation_ratings[0])),
            'learning_environment_rating': int(round(operation_ratings[1])),
            'peer_interaction_satisfaction': int(round(operation_ratings[2])),
            'extracurricular_satisfaction': int(round(operation_ratings[3])),

            # ISO 21001 Learner Success Indicators (Clause 9.1)
            'academic_progress_rating': int(round(performance_ratings[0])),
            'skill_development_rating': int(round(performance_ratings[1])),
            'critical_thinking_improvement': int(round(performance_ratings[2])),
            'problem_solving_confidence': int(round(performance_ratings[3])),

            # ISO 21001 Learner Safety Assessment (Clause 7.3)
            'physical_safety_rating': int(round(safety_ratings[0])),
            'psychological_safety_rating': int(round(safety_ratings[1])),
            'bullying_prevention_effectiveness': int(round(safety_ratings[2])),
            'emergency_preparedness_rating': int(round(safety_ratings[3])),

            # ISO 21001 Learner Wellbeing Metrics (Clause 7.1.4)
            'mental_health_support_rating': int(round(wellbeing_ratings[0])),
            'stress_management_support': int(round(wellbeing_ratings[1])),
            'physical_health_support': int(round(wellbeing_ratings[2])),
            'overall_wellbeing_rating': int(round(wellbeing_ratings[3])),

            # Overall Satisfaction and Feedback
            'overall_satisfaction': int(round(overall_sat)),
            'positive_aspects': pos_comment,
            'improvement_suggestions': imp_comment,
            'additional_comments': add_comment,

            # Privacy and Consent
            'consent_given': True,
            'ip_address': f'192.168.{random.randint(1, 255)}.{random.randint(1, 255)}',

            # Indirect Metrics (from university systems)
            'attendance_rate': attendance,
            'grade_average': gpa,
            'participation_score': participation,
            'extracurricular_hours': random.randint(0, 20),
            'counseling_sessions': random.randint(0, 5) if base_satisfaction < 3.0 else random.randint(0, 2),

            # Metadata
            'created_at': timestamp.isoformat(),
            'updated_at': timestamp.isoformat(),
        }

        return response

    def generate_dataset(self, n_samples: int = 1000,
                        days_back: int = 180) -> pd.DataFrame:
        """
        Generate a complete dataset with temporal patterns

        Args:
            n_samples: Number of survey responses to generate
            days_back: How many days back to distribute responses

        Returns:
            pandas DataFrame with all responses
        """
        responses = []

        # Generate timestamps with realistic distribution
        # More responses during academic year, fewer during breaks
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for i in range(n_samples):
            # Generate random timestamp
            random_days = random.randint(0, days_back)
            timestamp = start_date + timedelta(days=random_days)

            # Select persona
            persona = self._select_persona()

            # Generate response
            response = self.generate_single_response(persona, timestamp)
            responses.append(response)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} responses...")

        df = pd.DataFrame(responses)
        print(f"\n✅ Generated {len(df)} synthetic ISO 21001 survey responses")

        return df

    def generate_with_trends(self, n_samples: int = 1000,
                            improvement_trend: bool = True) -> pd.DataFrame:
        """
        Generate dataset with temporal improvement/decline trends

        Args:
            n_samples: Number of responses
            improvement_trend: True for improving metrics, False for declining

        Returns:
            DataFrame with trending data
        """
        responses = []
        days_back = 180

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for i in range(n_samples):
            # Calculate progress factor (0 to 1)
            progress = i / n_samples

            # Adjust persona base satisfaction based on trend
            persona = self._select_persona()

            if improvement_trend:
                # Gradually improve satisfaction
                trend_adjustment = progress * 0.5  # Up to +0.5 improvement
            else:
                # Gradually decline satisfaction
                trend_adjustment = -progress * 0.5  # Up to -0.5 decline

            # Modify persona
            modified_persona = persona.copy()
            modified_persona['base_satisfaction'] = np.clip(
                persona['base_satisfaction'] + trend_adjustment,
                1.0, 5.0
            )

            # Generate timestamp
            random_days = int(progress * days_back)
            timestamp = start_date + timedelta(days=random_days)

            # Generate response
            response = self.generate_single_response(modified_persona, timestamp)
            responses.append(response)

            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_samples} responses with {'improvement' if improvement_trend else 'decline'} trend...")

        df = pd.DataFrame(responses)
        print(f"\n✅ Generated {len(df)} responses with temporal trends")

        return df

    def export_to_csv(self, df: pd.DataFrame, filename: str = 'iso21001_training_data.csv'):
        """Export dataset to CSV file"""
        df.to_csv(filename, index=False)
        print(f"✅ Exported to {filename}")

    def export_to_json(self, df: pd.DataFrame, filename: str = 'iso21001_training_data.json'):
        """Export dataset to JSON file"""
        df.to_json(filename, orient='records', indent=2)
        print(f"✅ Exported to {filename}")

    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the dataset"""
        stats = {
            'total_responses': len(df),
            'date_range': {
                'start': df['created_at'].min(),
                'end': df['created_at'].max()
            },
            'demographics': {
                'grade_levels': df['grade_level'].value_counts().to_dict(),
                'gender_distribution': df['gender'].value_counts().to_dict(),
            },
            'satisfaction_metrics': {
                'overall_satisfaction': {
                    'mean': float(df['overall_satisfaction'].mean()),
                    'std': float(df['overall_satisfaction'].std()),
                    'min': int(df['overall_satisfaction'].min()),
                    'max': int(df['overall_satisfaction'].max())
                },
                'attendance_rate': {
                    'mean': float(df['attendance_rate'].mean()),
                    'std': float(df['attendance_rate'].std())
                },
                'grade_average': {
                    'mean': float(df['grade_average'].mean()),
                    'std': float(df['grade_average'].std())
                }
            },
            'iso21001_compliance': {
                'learner_needs_avg': float(df[[
                    'curriculum_relevance_rating',
                    'learning_pace_appropriateness',
                    'individual_support_availability',
                    'learning_style_accommodation'
                ]].mean().mean()),
                'safety_avg': float(df[[
                    'physical_safety_rating',
                    'psychological_safety_rating',
                    'bullying_prevention_effectiveness',
                    'emergency_preparedness_rating'
                ]].mean().mean()),
                'wellbeing_avg': float(df[[
                    'mental_health_support_rating',
                    'stress_management_support',
                    'physical_health_support',
                    'overall_wellbeing_rating'
                ]].mean().mean())
            }
        }

        return stats


if __name__ == '__main__':
    print("🎓 ISO 21001 Synthetic Data Generator")
    print("=" * 50)

    # Initialize generator
    generator = ISO21001DataGenerator(seed=42)

    # Generate training dataset
    print("\n📊 Generating training dataset...")
    df = generator.generate_dataset(n_samples=1000, days_back=180)

    # Generate validation dataset with different seed
    print("\n📊 Generating validation dataset...")
    validator = ISO21001DataGenerator(seed=123)
    df_val = validator.generate_dataset(n_samples=200, days_back=180)

    # Export datasets
    print("\n💾 Exporting datasets...")
    generator.export_to_csv(df, 'data/iso21001_training_data.csv')
    generator.export_to_json(df, 'data/iso21001_training_data.json')
    generator.export_to_csv(df_val, 'data/iso21001_validation_data.csv')

    # Print statistics
    print("\n📈 Dataset Statistics:")
    stats = generator.get_dataset_statistics(df)
    print(json.dumps(stats, indent=2))

    print("\n✅ Data generation complete!")
