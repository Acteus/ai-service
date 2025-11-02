"""
Advanced Student Clusterer using Machine Learning
Implements clustering algorithms for student segmentation and targeted interventions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class StudentClusterer:
    def __init__(self, model_path='models/clusterer.pkl', scaler_path='models/cluster_scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.pca = None
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
                logger.info("Loaded existing clustering model")

            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Loaded existing scaler")

            self.is_trained = self.model is not None and self.scaler is not None

        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            self.is_trained = False

    def preprocess_data(self, data):
        """Preprocess student data for clustering"""
        if isinstance(data, list):
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("Data must be a list of dictionaries or pandas DataFrame")

        # Select relevant features for clustering
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
            'overall_satisfaction', 'grade_average', 'attendance_rate',
            'participation_score'
        ]

        # Filter to available columns
        available_columns = [col for col in feature_columns if col in df.columns]
        if not available_columns:
            raise ValueError("No suitable features found for clustering")

        # Extract features
        X = df[available_columns].fillna(0)  # Fill NaN with 0

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA for dimensionality reduction (optional, helps with high-dimensional data)
        if X_scaled.shape[1] > 10:
            self.pca = PCA(n_components=min(10, X_scaled.shape[1]))
            X_scaled = self.pca.fit_transform(X_scaled)

        return X_scaled, df

    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using silhouette score"""
        best_score = -1
        best_k = 2

        for k in range(2, min(max_clusters + 1, len(X))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logger.warning(f"Error evaluating k={k}: {e}")
                continue

        return best_k, best_score

    def train(self, data, k=None, algorithm='kmeans'):
        """Train clustering model (alias for cluster method for consistency with training pipeline)"""
        return self.cluster(data, k, algorithm)

    def cluster(self, data, k=None, algorithm='kmeans'):
        """Perform clustering on student data"""
        try:
            # Preprocess data
            X_scaled, df = self.preprocess_data(data)

            if len(X_scaled) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient data for clustering (minimum 2 samples required)',
                    'clusters': []
                }

            # Determine number of clusters
            if k is None or k < 2:
                optimal_k, silhouette = self.find_optimal_clusters(X_scaled)
                k = optimal_k
                logger.info(f"Optimal number of clusters determined: {k} (silhouette score: {silhouette:.3f})")

            # Perform clustering
            if algorithm.lower() == 'kmeans':
                self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif algorithm.lower() == 'dbscan':
                # Use DBSCAN with automatic parameter selection
                self.model = DBSCAN(eps=0.5, min_samples=5)
            elif algorithm.lower() == 'hierarchical':
                self.model = AgglomerativeClustering(n_clusters=k)
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

            # Fit and predict
            labels = self.model.fit_predict(X_scaled)

            # Analyze clusters
            cluster_analysis = self._analyze_clusters(df, labels, k)

            # Calculate clustering metrics
            try:
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    calinski = calinski_harabasz_score(X_scaled, labels)
                else:
                    silhouette = 0
                    calinski = 0
            except:
                silhouette = 0
                calinski = 0

            # Save model and scaler
            try:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                self.is_trained = True
            except Exception as e:
                logger.warning(f"Could not save model: {e}")

            return {
                'success': True,
                'num_clusters': len(np.unique(labels)),
                'algorithm': algorithm,
                'clusters': cluster_analysis,
                'metrics': {
                    'silhouette_score': round(silhouette, 3),
                    'calinski_harabasz_score': round(calinski, 3),
                    'total_samples': len(df),
                    'noise_points': int(np.sum(labels == -1)) if algorithm.lower() == 'dbscan' else 0
                },
                'insights': self._generate_clustering_insights(cluster_analysis),
                'model_used': f'{algorithm.upper()} Clustering'
            }

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'clusters': []
            }

    def _analyze_clusters(self, df, labels, k):
        """Analyze characteristics of each cluster"""
        clusters = []

        for i in range(k):
            cluster_mask = labels == i
            cluster_data = df[cluster_mask]

            if len(cluster_data) == 0:
                continue

            # Calculate cluster statistics
            cluster_stats = {
                'cluster_id': i + 1,
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(df) * 100, 2),
                'average_satisfaction': round(cluster_data.get('overall_satisfaction', pd.Series([0])).mean(), 2),
                'average_performance': round(cluster_data.get('grade_average', pd.Series([0])).mean(), 2),
                'average_attendance': round(cluster_data.get('attendance_rate', pd.Series([0])).mean(), 2),
                'average_participation': round(cluster_data.get('participation_score', pd.Series([0])).mean(), 2),
                'characteristics': self._identify_cluster_characteristics(cluster_data),
                'risk_profile': self._assess_cluster_risk(cluster_data),
                'recommended_interventions': self._generate_interventions(cluster_data)
            }

            clusters.append(cluster_stats)

        return clusters

    def _identify_cluster_characteristics(self, cluster_data):
        """Identify key characteristics of a cluster"""
        characteristics = []

        # Analyze satisfaction levels
        avg_satisfaction = cluster_data.get('overall_satisfaction', pd.Series([0])).mean()
        if avg_satisfaction >= 4.0:
            characteristics.append("High satisfaction learners")
        elif avg_satisfaction >= 3.0:
            characteristics.append("Moderate satisfaction learners")
        else:
            characteristics.append("Low satisfaction learners")

        # Analyze academic performance
        avg_grades = cluster_data.get('grade_average', pd.Series([0])).mean()
        if avg_grades >= 3.5:
            characteristics.append("High academic performers")
        elif avg_grades >= 2.5:
            characteristics.append("Average academic performers")
        else:
            characteristics.append("Low academic performers")

        # Analyze attendance
        avg_attendance = cluster_data.get('attendance_rate', pd.Series([0])).mean()
        if avg_attendance >= 90:
            characteristics.append("Regular attendees")
        elif avg_attendance >= 75:
            characteristics.append("Moderate attendance")
        else:
            characteristics.append("Irregular attendees")

        # Analyze safety perceptions
        avg_safety = cluster_data.get('physical_safety_rating', pd.Series([0])).mean()
        if avg_safety >= 4.0:
            characteristics.append("High safety perception")
        elif avg_safety < 3.0:
            characteristics.append("Low safety perception")

        return characteristics

    def _assess_cluster_risk(self, cluster_data):
        """Assess risk profile of a cluster"""
        risk_factors = []

        # Low satisfaction
        if cluster_data.get('overall_satisfaction', pd.Series([0])).mean() < 3.0:
            risk_factors.append("Low satisfaction")

        # Poor academic performance
        if cluster_data.get('grade_average', pd.Series([0])).mean() < 2.5:
            risk_factors.append("Poor academic performance")

        # Low attendance
        if cluster_data.get('attendance_rate', pd.Series([0])).mean() < 75:
            risk_factors.append("Low attendance")

        # Safety concerns
        if cluster_data.get('physical_safety_rating', pd.Series([0])).mean() < 3.0:
            risk_factors.append("Safety concerns")

        # Determine overall risk level
        if len(risk_factors) >= 3:
            risk_level = "High Risk"
        elif len(risk_factors) >= 2:
            risk_level = "Medium Risk"
        elif len(risk_factors) >= 1:
            risk_level = "Low Risk"
        else:
            risk_level = "Low Risk"

        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'risk_score': len(risk_factors)
        }

    def _generate_interventions(self, cluster_data):
        """Generate recommended interventions for a cluster"""
        interventions = []

        avg_satisfaction = cluster_data.get('overall_satisfaction', pd.Series([0])).mean()
        avg_performance = cluster_data.get('grade_average', pd.Series([0])).mean()
        avg_safety = cluster_data.get('physical_safety_rating', pd.Series([0])).mean()

        if avg_satisfaction < 3.5:
            interventions.append("Implement targeted satisfaction improvement programs")

        if avg_performance < 2.5:
            interventions.append("Provide academic support and tutoring services")

        if avg_safety < 3.5:
            interventions.append("Enhance safety protocols and student support services")

        if len(interventions) == 0:
            interventions.append("Monitor cluster performance and maintain current support levels")

        return interventions

    def _generate_clustering_insights(self, clusters):
        """Generate overall insights from clustering results"""
        insights = []

        # Analyze cluster distribution
        sizes = [c['size'] for c in clusters]
        if len(sizes) > 1:
            size_variation = max(sizes) / min(sizes) if min(sizes) > 0 else 1
            if size_variation > 3:
                insights.append("Significant variation in cluster sizes - consider reviewing clustering parameters")

        # Identify at-risk clusters
        at_risk_clusters = [c for c in clusters if c['risk_profile']['risk_level'] in ['High Risk', 'Medium Risk']]
        if at_risk_clusters:
            insights.append(f"Identified {len(at_risk_clusters)} at-risk clusters requiring immediate attention")

        # Performance insights
        high_performers = [c for c in clusters if c['average_performance'] >= 3.5]
        if high_performers:
            insights.append(f"Found {len(high_performers)} high-performing clusters - identify and scale best practices")

        # ISO 21001 specific insights
        insights.extend(self._generate_iso21001_insights(clusters))

        return insights

    def _generate_iso21001_insights(self, clusters):
        """Generate ISO 21001 specific insights from clustering"""
        iso_insights = []

        # Analyze learner needs across clusters
        learner_needs_scores = [c.get('average_satisfaction', 0) for c in clusters]
        if learner_needs_scores:
            avg_learner_needs = sum(learner_needs_scores) / len(learner_needs_scores)
            if avg_learner_needs < 3.5:
                iso_insights.append("ISO 21001:7.1 - Learner needs assessment indicates areas for improvement across clusters")

        # Safety and wellbeing analysis
        safety_concern_clusters = sum(1 for c in clusters if any('safety' in char.lower() for char in c.get('characteristics', [])))
        if safety_concern_clusters > 0:
            iso_insights.append(f"ISO 21001:7.2 - {safety_concern_clusters} clusters show safety concerns requiring attention")

        # Performance differentiation
        performance_variance = np.var([c.get('average_performance', 0) for c in clusters])
        if performance_variance > 0.5:
            iso_insights.append("ISO 21001:7.1.2 - Significant performance variance indicates need for differentiated support strategies")

        # Intervention prioritization
        high_risk_count = sum(1 for c in clusters if c.get('risk_profile', {}).get('risk_level') == 'High Risk')
        if high_risk_count > 0:
            iso_insights.append(f"ISO 21001:7.3 - {high_risk_count} high-risk clusters identified for priority intervention planning")

        return iso_insights
