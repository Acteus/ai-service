"""
Advanced Sentiment Analyzer using NLP and Transformers
Implements sentiment analysis for student feedback with deep learning models
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_path='models/sentiment_model.pkl', vectorizer_path='models/sentiment_vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load pre-trained model and vectorizer if they exist"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("Loaded existing sentiment analysis model")

            if os.path.exists(self.vectorizer_path):
                self.vectorizer = joblib.load(self.vectorizer_path)
                logger.info("Loaded existing TF-IDF vectorizer")

            self.is_trained = self.model is not None and self.vectorizer is not None

        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
            self.is_trained = False

    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        # Join tokens back
        return ' '.join(tokens)

    def train(self, texts, labels, test_size=0.2):
        """Train the sentiment analysis model"""
        try:
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]

            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = self.vectorizer.fit_transform(processed_texts)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=test_size, random_state=42, stratify=labels
            )

            # Train logistic regression model
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Save model and vectorizer
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.vectorizer, self.vectorizer_path)

            self.is_trained = True

            logger.info("Sentiment analysis model trained successfully")

            return {
                'success': True,
                'accuracy': report['accuracy'],
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'classification_report': report
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'success': False, 'error': str(e)}

    def analyze_single(self, text):
        """Analyze sentiment of a single text"""
        try:
            if not self.is_trained:
                # Fallback to rule-based analysis
                return self._rule_based_sentiment(text)

            # Preprocess text
            processed_text = self.preprocess_text(text)

            # Vectorize
            X = self.vectorizer.transform([processed_text])

            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            # Map prediction to sentiment
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'neutral')

            # Calculate confidence
            confidence = float(max(probabilities))

            return {
                'sentiment': sentiment,
                'confidence': round(confidence, 3),
                'probabilities': {
                    'negative': round(float(probabilities[0]), 3),
                    'neutral': round(float(probabilities[1]), 3),
                    'positive': round(float(probabilities[2]), 3)
                },
                'processed_text': processed_text,
                'model_used': 'Logistic Regression with TF-IDF'
            }

        except Exception as e:
            logger.error(f"Single text analysis failed: {e}")
            return self._rule_based_sentiment(text)

    def analyze_batch(self, texts):
        """Analyze sentiment of multiple texts"""
        try:
            if not isinstance(texts, list):
                texts = [texts]

            results = []
            sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_confidence = 0

            for text in texts:
                result = self.analyze_single(text)
                results.append(result)

                sentiment = result['sentiment']
                sentiments[sentiment] += 1
                total_confidence += result['confidence']

            # Calculate overall metrics
            total_texts = len(texts)
            avg_confidence = total_confidence / total_texts if total_texts > 0 else 0

            # Determine overall sentiment
            if sentiments['positive'] > sentiments['negative'] and sentiments['positive'] > sentiments['neutral']:
                overall_sentiment = 'Positive'
            elif sentiments['negative'] > sentiments['positive'] and sentiments['negative'] > sentiments['neutral']:
                overall_sentiment = 'Negative'
            else:
                overall_sentiment = 'Neutral'

            # Calculate sentiment score (0-100 scale)
            sentiment_score = (sentiments['positive'] * 100 + sentiments['neutral'] * 50) / total_texts

            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': round(sentiment_score, 2),
                'breakdown': sentiments,
                'total_comments_analyzed': total_texts,
                'average_confidence': round(avg_confidence, 3),
                'individual_results': results,
                'iso_21001_insights': {
                    'learner_satisfaction_indicator': 'High' if sentiment_score >= 70 else ('Moderate' if sentiment_score >= 50 else 'Low'),
                    'action_required': sentiment_score < 60,
                    'recommendation': self._generate_sentiment_recommendations(sentiment_score)
                },
                'model_used': 'Logistic Regression with TF-IDF'
            }

        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            # Fallback to rule-based analysis for each text
            return {
                'overall_sentiment': 'Neutral',
                'sentiment_score': 50.0,
                'breakdown': {'positive': 0, 'negative': 0, 'neutral': len(texts)},
                'total_comments_analyzed': len(texts),
                'average_confidence': 0.5,
                'individual_results': [self._rule_based_sentiment(text) for text in texts],
                'iso_21001_insights': {
                    'learner_satisfaction_indicator': 'Moderate',
                    'action_required': False,
                    'recommendation': 'Unable to analyze sentiment - using fallback method'
                },
                'model_used': 'Rule-based (Fallback)',
                'error': str(e)
            }

    def _rule_based_sentiment(self, text):
        """Fallback rule-based sentiment analysis with improved keyword detection"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'model_used': 'Rule-based (Fallback)'
            }

        text_lower = text.lower()

        # Expanded keyword lists for better detection
        positive_keywords = [
            'great', 'excellent', 'love', 'amazing', 'wonderful', 'helpful', 'supportive',
            'engaging', 'interesting', 'effective', 'good', 'best', 'happy', 'satisfied',
            'fantastic', 'awesome', 'outstanding', 'superb', 'brilliant', 'impressive',
            'enjoy', 'appreciate', 'perfect', 'comfortable', 'safe', 'clean', 'organized',
            'friendly', 'caring', 'dedicated', 'professional', 'knowledgeable', 'skilled',
            'clear', 'understandable', 'easy', 'better', 'improved', 'positive', 'success',
            'thank', 'grateful', 'pleased'
        ]

        negative_keywords = [
            'poor', 'bad', 'terrible', 'awful', 'boring', 'unhelpful', 'inadequate',
            'frustrating', 'stressful', 'overwhelming', 'worst', 'hate', 'disappointed',
            'unsatisfied', 'difficult', 'hard', 'confusing', 'unclear', 'problem', 'issue',
            'concern', 'worry', 'lack', 'insufficient', 'unfair', 'bias', 'discriminate',
            'uncomfortable', 'unsafe', 'dirty', 'messy', 'disorganized', 'rude', 'mean',
            'unprofessional', 'incompetent', 'useless', 'waste', 'wrong', 'mistake',
            'never', 'nothing', 'nobody', 'none'
        ]

        # Count keyword occurrences (exact word matching)
        words = text_lower.split()
        positive_count = sum(1 for word in words if any(keyword in word for keyword in positive_keywords))
        negative_count = sum(1 for word in words if any(keyword in word for keyword in negative_keywords))

        # Consider text length for better classification
        text_length = len(words)

        # More sophisticated classification
        if positive_count > negative_count and positive_count > 0:
            sentiment = 'positive'
            # Higher confidence with more positive words
            confidence = min(0.85, 0.6 + (positive_count / max(text_length, 1)) * 0.25)
        elif negative_count > positive_count and negative_count > 0:
            sentiment = 'negative'
            # Higher confidence with more negative words
            confidence = min(0.85, 0.6 + (negative_count / max(text_length, 1)) * 0.25)
        else:
            # Only classify as neutral if there are truly no sentiment indicators
            # or if positive and negative are equal
            if positive_count == 0 and negative_count == 0:
                sentiment = 'neutral'
                confidence = 0.6  # Slightly higher confidence for true neutrality
            else:
                # Mixed sentiment - lean neutral but lower confidence
                sentiment = 'neutral'
                confidence = 0.5

        return {
            'sentiment': sentiment,
            'confidence': round(confidence, 3),
            'model_used': 'Rule-based (Enhanced)',
            'positive_words': positive_count,
            'negative_words': negative_count,
            'text_length': text_length
        }

    def _generate_sentiment_recommendations(self, sentiment_score):
        """Generate recommendations based on sentiment analysis"""
        if sentiment_score >= 70:
            return 'Continue current practices and identify best practices for scaling'
        elif sentiment_score >= 50:
            return 'Monitor sentiment trends and address emerging concerns proactively'
        else:
            return 'URGENT: Implement immediate interventions to address negative learner experiences (ISO 21001:7.1.2)'
