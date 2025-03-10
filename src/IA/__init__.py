# src/IA/__init__.py

from .config import NEWS_API_KEY, NEWS_API_URL
from .news_api import fetch_news, analyze_sentiment
from .data_preprocessing import load_and_preprocess_data, add_technical_indicators, create_sequences_with_sentiment
from .model import build_lstm_attention_model
from .predict import generate_future_predictions
