# news_api.py

import requests
from textblob import TextBlob
import numpy as np
from .config import NEWS_API_KEY, NEWS_API_URL

def fetch_news(company_name):
    params = {
        'q': company_name,
        'apiKey': NEWS_API_KEY,
        'language': 'pt',  # Notícias em português
        'sortBy': 'publishedAt',
        'pageSize': 10  # Limite de 10 notícias
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print(f"Erro ao buscar notícias: {response.status_code}")
        return []

def analyze_sentiment(news):
    sentiment_scores = []
    for article in news:
        text = article['title'] + " " + article['description']
        blob = TextBlob(text)
        sentiment_scores.append(blob.sentiment.polarity)  # Polaridade: -1 (negativo) a 1 (positivo)
    return np.mean(sentiment_scores) if sentiment_scores else 0  # Média do sentimento