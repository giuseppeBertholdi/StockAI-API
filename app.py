from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.IA.news_api import fetch_news, analyze_sentiment
from src.IA.data_preprocessing import load_and_preprocess_data, add_technical_indicators, create_sequences_with_sentiment
from src.IA.model import build_lstm_attention_model
from src.IA.predict import generate_future_predictions
import os
import sys

# Adiciona o diretório src/ ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    """
    Rota explicativa da API.
    """
    return jsonify({
        "message": "Bem-vindo à API de Previsão de Ações!",
        "endpoints": {
            "/predict/<empresa>": {
                "method": "GET",
                "description": "Prever o preço futuro de uma ação.",
                "parameters": {
                    "empresa": "Nome da empresa (ex: PETR4)."
                },
                "example_request": "GET /predict/PETR4",
                "example_response": {
                    "empresa": "PETR4",
                    "future_predictions": [
                        {"original": -1.8556256, "formatted": "-1.86"},
                        {"original": 38.5234567, "formatted": "38.52"},
                        {"original": 38.8912345, "formatted": "38.89"}
                    ],
                    "future_variation": [1.32, 0.53, -0.53],
                    "future_increase": [1.32, 0.53, 0.00],
                    "technical_indicators": {
                        "rsi_14": {
                            "value": 48.5,
                            "description": "Índice de Força Relativa (RSI) mede a velocidade e mudança de movimentos de preços."
                        },
                        "sma_50": {
                            "value": 36.9,
                            "description": "Média Móvel Simples de 50 dias (SMA)."
                        },
                        "sma_200": {
                            "value": 37.2,
                            "description": "Média Móvel Simples de 200 dias (SMA)."
                        },
                        "ema_20": {
                            "value": 37.5,
                            "description": "Média Móvel Exponencial de 20 dias (EMA)."
                        },
                        "volume_medio": {
                            "value": 1000000,
                            "description": "Volume Médio dos últimos 20 dias."
                        }
                    },
                    "sentiment": {
                        "score": 0.1,
                        "classification": "ligeiramente positivo"
                    },
                    "news": ["Notícia 1", "Notícia 2"]
                }
            },
            "/docs": {
                "method": "GET",
                "description": "Documentação detalhada da API."
            }
        },
        "how_to_use": {
            "step_1": "Acesse o endpoint /predict/<empresa> para obter previsões.",
            "step_2": "Substitua <empresa> pelo código da ação desejada (ex: PETR4).",
            "step_3": "A resposta incluirá previsões, indicadores técnicos, sentimento e notícias."
        },
        "contact": {
            "email": "suporte@tradingai.com",
            "github": "https://github.com/giuseppeBertholdi"
        }
    })

def classify_sentiment(score: float) -> str:
    """
    Classifica o sentimento com base no score.

    Parâmetros:
        score (float): Score de sentimento.

    Retorna:
        str: Classificação do sentimento.
    """
    if score > 0.2:
        return "positivo"
    elif score > 0.05:
        return "ligeiramente positivo"
    elif score < -0.2:
        return "negativo"
    elif score < -0.05:
        return "ligeiramente negativo"
    else:
        return "neutro"

def calculate_percentage_change(predictions: list[float], last_price: float) -> list[float]:
    """
    Calcula a variação percentual das previsões em relação ao último preço conhecido.

    Parâmetros:
        predictions (List[float]): Lista de previsões futuras.
        last_price (float): Último preço conhecido.

    Retorna:
        List[float]: Variação percentual das previsões.
    """
    return [(pred - last_price) / last_price * 100 for pred in predictions]

def get_technical_indicators(data: pd.DataFrame) -> dict[str, any]:
    """
    Extrai indicadores técnicos do DataFrame.

    Parâmetros:
        data (pd.DataFrame): DataFrame com dados financeiros.

    Retorna:
        Dict[str, any]: Dicionário com indicadores técnicos e suas descrições.
    """
    return {
        "rsi_14": {
            "value": data['RSI'].iloc[-1],
            "description": "Índice de Força Relativa (RSI) mede a velocidade e mudança de movimentos de preços."
        },
        "sma_50": {
            "value": data['SMA_50'].iloc[-1],
            "description": "Média Móvel Simples de 50 dias (SMA)."
        },
        "sma_200": {
            "value": data['SMA_200'].iloc[-1],
            "description": "Média Móvel Simples de 200 dias (SMA)."
        },
        "ema_20": {
            "value": data['Fechamento'].ewm(span=20, adjust=False).mean().iloc[-1],
            "description": "Média Móvel Exponencial de 20 dias (EMA)."
        },
        "volume_medio": {
            "value": data['Volume'].rolling(window=20).mean().iloc[-1],
            "description": "Volume Médio dos últimos 20 dias."
        }
    }

@app.route('/predict/<string:empresa>', methods=['GET'])
def predict(empresa):
    """
    Endpoint para prever o preço futuro de uma ação.

    Parâmetros:
        empresa (str): Nome da empresa (ex: PETR4).

    Retorna:
        JSON: Previsões, indicadores técnicos, sentimento e notícias.
    """
    # Carregar e pré-processar os dados
    file_name = 'src/historico/dados_b3.csv'
    data = load_and_preprocess_data(file_name)
    data_with_indicators = add_technical_indicators(data)

    # Filtrar dados da empresa
    empresa_data = data_with_indicators[data_with_indicators['Empresa'] == empresa]
    if empresa_data.empty:
        return jsonify({"error": f"Empresa '{empresa}' não encontrada."}), 404

    # Verificar dados suficientes
    if len(empresa_data) < 20:
        return jsonify({"error": f"Não há dados suficientes para a empresa '{empresa}'. São necessários pelo menos 20 dias de dados."}), 400

    # Buscar notícias e calcular sentimento
    news = fetch_news(empresa)
    sentiment_score = analyze_sentiment(news)
    empresa_data.loc[:, 'Sentimento'] = sentiment_score  # Adiciona a coluna 'Sentimento'

    # Verifica se a coluna 'Sentimento' foi criada
    if 'Sentimento' not in empresa_data.columns:
        return jsonify({"error": "Erro ao adicionar a coluna 'Sentimento'."}), 500

    # Normalização robusta
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(empresa_data[['Fechamento', 'Volume', 'SMA_10', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'Bollinger_Mid', 'Bollinger_Upper', 'Bollinger_Lower', 'VWAP', 'Sentimento']])
    empresa_data.loc[:, ['Fechamento', 'Volume', 'SMA_10', 'SMA_30', 'RSI', 'MACD', 'MACD_signal', 'Bollinger_Mid', 'Bollinger_Upper', 'Bollinger_Lower', 'VWAP', 'Sentimento']] = scaled_features

    # Criar sequências com sentimento
    time_step = 20
    X, Y, scaler = create_sequences_with_sentiment(empresa_data, time_step)
    if len(X) == 0:
        return jsonify({"error": "Não há dados suficientes para criar sequências."}), 400

    # Construir e treinar o modelo (com 100 epochs)
    model = build_lstm_attention_model((time_step, X.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
    model.fit(X, Y, batch_size=64, epochs=100, callbacks=[early_stop, checkpoint])  # epochs reduzidos para 100

    # Gerar previsões futuras
    last_sequence = X[-1:]
    future_steps = 5
    future_predictions_final = generate_future_predictions(
        model=model,
        last_sequence=last_sequence,
        future_steps=future_steps,
        scaler=scaler,
        last_price=empresa_data['Fechamento'].iloc[-1],  # Último preço conhecido
        data=empresa_data  # DataFrame completo
    )

    # Calcular variação percentual e aumento em porcentagem
    last_price = empresa_data['Fechamento'].iloc[-1]
    future_variation = calculate_percentage_change(future_predictions_final["future_predictions"], last_price)
    future_increase = [max(0, change) for change in future_variation]  # Apenas aumentos

    # Extrair indicadores técnicos
    technical_indicators = get_technical_indicators(empresa_data)

    # Classificar sentimento
    sentiment_classification = classify_sentiment(sentiment_score)

    # Formatar previsões (valor original e valor formatado)
    formatted_predictions = [
        {"original": pred, "formatted": f"{pred:.2f}"}  # Exibe o valor original e o valor formatado com 2 casas decimais
        for pred in future_predictions_final["future_predictions"]
    ]

    # Retornar a resposta em JSON
    return jsonify({
        "empresa": empresa,
        "future_predictions": formatted_predictions,  # Previsões formatadas
        "future_increase": future_increase,  # Aumento em porcentagem
        "technical_indicators": technical_indicators,
        "sentiment": {
            "score": sentiment_score,
            "classification": sentiment_classification
        },
        "news": news
    })

@app.route('/docs', methods=['GET'])
def docs():
    """
    Endpoint de documentação da API.
    """
    return jsonify({
        "endpoints": {
            "/predict/<empresa>": {
                "method": "GET",
                "description": "Prever o preço futuro de uma ação.",
                "parameters": {
                    "empresa": "Nome da empresa (ex: PETR4)."
                }
            },
            "/docs": {
                "method": "GET",
                "description": "Documentação da API."
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)