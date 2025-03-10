import numpy as np
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def calculate_percentage_variation(predictions: List[float], last_price: float) -> List[float]:
    """
    Calcula a variação percentual das previsões em relação ao último preço conhecido.

    Parâmetros:
        predictions (List[float]): Lista de previsões futuras.
        last_price (float): Último preço conhecido.

    Retorna:
        List[float]: Variação percentual das previsões.
    """
    return [(pred - last_price) / last_price * 100 for pred in predictions]

def get_technical_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """
    Extrai indicadores técnicos do DataFrame.

    Parâmetros:
        data (pd.DataFrame): DataFrame com dados financeiros.

    Retorna:
        Dict[str, float]: Dicionário com indicadores técnicos.
    """
    return {
        "rsi_14": data['RSI'].iloc[-1],
        "sma_50": data['SMA_50'].iloc[-1],
        "sma_200": data['SMA_200'].iloc[-1]
    }

def generate_future_predictions(
    model,
    last_sequence: np.ndarray,
    future_steps: int,
    scaler: MinMaxScaler,
    last_price: float,
    data: pd.DataFrame
) -> Dict[str, any]:
    """
    Gera previsões futuras e informações adicionais.

    Parâmetros:
        model: Modelo treinado para fazer previsões.
        last_sequence (np.ndarray): Última sequência de dados usada para prever o próximo passo.
        future_steps (int): Número de passos futuros a serem previstos.
        scaler (MinMaxScaler): Scaler usado para normalizar os dados.
        last_price (float): Último preço conhecido.
        data (pd.DataFrame): DataFrame com dados financeiros.

    Retorna:
        Dict[str, any]: Dicionário com previsões, variação percentual e indicadores técnicos.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(future_steps):
        # Faz a previsão para o próximo passo
        next_prediction = model.predict(current_sequence)
        future_predictions.append(next_prediction[0, 0])

        # Atualiza a sequência para incluir a previsão
        next_input = np.append(current_sequence[:, -1, :-1], next_prediction)  # Mantém outras features e adiciona a previsão
        next_input = np.expand_dims(next_input, axis=0)  # Expande as dimensões para (1, features)
        next_input = np.expand_dims(next_input, axis=0)  # Expande as dimensões para (1, 1, features)
        current_sequence = np.append(current_sequence[:, 1:, :], next_input, axis=1)

    # Inverte a normalização das previsões
    future_predictions_reshaped = np.array(future_predictions).reshape(-1, 1)
    future_predictions_final = scaler.inverse_transform(future_predictions_reshaped).flatten()

    # Calcula a variação percentual
    future_variation = calculate_percentage_variation(future_predictions_final, last_price)

    # Extrai indicadores técnicos
    technical_indicators = get_technical_indicators(data)

    return {
        "future_predictions": future_predictions_final.tolist(),
        "future_variation": future_variation,
        "technical_indicators": technical_indicators
    }