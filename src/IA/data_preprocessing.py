import pandas as pd
import numpy as np
import os
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_name: str) -> pd.DataFrame:
    """
    Carrega e pré-processa os dados de um arquivo CSV.

    Parâmetros:
        file_name (str): Caminho do arquivo CSV.

    Retorna:
        pd.DataFrame: DataFrame pré-processado.

    Levanta:
        FileNotFoundError: Se o arquivo não for encontrado.
        ValueError: Se o arquivo estiver mal formatado.
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Erro: O arquivo '{file_name}' não foi encontrado.")

    try:
        # Carrega os dados
        all_data = pd.read_csv(file_name)

        # Verifica colunas obrigatórias
        required_columns = {'Data', 'Fechamento', 'Máxima', 'Mínima', 'Volume'}
        if not required_columns.issubset(all_data.columns):
            raise ValueError(f"Erro: O arquivo deve conter as colunas {required_columns}.")

        # Verifica dados faltantes
        print("\nDados faltantes por coluna:")
        print(all_data.isnull().sum())
        all_data = all_data.dropna()

        # Converte a coluna 'Data' para datetime
        all_data['Data'] = pd.to_datetime(all_data['Data'], errors='coerce')

        # Remove linhas com datas inválidas
        all_data = all_data.dropna(subset=['Data'])

        # Ordena por data
        all_data = all_data.sort_values(by='Data').reset_index(drop=True)

        return all_data

    except Exception as e:
        raise ValueError(f"Erro ao processar o arquivo: {e}")

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona indicadores técnicos ao DataFrame.

    Parâmetros:
        df (pd.DataFrame): DataFrame com dados financeiros.

    Retorna:
        pd.DataFrame: DataFrame com indicadores técnicos adicionados.
    """
    # Médias móveis
    df['SMA_10'] = df['Fechamento'].rolling(window=10, min_periods=1).mean()
    df['SMA_30'] = df['Fechamento'].rolling(window=30, min_periods=1).mean()
    df['SMA_50'] = df['Fechamento'].rolling(window=50, min_periods=1).mean()
    df['SMA_200'] = df['Fechamento'].rolling(window=200, min_periods=1).mean()

    # RSI (Relative Strength Index)
    delta = df['Fechamento'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['Fechamento'].ewm(span=12, adjust=False).mean() - df['Fechamento'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['Bollinger_Mid'] = df['Fechamento'].rolling(window=20, min_periods=1).mean()
    df['Bollinger_Upper'] = df['Bollinger_Mid'] + 2 * df['Fechamento'].rolling(window=20, min_periods=1).std()
    df['Bollinger_Lower'] = df['Bollinger_Mid'] - 2 * df['Fechamento'].rolling(window=20, min_periods=1).std()

    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['Volume'] * (df['Máxima'] + df['Mínima'] + df['Fechamento']) / 3).cumsum() / df['Volume'].cumsum()

    return df.dropna()

def create_sequences_with_sentiment(data: pd.DataFrame, time_step: int = 20) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Cria sequências de dados com sentimentos para treinamento do modelo.

    Parâmetros:
        data (pd.DataFrame): DataFrame com dados financeiros e sentimentos.
        time_step (int): Número de timesteps por sequência.

    Retorna:
        Tuple[np.ndarray, np.ndarray, MinMaxScaler]: Sequências de entrada (X), rótulos (Y) e scaler.
    """
    X, Y = [], []
    scaler = MinMaxScaler()

    # Normaliza apenas o preço de fechamento
    data['Fechamento_normalizado'] = scaler.fit_transform(data[['Fechamento']])

    # Seleciona colunas numéricas (exclui 'Data' e 'Empresa')
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns.drop('Fechamento')  # Remove 'Fechamento' para evitar duplicação

    for i in range(len(data) - time_step - 1):
        # Features técnicas (inclui 'Fechamento_normalizado' e outras colunas numéricas)
        sequence = data.iloc[i:(i + time_step)][numeric_columns].values
        # Sentimento das notícias
        sentiment = data.iloc[i + time_step, data.columns.get_loc('Sentimento')]
        # Expande as dimensões do sentimento para (timesteps, 1)
        sentiment_expanded = np.full((sequence.shape[0], 1), sentiment)
        # Concatena as features técnicas com o sentimento
        sequence_with_sentiment = np.concatenate([sequence, sentiment_expanded], axis=1)
        X.append(sequence_with_sentiment)
        # Rótulo (preço de fechamento normalizado)
        Y.append(data.iloc[i + time_step, data.columns.get_loc('Fechamento_normalizado')])

    return np.array(X), np.array(Y), scaler