from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Concatenate, Input, BatchNormalization, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MaxNorm
import os

def build_lstm_attention_model(input_shape, lstm_units=[200, 150, 100], dropout_rate=0.4, learning_rate=0.0005, l2_reg=0.001):
    """
    Constrói um modelo LSTM com camadas de atenção e normalização.

    Parâmetros:
        input_shape (tuple): Formato da entrada (timesteps, features).
        lstm_units (list): Número de unidades em cada camada LSTM.
        dropout_rate (float): Taxa de dropout para regularização.
        learning_rate (float): Taxa de aprendizado do otimizador.
        l2_reg (float): Fator de regularização L2.

    Retorna:
        model (Model): Modelo Keras compilado.
    """
    # Camada de entrada
    inputs = Input(shape=input_shape)

    # Primeira camada LSTM
    x = LSTM(
        lstm_units[0],
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(3.0)
    )(inputs)
    x = BatchNormalization()(x)
    x = LayerNormalization()(x)  # Adiciona normalização de camada
    x = Dropout(dropout_rate)(x)

    # Segunda camada LSTM
    x = LSTM(
        lstm_units[1],
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(3.0)
    )(x)
    x = BatchNormalization()(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Mecanismo de atenção
    attention = Attention(use_scale=True)([x, x])  # Adiciona escala para melhorar a atenção
    concat = Concatenate()([x, attention])

    # Terceira camada LSTM
    x = LSTM(
        lstm_units[2],
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
        kernel_constraint=MaxNorm(3.0)
    )(concat)
    x = BatchNormalization()(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Camadas densas finais
    x = Dense(50, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    outputs = Dense(1, kernel_regularizer=l2(l2_reg))(x)

    # Criação do modelo
    model = Model(inputs, outputs)

    # Compilação do modelo
    optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)  # Adiciona clipvalue para evitar exploding gradients
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model

def get_callbacks(model_name, logs_dir='logs', patience=20):
    """
    Retorna callbacks para treinamento do modelo.

    Parâmetros:
        model_name (str): Nome do modelo para salvar checkpoints.
        logs_dir (str): Diretório para salvar logs do TensorBoard.
        patience (int): Paciência para early stopping.

    Retorna:
        list: Lista de callbacks.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6),
        TensorBoard(log_dir=logs_dir)
    ]
    return callbacks