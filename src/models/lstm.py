import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTMV1, Dense, Embedding, Dropout

def build_lstm_model(input_dim, output_dim, seq_length):
    """
    Build a simple and efficient LSTM-based model for music generation.
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(128, activation='relu'),  # Reduced dense layer size
        Dense(output_dim, activation='softmax')  # Output layer
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
