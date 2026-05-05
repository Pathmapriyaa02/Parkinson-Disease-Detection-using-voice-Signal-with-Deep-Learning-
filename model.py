from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    UpSampling1D, Bidirectional,
    LSTM, RepeatVector, BatchNormalization
)

def build_model(input_shape):

    inputs = Input(shape=input_shape)

    # CNN Encoder
    x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2, padding='same')(x)

    # BiLSTM Bottleneck
    x = Bidirectional(LSTM(64, return_sequences=False))(x)

    # Repeat for Decoder
    x = RepeatVector(input_shape[0] // 4)(x)

    # Decoder BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)

    x = UpSampling1D(2)(x)
    outputs = Conv1D(input_shape[1], 3, activation='linear', padding='same')(x)

    model = Model(inputs, outputs)
    return model
