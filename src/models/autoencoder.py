from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.models import Model

def build_autoencoder(input_shape):
    """
    Builds the convolutional autoencoder model.

    Args:
        input_shape (tuple): The shape of the input segments, e.g., (700, 1).

    Returns:
        Model: The compiled Keras autoencoder model.
    """
    input_layer = Input(shape=input_shape)

    # Encoder
    x = Conv1D(16, 3, activation='relu', padding='same')(input_layer)
    encoded = MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = Conv1D(16, 3, activation='relu', padding='same')(encoded)
    decoded = UpSampling1D(2)(x)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
