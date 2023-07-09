"""
Autoencoder Model-Building Function
Signal analysis and classification of photoplethysmography (PPG) waveforms for predicting clinical outcomes
"""

import tensorflow as tf


def AE_model_builder(encoder_layer, activation_func, input_dim):
    AE_model = tf.keras.Sequential()
    enc_layer = encoder_layer.copy()
    decoder_layer = list(reversed(enc_layer))
    decoder_layer.pop(0)
    decoder_layer.append(input_dim)
    # Encoder
    AE_model.add(tf.keras.layers.Dense(enc_layer[0], input_shape=(input_dim,), activation=activation_func))
    enc_layer.pop(0)
    if len(enc_layer) != 0:
        for e_l in enc_layer:
            AE_model.add(tf.keras.layers.Dense(e_l, activation=activation_func))
    # Decoder
    for d_l in decoder_layer:
        AE_model.add(tf.keras.layers.Dense(d_l, activation=activation_func))
    AE_model.summary()
    return AE_model
