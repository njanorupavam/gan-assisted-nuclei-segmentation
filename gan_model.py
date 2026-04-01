import tensorflow as tf
from tensorflow.keras import layers

# 🔹 Generator (Enhancer)
def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(256, 256, 1)),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),

        layers.Conv2D(1, 3, padding='same', activation='sigmoid')
    ])
    return model


# 🔹 Discriminator (Judge)
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(256, 256, 1)),

        layers.Conv2D(64, 3, strides=2, padding='same'),
        layers.LeakyReLU(),

        layers.Conv2D(128, 3, strides=2, padding='same'),
        layers.LeakyReLU(),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model