import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential([

    layers.Rescaling(1./255,input_shape=(224,224,3)),

    layers.Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation='relu'
    ),

    layers.Dense(
        5,
        activation='softmax'
    )
])

model.summary()