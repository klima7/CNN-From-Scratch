import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets.fashion_mnist import load_data as load_data_Fashion_MNIST

import nn.layers
import nn.network

model = None
fmnist_x, fmnist_y = None, None


def prepare(n_samples=5000):
    global model, fmnist_x, fmnist_y

    (fmnist_x, fmnist_y), (_, _) = load_data_Fashion_MNIST()
    fmnist_x = fmnist_x[:n_samples, ..., np.newaxis] / 255.0
    fmnist_y = np.array(tf.one_hot(fmnist_y[:n_samples], 10))

    layers = [
        nn.layers.InputLayer((28, 28, 1)),
        nn.layers.Conv2DLayer(8, kernel_size=3),
        nn.layers.Pool2DLayer(pool_size=2),
        nn.layers.ActivationLayer('relu'),

        nn.layers.FlattenLayer(),

        nn.layers.DenseLayer(128),
        nn.layers.BiasLayer(),
        nn.layers.ActivationLayer('relu'),
        nn.layers.DropoutLayer(0.25),

        nn.layers.DenseLayer(10),
        nn.layers.BiasLayer(),
    ]

    model = nn.network.Sequential(layers)
    model.build('softmax_cce', metrics=['categorical_accuracy'])


def start():
    model.fit(
        fmnist_x,
        fmnist_y,
        learning_rate=0.01,
        epochs=1
    )
