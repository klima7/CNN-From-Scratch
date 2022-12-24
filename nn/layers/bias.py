from ..utils.shortcuts import get_initializer
from .base import Layer


class BiasLayer(Layer):

    def __init__(self, initializer='constant'):
        super().__init__()
        self.initializer = get_initializer(initializer)
        self.all_weights = [None]

    @property
    def bias(self):
        return self.all_weights[0]

    @bias.setter
    def bias(self, value):
        self.all_weights[0] = value

    def get_output_shape(self):
        return self.input_shape

    def initialize(self):
        initializer_kwargs = {
            'fan_in': self.input_shape[0],
            'fan_out': self.output_shape[0]
        }
        self.bias = self.initializer(self.input_shape, **initializer_kwargs)

    def propagate(self, x):
        return x + self.bias

    def backpropagate(self, delta):
        self.bias -= self.nn.learning_rate * delta
        return delta
