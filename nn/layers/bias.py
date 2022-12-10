from ..initializers import RandomUniformInitializer
from .base import Layer


class BiasLayer(Layer):

    def __init__(self, initializer=None):
        super().__init__()
        self.initializer = initializer or RandomUniformInitializer(-0.5, 0.5)
        self.bias = None

    def get_output_shape(self):
        return self.input_shape

    def initialize(self):
        initializer_kwargs = {
            'fan_in': self.input_shape[0],
            'fan_out': self.output_shape[0]
        }
        self.bias = self.initializer(self.input_shape, **initializer_kwargs)
        self.params_count = self.bias.size

    def propagate(self, x):
        return x + self.bias

    def backpropagate(self, delta):
        self.bias -= self.nn.learning_rate * delta
        return delta
