from ..initializers import RandomUniformInitializer
from .base import Layer


class BiasLayer(Layer):

    def __init__(self, initializer=RandomUniformInitializer(-0.5, 0.5)):
        super().__init__()
        self.initializer = initializer
        self.bias = None

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return self.input_shape

    def initialize(self):
        kwargs = {
            'fan_in': self.input_shape[0],
            'fan_out': self.output_shape[0]
        }
        self.bias = self.initializer(self.input_shape, **kwargs)

    def propagate(self, x):
        return x + self.bias

    def backpropagate(self, delta):
        self.bias -= self.nn.learning_rate * delta
        return delta
