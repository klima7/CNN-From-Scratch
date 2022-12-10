from ..initializers import RandomUniformInitializer
from ..exceptions import InvalidShapeException
from .base import Layer


class DenseLayer(Layer):

    def __init__(self, neurons_count, initializer=RandomUniformInitializer()):
        super().__init__()
        self.neurons_count = neurons_count
        self.initializer = initializer
        self.input_data = None
        self.weights = None

    def validate_input_shape(self):
        if len(self.input_shape) != 1:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 1D, but is {self.input_shape}')

    def get_output_shape(self):
        return tuple((self.neurons_count,))

    def initialize(self):
        shape = (self.neurons_count, self.input_shape[0])
        kwargs = {
            'fan_in': self.input_shape[0],
            'fan_out': self.output_shape[0]
        }
        self.weights = self.initializer(shape, **kwargs)
        self.params_count = self.weights.size

    def propagate(self, x):
        self.input_data = x
        return x @ self.weights.T

    def backpropagate(self, delta):
        next_delta = self.__get_next_delta(delta)
        self.__adjust_weights(delta)
        return next_delta.flatten()

    def __adjust_weights(self, delta):
        weights_delta = self.nn.learning_rate * delta.reshape(-1, 1) @ self.input_data.reshape(1, -1)
        self.weights += weights_delta

    def __get_next_delta(self, delta):
        next_delta = delta @ self.weights
        return next_delta
