from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidShapeError


class Layer(ABC):

    def __init__(self):
        self.nn = None
        self.prev_layer = None
        self.next_layer = None
        self.output_shape = None

    def __repr__(self):
        return self.__class__.__name__

    @property
    def input_shape(self):
        return self.prev_layer.output_shape if self.prev_layer else None

    def connect(self, nn, prev_layer, next_layer):
        self.nn = nn
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.output_shape = self.get_output_shape()

        if not self.is_input_shape_valid(self.input_shape):
            raise InvalidShapeError(f'Inferred layer input shape is invalid {self.input_shape}')

        self.initialize()

    def propagate_with_validation(self, x):
        if self.input_shape and x.shape != self.input_shape:
            raise InvalidShapeError(f'Array with invalid shape passed to propagate method. Should be {self.input_shape}, but is {x.shape}')
        propagated_data = self.propagate(x)
        if propagated_data.shape != self.output_shape:
            raise InvalidShapeError(f'Array with invalid shape returned from propagate method. Should be {self.output_shape}, but is {propagated_data.shape}')
        return propagated_data

    def backpropagate_with_validation(self, delta):
        if delta.shape != self.output_shape:
            raise InvalidShapeError(f'Array with invalid shape passed to backpropagate method. Should be {self.output_shape}, but is {delta.shape}')
        prev_delta = self.backpropagate(delta)
        if prev_delta.shape != self.input_shape:
            raise InvalidShapeError(f'Backpropagate method returned array with invalid shape. Should be {self.input_shape}, but is {prev_delta.shape}')
        return prev_delta

    def initialize(self):
        pass

    @abstractmethod
    def is_input_shape_valid(self, input_shape):
        pass

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def propagate(self, x):
        pass

    @abstractmethod
    def backpropagate(self, delta):
        pass


class InputLayer(Layer):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'{self.__class__.__name__}(shape: {self.shape})'

    @property
    def input_shape(self):
        return self.output_shape

    def is_input_shape_valid(self, input_shape):
        return input_shape == self.input_shape

    def get_output_shape(self):
        return self.shape

    def propagate(self, x):
        return x

    def backpropagate(self, delta):
        pass


class DenseLayer(Layer):

    def __init__(self, neurons_count):
        super().__init__()
        self.neurons_count = neurons_count
        self.input_data = None
        self.weights = None

    def __repr__(self):
        return f'{self.__class__.__name__}(neurons_count: {self.neurons_count})'

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 1

    def get_output_shape(self):
        return tuple((self.neurons_count,))

    def initialize(self):
        self.weights = np.random.rand(self.neurons_count, self.input_shape[0])

    def propagate(self, x):
        self.input_data = x
        return x @ self.weights.T

    def backpropagate(self, delta):
        next_delta = self.__get_next_delta(delta)
        self.__adjust_weights(delta)
        return next_delta

    def __adjust_weights(self, delta):
        weights_delta = self.nn.learning_rate * delta.T @ self.input_data
        self.weights += weights_delta

    def __get_next_delta(self, delta):
        next_delta = delta @ self.weights
        return next_delta


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.state = None

    def __repr__(self):
        return f'{self.__class__.__name__}(activation: {self.activation})'

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        state = self.activation.call(x)
        self.state = state.reshape(-1, 1)
        return state

    def backpropagate(self, delta):
        return self.__get_next_delta(delta)

    def __get_next_delta(self, delta):
        deriv = self.activation.deriv(self.state.T)
        return delta * deriv
