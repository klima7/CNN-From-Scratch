from abc import ABC, abstractmethod

import numpy as np

from ..exceptions import InvalidShapeException


class Layer(ABC):

    def __init__(self):
        self.nn = None
        self.prev_layer = None
        self.next_layer = None
        self.output_shape = None
        self.params_count = 0

    def __repr__(self):
        return self.__class__.__name__

    @property
    def input_shape(self):
        return self.prev_layer.output_shape if self.prev_layer else None

    def connect(self, nn, prev_layer, next_layer):
        self.nn = nn
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        self.validate_input_shape()
        self.output_shape = np.array(self.get_output_shape())
        self.initialize()

    def propagate_with_validation(self, x):
        if not np.array_equal(x.shape, self.input_shape):
            raise InvalidShapeException(f'Array with invalid shape passed to propagate method. Should be {self.input_shape}, but is {x.shape}')
        propagated_data = self.propagate(x)
        if not np.array_equal(propagated_data.shape, self.output_shape):
            raise InvalidShapeException(f'Array with invalid shape returned from propagate method. Should be {self.output_shape}, but is {propagated_data.shape}')
        return propagated_data

    def backpropagate_with_validation(self, delta):
        if not np.array_equal(delta.shape, self.output_shape):
            raise InvalidShapeException(f'Array with invalid shape passed to backpropagate method. Should be {self.output_shape}, but is {delta.shape}')
        prev_delta = self.backpropagate(delta)
        if not np.array_equal(prev_delta.shape, self.input_shape):
            raise InvalidShapeException(f'Backpropagate method returned array with invalid shape. Should be {self.input_shape}, but is {prev_delta.shape}')
        return prev_delta

    def initialize(self):
        pass

    def validate_input_shape(self):
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
