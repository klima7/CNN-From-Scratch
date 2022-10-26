import numpy as np

from .base import Layer


class InputLayer(Layer):

    def __init__(self, shape):
        super().__init__()
        self.shape = np.array(shape)

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
        return delta
