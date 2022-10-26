import numpy as np

from .base import Layer


class FlattenLayer(Layer):

    def __init__(self):
        super().__init__()

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return tuple((np.prod(self.input_shape),))

    def propagate(self, x):
        return x.flatten()

    def backpropagate(self, delta):
        return delta.reshape(*self.input_shape)