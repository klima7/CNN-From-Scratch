import numpy as np

from .base import Layer
from ..exceptions import InvalidShapeException


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()
        self.__y = None

    def get_output_shape(self):
        return self.input_shape

    def validate_input_shape(self):
        if len(self.input_shape) != 1:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 1D, but is {self.input_shape}')

    def propagate(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        self.__y = e / s if s != 0 else np.zeros_like(e)
        return self.__y

    def backpropagate(self, delta):
        dx = self.__y * delta
        s = dx.sum(axis=dx.ndim - 1, keepdims=True)
        dx -= self.__y * s
        return dx
