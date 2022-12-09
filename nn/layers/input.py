import numpy as np

from ..exceptions import InvalidLayerPositionException
from .base import Layer


class InputLayer(Layer):

    def __init__(self, shape):
        super().__init__()
        self.shape = np.array(shape)

    @property
    def input_shape(self):
        return self.shape

    def validate_input_shape(self, input_shape):
        if super().input_shape is not None:
            raise InvalidLayerPositionException(f'{self.__class__.__name__} must be the first layer')

    def get_output_shape(self):
        return self.shape

    def propagate(self, x):
        return x

    def backpropagate(self, delta):
        return delta
