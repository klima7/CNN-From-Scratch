import numpy as np

from .base import Layer


class DropoutLayer(Layer):

    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_mask = None
        self.__scale_factor = 1 / (1-self.drop_rate)

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        if not self.nn.training:
            return x

        self.drop_mask = self.__create_drop_mask(x.shape)
        new_x = np.array(x)
        new_x[self.drop_mask] = 0
        new_x *= self.__scale_factor
        return new_x

    def backpropagate(self, delta):
        new_delta = np.array(delta)
        new_delta[self.drop_mask] = 0
        new_delta *= self.__scale_factor
        return new_delta

    def __create_drop_mask(self, shape):
        return np.random.rand(*shape) < self.drop_rate
