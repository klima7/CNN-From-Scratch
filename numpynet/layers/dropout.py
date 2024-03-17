import numpy as np

from .base import Layer


class DropoutLayer(Layer):

    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate
        self.__keep_mask = None
        self.__scale_factor = 1 / (1-self.drop_rate)

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        if not self.nn.training:
            return x

        self.__keep_mask = np.random.binomial(1, 1-self.drop_rate, size=x.shape)
        return x * self.__keep_mask * self.__scale_factor

    def backpropagate(self, delta):
        return delta * self.__keep_mask * self.__scale_factor
