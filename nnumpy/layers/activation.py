from .base import Layer
from nnumpy.utils.shortcuts import get_activation


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = get_activation(activation)
        self.__state = None

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        self.__state = self.activation.call(x)
        return self.__state

    def backpropagate(self, delta):
        deriv = self.activation.deriv(self.__state)
        return delta * deriv
