from .base import Layer
from ..activations import get_activation_from_name


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = get_activation_from_name(activation)
        self.__state = None

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        self.__state = self.activation.call(x)
        return self.__state

    def backpropagate(self, delta):
        deriv = self.activation.deriv(self.__state)
        return delta * deriv
