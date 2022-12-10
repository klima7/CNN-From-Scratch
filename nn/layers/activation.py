from .base import Layer


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.__state = None

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        self.__state = self.activation.call(x)
        return self.__state

    def backpropagate(self, delta):
        deriv = self.activation.deriv(self.__state)
        return delta * deriv
