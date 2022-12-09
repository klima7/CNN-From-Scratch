from .base import Layer


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.state = None

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        self.state = self.activation.call(x)
        return self.state

    def backpropagate(self, delta):
        deriv = self.activation.deriv(self.state)
        return delta * deriv
