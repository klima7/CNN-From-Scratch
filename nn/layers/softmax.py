import numpy as np

from .activation import Layer
from .log import BaseLogLayer
from ..exceptions import InvalidLayerPositionException, InvalidLabelsException


class SoftmaxLayer(Layer):

    def __init__(self):
        super().__init__()
        self.state = None

    def initialize(self):
        self.__ensure_is_last_layer()

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        e = np.exp(x)
        self.state = e / np.sum(e)
        return self.state

    def backpropagate(self, delta):
        self.__ensure_labels_are_onehot()
        one_pos = self.__get_one_position()
        new_delta = -self.state
        new_delta[one_pos] = 1 - self.state[one_pos]
        return new_delta

    def __ensure_is_last_layer(self):
        meaningful_layers = list(filter(lambda layer: not isinstance(layer, BaseLogLayer), self.nn.layers))
        layers_count = len(meaningful_layers)
        layer_pos = meaningful_layers.index(self)
        if layer_pos != layers_count - 1:
            raise InvalidLayerPositionException('Softmax layer must be the last one')

    def __ensure_labels_are_onehot(self):
        labels = self.nn.current_labels
        ok = np.all(np.logical_or(labels == 0, labels == 1)) and np.sum(labels) == 1
        if not ok:
            raise InvalidLabelsException('Softmax requires labels to be onehot encoded')

    def __get_one_position(self):
        return np.where(self.nn.current_labels == 1)[0][0]
