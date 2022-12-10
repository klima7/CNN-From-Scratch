import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from .losses import MseLoss
from .exceptions import LayerConnectingException, PropagationException, BackpropagationException, NetworkException


class Sequential(BaseEstimator, ClassifierMixin):

    DEFAULT_EPOCHS = 1
    DEFAULT_LEARNING_RATE = 0.001

    def __init__(self, layers, loss=None):
        self.layers = layers
        self.loss = loss or MseLoss()
        self.epochs = None
        self.learning_rate = None
        self.training = False
        self.total_params_count = 0
        self.is_build = False

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    def add(self, layer):
        self.layers.append(layer)
        self.is_build = False

    def build(self):
        self.__connect_layers()
        self.total_params_count = sum([layer.params_count for layer in self.layers])
        self.is_build = True

    def fit(self, xs, ys, **kwargs):
        self.__assert_build()

        self.epochs = kwargs.get('epochs', self.DEFAULT_EPOCHS)
        self.learning_rate = kwargs.get('learning_rate', self.DEFAULT_LEARNING_RATE)

        self.training = True
        for epoch_no in range(self.epochs):
            self.__learn_epoch(xs, ys, epoch_no + 1)
        self.training = False

    def predict(self, xs):
        self.__assert_build()
        predictions = [self.__propagate(x) for x in xs]
        return np.array(predictions)

    def summary(self):
        print(f"{'NO':<4} | {'NAME':<20} | {'PARAMS':10} | {'INPUT':15} | {'OUTPUT':15}")
        for index, layer in enumerate(self.layers):
            name_text = str(layer)
            params_text = str(layer.params_count) if self.is_build else '?'
            input_text = f'{tuple(layer.input_shape)}' if self.is_build else '?'
            output_text = f'{tuple(layer.output_shape)}' if self.is_build else '?'
            print(f'{index:<4} | {name_text:<20} | {params_text:<10} | {input_text:<15} | {output_text:<15}')
        total_params_text = str(self.total_params_count) if self.is_build else '?'
        print(f'\nTotal parameters count: {total_params_text}')

    def __connect_layers(self):
        for i in range(len(self.layers)):
            self.__connect_single_layer(self.layers, i)

    def __connect_single_layer(self, layers, index):
        layer = layers[index]
        prev_layer = layers[index - 1] if index - 1 >= 0 else None
        next_layer = layers[index + 1] if index + 1 < len(layers) else None
        try:
            layer.connect(self, prev_layer, next_layer)
        except Exception as e:
            raise e from LayerConnectingException(index, layer)

    def __learn_epoch(self, xs, ys, epoch_no):
        xs, ys = self.__shuffle(xs, ys)

        for x, y in tqdm(zip(xs, ys), total=len(xs), desc=f'Epoch {epoch_no}'):
            self.__learn_single(x, y)

    def __learn_single(self, x, y):
        prediction = self.__propagate(x)
        delta = self.loss(prediction, y)
        self.__backpropagate(delta)

    def __propagate(self, x):
        for layer_no, layer in enumerate(self.layers):
            try:
                x = layer.propagate_save(x)
            except Exception as e:
                raise e from PropagationException(layer_no, layer)
        return x

    def __backpropagate(self, delta):
        for layer_no, layer in reversed(list(enumerate(self.layers))):
            try:
                delta = layer.backpropagate_save(delta)
            except Exception as e:
                raise e from BackpropagationException(layer_no, layer)

    def __assert_build(self):
        if not self.is_build:
            raise NetworkException('Network must be build to perform requested operation')

    @staticmethod
    def __shuffle(xs, ys):
        permutation = np.random.permutation(len(xs))
        return xs[permutation], ys[permutation]
