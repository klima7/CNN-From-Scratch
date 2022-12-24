from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from .utils.shortcuts import get_loss, get_metric
from .utils.statistics import RollingAverage
from .exceptions import LayerConnectingException, PropagationException, BackpropagationException, NetworkException


class Sequential(BaseEstimator, ClassifierMixin):

    def __init__(self, layers):
        self.layers = layers
        self.loss = None
        self.epochs = None
        self.learning_rate = None
        self.training = False
        self.total_params_count = 0
        self.is_build = False
        self.metrics = []
        self._history = defaultdict(list)

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    @property
    def history(self):
        return dict(self._history)

    def add(self, layer):
        self.layers.append(layer)
        self.is_build = False

    def build(self, loss='mse', metrics=()):
        self.loss = get_loss(loss)
        self.metrics = [get_metric(metric) for metric in metrics]
        self.__connect_layers()
        self.total_params_count = sum([layer.params_count for layer in self.layers])
        self._history.clear()
        self.is_build = True

    def fit(self, xs, ys, epochs=1, learning_rate=0.001, validation_data=None):
        self.__assert_build()
        self.epochs = epochs
        self.learning_rate = learning_rate

        for epoch_no in range(self.epochs):
            self.__learn_epoch(xs, ys, epoch_no + 1)

            if validation_data is not None:
                self.__perform_validation(validation_data)

    def predict(self, xs):
        self.__assert_build()
        iterator = tqdm(xs, desc='Predict', total=len(xs))
        predictions = [self.__propagate(x) for x in iterator]
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

        # reset metrics
        loss_rolling_avg = RollingAverage()
        for metric in self.metrics:
            metric.reset()

        self.training = True

        iterator = tqdm(enumerate(zip(xs, ys)), total=len(xs), desc=f'Epoch {epoch_no:<2}')
        for i, (x, y) in iterator:
            prediction, loss = self.__learn_single(x, y)

            # update metrics
            loss_rolling_avg.update(loss)
            for metric in self.metrics:
                metric.update(np.array([y]), np.array([prediction]))

            # show metrics
            iterator.set_postfix_str(self.__get_metrics_string(loss=loss_rolling_avg.value))

        self.training = False

        # add metrics to history
        self._history['loss'].append(loss_rolling_avg.value)
        for metric in self.metrics:
            self._history[metric.NAME].append(metric.value)

    def __learn_single(self, x, y):
        prediction = self.__propagate(x)
        loss = self.loss.call(prediction, y)
        delta = self.loss.deriv(prediction, y)
        self.__backpropagate(delta)
        return prediction, loss

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

    def __perform_validation(self, validation_data):
        val_xs, val_ys = validation_data

        loss_rolling_avg = RollingAverage()
        for metric in self.metrics:
            metric.reset()

        iterator = tqdm(zip(val_xs, val_ys), desc='Validate', total=len(val_xs))
        for x, y in iterator:
            prediction = self.__propagate(x)
            loss = self.loss.call(prediction, y)
            loss_rolling_avg.update(loss)
            for metric in self.metrics:
                metric.update(np.array([prediction]), np.array([y]))

            iterator.set_postfix_str(self.__get_metrics_string(loss_rolling_avg.value, prefix='val_'))

        self._history['val_loss'].append(loss_rolling_avg.value)
        for metric in self.metrics:
            self._history[f'val_{metric.NAME}'].append(metric.value)

    def __get_metrics_string(self, loss, prefix=''):
        parts = [f'{prefix}{metric.NAME}={metric.value:.4f}' for metric in self.metrics]
        parts.insert(0, f'{prefix}loss={loss:.4f}')
        return ', '.join(parts)

    def __assert_build(self):
        if not self.is_build:
            raise NetworkException('Network must be build to perform requested operation')

    @staticmethod
    def __shuffle(xs, ys):
        permutation = np.random.permutation(len(xs))
        return xs[permutation], ys[permutation]
