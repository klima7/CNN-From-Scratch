from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm

from .losses import get_loss
from .metrics import get_metric
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
        self.__history = defaultdict(list)

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    @property
    def history(self):
        return dict(self.__history)

    def add(self, layer):
        self.layers.append(layer)
        self.is_build = False

    def build(self, loss='mse', metrics=()):
        self.loss = get_loss(loss)
        self.metrics = [get_metric(metric) for metric in metrics]
        self.__connect_layers()
        self.total_params_count = sum([layer.params_count for layer in self.layers])
        self.__history.clear()
        self.is_build = True

    def fit(self, xs, ys, epochs=1, learning_rate=0.001, validation_data=None):
        self.__assert_build()
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.training = True
        for epoch_no in range(self.epochs):
            self.__learn_epoch(xs, ys, epoch_no + 1)
            self.__perform_validation(validation_data)
        self.training = False

        return self.__history

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
        losses_sum = 0
        avg_loss = 0

        iterator = tqdm(enumerate(zip(xs, ys)), total=len(xs), desc=f'Epoch {epoch_no:<2}')

        for i, (x, y) in iterator:
            loss = self.__learn_single(x, y)
            losses_sum += loss
            avg_loss = losses_sum / (i+1)
            iterator.set_postfix_str(f'loss={avg_loss:.3f}')

        self.__history['loss'].append(avg_loss)

    def __learn_single(self, x, y):
        prediction = self.__propagate(x)
        loss = self.loss.call(prediction, y)
        delta = self.loss.deriv(prediction, y)
        self.__backpropagate(delta)
        return loss

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
        iterator = tqdm(val_xs, desc='Validate', leave=False)
        predictions = np.array([self.__propagate(x) for x in iterator])

        metrics_results = self.__calculate_metrics(predictions, val_ys)
        for metric_name, metric_value in metrics_results.items():
            self.__history[metric_name].append(metric_value)

        tqdm(
            [],
            desc='Validate',
            initial=len(val_xs),
            total=len(val_xs),
            postfix=self.__cvt_metrics_results_to_string(metrics_results)
        ).display()

    def __calculate_metrics(self, predictions, target):
        samples_val_loss = [self.loss.call(pred_y, target_y) for pred_y, target_y in zip(predictions, target)]
        val_loss = np.mean(samples_val_loss)

        metrics_result = {'val_loss': val_loss}

        for metric in self.metrics:
            metrics_result[metric.NAME] = metric(predictions, target)

        return metrics_result

    @staticmethod
    def __cvt_metrics_results_to_string(metrics_results):
        text = ''
        for metric_name, metric_value in metrics_results.items():
            part = f'{metric_name}={metric_value:.3f}, '
            text += part
        return text[:-2]

    def __assert_build(self):
        if not self.is_build:
            raise NetworkException('Network must be build to perform requested operation')

    @staticmethod
    def __shuffle(xs, ys):
        permutation = np.random.permutation(len(xs))
        return xs[permutation], ys[permutation]
