from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidParameterException


class Metric(ABC):

    NAME = None

    @abstractmethod
    def __call__(self, predicted, expected):
        pass


class CategoricalAccuracy(Metric):

    NAME = 'categorical_accuracy'

    def __call__(self, predictions, target):
        predicted_classes = np.argmax(predictions, axis=1)
        target_classes = np.argmax(target, axis=1)
        correct = predicted_classes == target_classes
        return np.sum(correct) / correct.size


def get_metric(metric):
    if isinstance(metric, str):
        return get_metric_from_name(metric)
    elif isinstance(metric, Metric):
        return metric
    else:
        raise InvalidParameterException(f'Invalid loss: {loss}')


def get_metric_from_name(name):
    metrics = [CategoricalAccuracy]

    for metric in metrics:
        if metric.NAME == name:
            return metric()

    raise InvalidParameterException(f'Unknown metric name: {name}')
