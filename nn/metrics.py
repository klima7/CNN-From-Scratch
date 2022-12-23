from abc import ABC, abstractmethod

import numpy as np


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
