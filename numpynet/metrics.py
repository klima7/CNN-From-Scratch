from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):

    NAME = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, targets, predictions):
        pass

    @property
    @abstractmethod
    def value(self):
        pass


class CategoricalAccuracy(Metric):

    NAME = 'categorical_accuracy'

    def __init__(self):
        self.total = 0
        self.count = 0

    def reset(self):
        self.total = 0
        self.count = 0

    def update(self, targets, predictions):
        predicted_classes = np.argmax(predictions, axis=1)
        target_classes = np.argmax(targets, axis=1)
        correct_count = np.sum(predicted_classes == target_classes)
        self.total += correct_count
        self.count += len(targets)

    @property
    def value(self):
        return self.total / self.count if self.count > 0.0 else 0.0
