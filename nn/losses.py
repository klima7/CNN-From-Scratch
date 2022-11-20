from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidLabelsException


class Loss(ABC):

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, prediction, target):
        pass


class MseLoss(Loss):

    def __call__(self, prediction, target):
        return target - prediction


class CrossEntropyLoss(Loss):
    # This loss uses simplified formula assessing that last layer was softmax layer

    def __call__(self, prediction, target):
        self.__ensure_is_onehot(target)
        one_pos = self.__get_one_position(target)
        delta = -prediction
        delta[one_pos] = 1 - prediction[one_pos]
        return delta

    @staticmethod
    def __ensure_is_onehot(target):
        ok = np.all(np.logical_or(target == 0, target == 1)) and np.sum(target) == 1
        if not ok:
            raise InvalidLabelsException('Softmax requires labels to be onehot encoded')

    @staticmethod
    def __get_one_position(target):
        return np.where(target == 1)[0][0]
