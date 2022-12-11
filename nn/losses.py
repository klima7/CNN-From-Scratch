from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidParameterException


class Loss(ABC):

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def call(self, prediction, target):
        pass

    @abstractmethod
    def deriv(self, prediction, target):
        pass


class MseLoss(Loss):

    def call(self, prediction, target):
        return np.sum(np.power(target - prediction, 2))

    def deriv(self, prediction, target):
        return target - prediction


class CategoricalCrossEntropyLoss(Loss):

    def __init__(self, softmax=False):
        self.softmax = softmax

    def call(self, prediction, target):
        self.__ensure_is_onehot(target)

        if self.softmax:
            prediction = self.__softmax(prediction)

        logs = np.log(prediction, where=prediction > 0)
        return - logs @ target

    def deriv(self, prediction, target):
        self.__ensure_is_onehot(target)

        if self.softmax:
            return self.__deriv_with_softmax(prediction, target)
        else:
            return self.__deriv_without_softmax(prediction, target)

    @staticmethod
    def __deriv_with_softmax(prediction, target):
        prediction = CategoricalCrossEntropyLoss.__softmax(prediction)
        one_pos = CategoricalCrossEntropyLoss.__get_one_position(target)
        delta = -prediction
        delta[one_pos] = 1 - prediction[one_pos]
        return delta

    @staticmethod
    def __deriv_without_softmax(prediction, target):
        return np.divide(target, prediction, out=np.zeros_like(prediction), where=prediction != 0)

    @staticmethod
    def __softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        return e / s if s != 0 else np.zeros_like(e)

    @staticmethod
    def __ensure_is_onehot(target):
        ok = np.all(np.logical_or(target == 0, target == 1)) and np.sum(target) == 1
        if not ok:
            raise InvalidParameterException('Softmax requires labels to be onehot encoded')

    @staticmethod
    def __get_one_position(target):
        return np.where(target == 1)[0][0]


def get_loss(loss):
    if isinstance(loss, str):
        return get_loss_from_name(loss)
    elif isinstance(loss, Loss):
        return loss
    else:
        raise InvalidParameterException(f'Invalid loss: {loss}')


def get_loss_from_name(name):
    losses = {
        'mse': MseLoss,
        'cce': CategoricalCrossEntropyLoss,
    }

    if name not in losses.keys():
        raise InvalidParameterException(f'Unknown loss name: {name}')

    return losses[name]()
