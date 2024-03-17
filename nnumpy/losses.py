from abc import ABC, abstractmethod

import numpy as np


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
        return np.mean(np.power(target - prediction, 2))

    def deriv(self, prediction, target):
        return 2 * (prediction - target) / prediction.size


class CceLoss(Loss):

    def call(self, prediction, target):
        logs = np.log(prediction, where=prediction > 0)
        return - logs @ target

    def deriv(self, input, target):
        return sum(target) / sum(input) - target / input


class SoftmaxCceLoss(Loss):

    def call(self, prediction, target):
        prediction = self.__softmax(prediction)
        logs = np.log(prediction, where=prediction > 0)
        return - logs @ target

    def deriv(self, prediction, target):
        prediction = self.__softmax(prediction)
        one_pos = self.__get_one_position(target)
        delta = prediction
        delta[one_pos] = prediction[one_pos] - 1
        return delta

    @staticmethod
    def __softmax(x):
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        return e / s if s != 0 else np.zeros_like(e)

    @staticmethod
    def __get_one_position(target):
        return np.where(target == 1)[0][0]
