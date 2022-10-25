from abc import ABC, abstractmethod

import numpy as np


class Initializer(ABC):

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, shape):
        pass


class ConstantInitializer(Initializer):

    def __init__(self, value=1):
        self.value = value

    def __call__(self, shape):
        np.ones(shape) * self.value


class RandomNormalInitializer(Initializer):

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        return np.random.normal(self.mean, self.std, shape)


class RandomUniform(Initializer):

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, shape):
        return np.random.uniform(self.min_val, self.max_val, shape)
