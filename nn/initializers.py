from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InitializerError


class Initializer(ABC):

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, shape, **kwargs):
        pass


class ConstantInitializer(Initializer):

    def __init__(self, value=1):
        self.value = value

    def __call__(self, shape, **kwargs):
        np.ones(shape) * self.value


class RandomNormalInitializer(Initializer):

    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, shape, **kwargs):
        return np.random.normal(self.mean, self.std, shape)


class RandomUniformInitializer(Initializer):

    def __init__(self, min_val=0, max_val=1):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, shape, **kwargs):
        return np.random.uniform(self.min_val, self.max_val, shape)


class GlorotUniformInitialization(Initializer):

    def __call__(self, shape, **kwargs):
        layer = kwargs['layer']

        if len(layer.input_shape) != 1 or len(layer.output_shape) != 1:
            raise InitializerError('Glorot uniform initialization works only with layers with flat input and output')

        inputs = layer.input_shape[0]
        outputs = layer.output_shape[0]

        x = np.sqrt(6 / (inputs + outputs))
        return np.random.uniform(-x, x, shape)
