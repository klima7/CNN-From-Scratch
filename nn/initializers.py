from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidParameterException


class Initializer(ABC):

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, shape, **kwargs):
        pass


class ConstantInitializer(Initializer):

    def __init__(self, value=0):
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

    def __init__(self, min_val=-0.5, max_val=0.5):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, shape, **kwargs):
        return np.random.uniform(self.min_val, self.max_val, shape)


class GlorotUniformInitialization(Initializer):

    def __call__(self, shape, **kwargs):
        fan_in, fan_out = kwargs['fan_in'], kwargs['fan_out']
        x = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-x, x, shape)


def get_initializer(initializer):
    if isinstance(initializer, str):
        return get_initializer_from_name(initializer)
    elif isinstance(initializer, Initializer):
        return initializer
    else:
        raise InvalidParameterException(f'Invalid initializer: {initializer}')


def get_initializer_from_name(name):
    initializers = {
        'constant': ConstantInitializer,
        'normal': RandomNormalInitializer,
        'uniform': RandomUniformInitializer,
        'glorot': GlorotUniformInitialization,
    }

    if name not in initializers.keys():
        raise InvalidParameterException(f'Unknown initializer name: {name}')

    return initializers[name]()
