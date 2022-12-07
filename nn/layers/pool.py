from abc import ABC

from .base import Layer
from ..exceptions import InvalidParameterException

import numpy as np


class BasePoolLayer(Layer, ABC):

    def __init__(self, pool_size, variant):
        super().__init__()

        self.pool_size = np.array(pool_size)
        self.variant = variant
        self.pool_function = self.get_pool_function(variant)

    @property
    def slices_count(self):
        return self.input_shape[-1]

    @property
    def input_slice_size(self):
        return self.input_shape[:-1]

    @property
    def output_slice_size(self):
        return (self.input_slice_size - self.pool_size) // self.pool_size + 1

    def get_output_shape(self):
        return tuple((*self.output_slice_size, self.slices_count))

    @staticmethod
    def get_pool_function(variant):
        if variant == 'max':
            return np.max
        elif variant == 'avg':
            return np.mean
        else:
            raise InvalidParameterException(f'Invalid pool variant: {variant}')


class Pool1DLayer(BasePoolLayer):

    def __init__(self, pool_size=2, stride=None, variant='max'):
        super().__init__(pool_size, stride, variant)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 2

    def propagate(self, x):
        output = np.zeros(self.output_shape, dtype=x.dtype)
        for i in range(output.shape[0]):
            group = x[i*self.stride:(i+1)*self.stride, :]
            output[i, :] = self.pool_function(group, axis=0)
        return output

    def backpropagate(self, delta):
        raise NotImplementedError


class Pool2DLayer(BasePoolLayer):

    def __init__(self, pool_size=(2, 2), variant='max'):
        super().__init__(pool_size, variant)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def propagate(self, x):
        output = np.zeros(self.output_shape, dtype=x.dtype)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                i_slice = np.s_[i*self.pool_size[0]:(i+1)*self.pool_size[0]]
                j_slice = np.s_[j*self.pool_size[1]:(j+1)*self.pool_size[1]]
                group = x[i_slice, j_slice, :]
                output[i, j, :] = self.pool_function(group, axis=(0, 1))
        return output

    def backpropagate(self, delta):
        return np.repeat(np.repeat(delta, self.pool_size[0], axis=0), self.pool_size[1], axis=1)
