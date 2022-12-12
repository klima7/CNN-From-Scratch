from .base import Layer
from ..exceptions import InvalidParameterException, InvalidShapeException

import numpy as np


class Pool2DLayer(Layer):

    def __init__(self, pool_size, variant='max'):
        super().__init__()

        self.pool_size = np.broadcast_to(pool_size, (2,))
        self.variant = variant
        self.pool_function, self.arg_function = self.__get_pool_functions(variant)
        self.__indexes = None

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

    def validate_input_shape(self):
        if len(self.input_shape) != 3:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 3D')

        if self.input_shape[0] % self.pool_size[0] != 0 or self.input_shape[1] % self.pool_size[1] != 0:
            msg = f'{self.__class__.__name__} input shape {self.input_shape[:2]} must be dividable by pool size {self.pool_size}'
            raise InvalidShapeException(msg)

    def propagate(self, x):
        slices = self.__get_slices(x)
        pooled = self.pool_function(slices, axis=-1)
        self.__indexes = self.arg_function(slices, axis=-1)
        return pooled

    def backpropagate(self, delta):
        new_delta = np.zeros(self.input_shape, dtype=delta.dtype)

        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                for k in range(self.slices_count):
                    index = self.__indexes[i, j, k]
                    unpooled_i = index // self.pool_size[1]
                    unpooled_j = index % self.pool_size[1]
                    new_delta[unpooled_i, unpooled_j, k] = delta[i, j, k]

        return new_delta

    def __get_slices(self, data):
        cols_count = np.prod(self.pool_size)
        slices = np.zeros((self.output_shape[0], self.output_shape[1], self.slices_count, cols_count), dtype=data.dtype)

        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                i_slice = np.s_[i * self.pool_size[0]:(i + 1) * self.pool_size[0]]
                j_slice = np.s_[j * self.pool_size[1]:(j + 1) * self.pool_size[1]]
                group = np.transpose(data[i_slice, j_slice, :], (2, 0, 1))
                slices[i, j, ...] = group.reshape(self.slices_count, -1)
        return slices

    @staticmethod
    def __get_pool_functions(variant):
        if variant == 'max':
            return np.max, np.argmax
        elif variant == 'min':
            return np.min, np.argmin
        else:
            raise InvalidParameterException(f'Invalid pool variant: {variant}')
