from abc import ABC, abstractmethod

import numpy as np

from .base import Layer
from ..exceptions import InvalidParameterException, InvalidShapeException


class ResizeLayer(Layer, ABC):

    def __init__(self, size, mode):
        super().__init__()
        self.size = self._get_unified_size(size)
        self.mode = mode

    @property
    def slices_count(self):
        return self.input_shape[-1]

    @property
    def input_slice_size(self):
        return self.input_shape[:-1]

    @property
    @abstractmethod
    def output_slice_size(self):
        pass

    def validate_input_shape(self):
        if len(self.input_shape) != 3:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 3D, but is {self.input_shape}')

    def get_output_shape(self):
        return tuple((*self.output_slice_size, self.slices_count))

    @staticmethod
    def _get_unified_size(size):
        if isinstance(size, int):
            return (size, size), (size, size)
        elif isinstance(size, tuple):
            if isinstance(size[0], int):
                return (size[0], size[0]), (size[1], size[1])
            elif isinstance(size[0], tuple):
                return size

    @staticmethod
    def _add_padding(array, padding_sizes, mode):
        if mode == 'valid':
            return np.array(array)
        elif mode == 'same':
            return np.pad(array, padding_sizes, mode='constant', constant_values=0)
        elif mode == 'duplicate':
            return np.pad(array, padding_sizes, mode='edge')
        else:
            raise InvalidParameterException(f'Invalid padding mode: {mode}')

    @staticmethod
    def _remove_padding(array, paddings_size):
        slices = []
        for padding_size, dim in zip(paddings_size, array.shape):
            s = slice(padding_size[0], dim - padding_size[1])
            slices.append(s)
        return array[tuple(slices)]


class Padding2DLayer(ResizeLayer):

    def __init__(self, size, mode='same'):
        super().__init__(size, mode)

    @property
    def output_slice_size(self):
        if self.mode == 'valid':
            return self.input_slice_size
        paddings_sum = np.array([sum(self.size[0]), sum(self.size[1])])
        return self.input_slice_size + paddings_sum

    def propagate(self, x):
        size = [*self.size, (0, 0)]
        return self._add_padding(x, size, self.mode)

    def backpropagate(self, delta):
        size = [*self.size, (0, 0)]
        return self._remove_padding(delta, size)


class Crop2DLayer(ResizeLayer):

    def __init__(self, size, mode='same'):
        super().__init__(size, mode)

    @property
    def output_slice_size(self):
        crop_sum = np.array([sum(self.size[0]), sum(self.size[1])])
        return self.input_slice_size - crop_sum

    def propagate(self, x):
        size = [*self.size, (0, 0)]
        return self._remove_padding(x, size)

    def backpropagate(self, delta):
        size = [*self.size, (0, 0)]
        return self._add_padding(delta, size, self.mode)
