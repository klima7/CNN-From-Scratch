import numpy as np

from .base import Layer
from ..exceptions import InvalidParameterException, InvalidShapeException


class Padding2DLayer(Layer):

    def __init__(self, padding_size, mode='zeros'):
        super().__init__()
        self.mode = mode
        self.padding_size = Padding2DLayer.__get_unified_padding_size(padding_size)

    @property
    def slices_count(self):
        return self.input_shape[-1]

    @property
    def input_slice_size(self):
        return self.input_shape[:-1]

    @property
    def output_slice_size(self):
        if self.mode == 'valid':
            return self.input_slice_size
        paddings_sum = np.array([sum(self.padding_size[0]), sum(self.padding_size[1])])
        return self.input_slice_size + paddings_sum

    def validate_input_shape(self, input_shape):
        if len(input_shape) != 3:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 3D, but is {input_shape}')

    def get_output_shape(self):
        return tuple((*self.output_slice_size, self.slices_count))

    def propagate(self, x):
        padding_size = [*self.padding_size, (0, 0)]
        return Padding2DLayer.__add_padding(x, padding_size, self.mode)

    def backpropagate(self, delta):
        padding_size = [*self.padding_size, (0, 0)]
        return Padding2DLayer.__remove_padding(delta, padding_size)

    @staticmethod
    def __get_unified_padding_size(size):
        if isinstance(size, int):
            return (size, size), (size, size)
        elif isinstance(size, tuple):
            if isinstance(size[0], int):
                return (size[0], size[0]), (size[1], size[1])
            elif isinstance(size[0], tuple):
                return size

    @staticmethod
    def __add_padding(array, padding_sizes, mode):
        if mode == 'valid':
            return np.array(array)
        elif mode == 'zeros':
            return np.pad(array, padding_sizes, mode='constant', constant_values=0)
        elif mode == 'edge':
            return np.pad(array, padding_sizes, mode='edge')
        else:
            raise InvalidParameterException(f'Invalid padding mode: {mode}')

    @staticmethod
    def __remove_padding(array, paddings_size):
        slices = []
        for padding_size, dim in zip(paddings_size, array.shape):
            s = slice(padding_size[0], dim - padding_size[1])
            slices.append(s)
        return array[tuple(slices)]
