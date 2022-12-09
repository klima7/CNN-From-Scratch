import numpy as np

from .base import Layer
from ..exceptions import InvalidParameterException


class Padding2DLayer(Layer):

    def __init__(self, padding_size, mode='zeros'):
        super().__init__()
        self.mode = mode
        self.padding_size = self._get_unified_padding_size(padding_size)

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
        else:
            return self.input_slice_size + np.array([sum(self.padding_size[0]), sum(self.padding_size[1])])

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def get_output_shape(self):
        return tuple((*self.output_slice_size, self.slices_count))

    def propagate(self, x):
        return self._apply_padding(x, [*self.padding_size, (0, 0)])

    def backpropagate(self, delta):
        return self._remove_padding(delta, [*self.padding_size, (0, 0)])

    @staticmethod
    def _get_unified_padding_size(size):
        if isinstance(size, int):
            return (size, size), (size, size)
        elif isinstance(size, tuple):
            if isinstance(size[0], int):
                return (size[0], size[0]), (size[1], size[1])
            elif isinstance(size[0], tuple):
                return size

    def _apply_padding(self, array, axes_pad_width):
        if self.mode == 'valid':
            return np.array(array)
        elif self.mode == 'zeros':
            return np.pad(array, axes_pad_width, mode='constant', constant_values=0)
        elif self.mode == 'edge':
            return np.pad(array, axes_pad_width, mode='edge')
        else:
            raise InvalidParameterException(f'Invalid padding mode: {self.mode}')

    @staticmethod
    def _remove_padding(array, paddings):
        reversed_padding = []
        for padding, dim in zip(paddings, array.shape):
            reversed_padding.append(slice(padding[0], dim - padding[1]))
        return array[tuple(reversed_padding)]
