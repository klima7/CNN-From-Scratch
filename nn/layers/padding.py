from abc import ABC

import numpy as np

from .base import Layer
from ..exceptions import InvalidParameterException


class BasePaddingLayer(Layer, ABC):

    def __init__(self, mode, padding_size):
        super().__init__()
        self.mode = mode
        self.padding_size = np.array(padding_size)

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
            return self.input_slice_size + 2*self.padding_size

    def get_output_shape(self):
        return tuple((*self.output_slice_size, self.slices_count))

    def _apply_padding(self, array, axes_pad_width):
        sym_pad_widths = [(val, val) for val in axes_pad_width]
        if self.mode == 'valid':
            return np.array(array)
        elif self.mode == 'zeros':
            return np.pad(array, sym_pad_widths, mode='constant', constant_values=0)
        elif self.mode == 'edge':
            return np.pad(array, sym_pad_widths, mode='edge')
        else:
            raise InvalidParameterException(f'Invalid padding mode: {self.mode}')

    @staticmethod
    def _remove_padding(array, axes_pad_width):
        reversed_padding = []
        for pad_width, dim in zip(axes_pad_width, array.shape):
            reversed_padding.append(slice(pad_width, dim - pad_width))
        return array[tuple(reversed_padding)]


class Padding1DLayer(BasePaddingLayer):

    def __init__(self, mode, padding_size=1):
        super().__init__(mode, padding_size)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 2

    def propagate(self, x):
        return self._apply_padding(x, [self.padding_size, 0])

    def backpropagate(self, delta):
        return self._remove_padding(delta, [self.padding_size, 0])


class Padding2DLayer(BasePaddingLayer):
    def __init__(self, mode, padding_size=(1, 1)):
        super().__init__(mode, padding_size)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def propagate(self, x):
        return self._apply_padding(x, [*self.padding_size, 0])

    def backpropagate(self, delta):
        return self._remove_padding(delta, [*self.padding_size, 0])
