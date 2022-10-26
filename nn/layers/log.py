from abc import ABC

import numpy as np
import matplotlib.pyplot as plt

from .base import Layer


class BaseLogLayer(Layer, ABC):

    def __init__(self, title):
        super().__init__()
        self.title = title

    def get_output_shape(self):
        return self.input_shape

    def backpropagate(self, delta):
        return delta

    def show_matrix(self, matrix, slice_no=None):
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        plt.matshow(matrix)

        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                c = matrix[y, x]
                text = f'{c:.1f}' if isinstance(c, np.floating) else str(c)
                plt.text(x, y, text, va='center', ha='center')

        title = f'{self.title} (slice {slice_no})' if slice_no is not None else self.title
        plt.title(title)
        plt.show()


class Log2DLayer(BaseLogLayer):

    def __init__(self, title='Log'):
        super().__init__(title)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) in [2, 3]

    def propagate(self, x):
        is_multislice = x.ndim == 3

        if is_multislice:
            for slice_no in range(x.shape[-1]):
                self.show_matrix(x[..., slice_no], slice_no)
        else:
            self.show_matrix(x)

        return x


class Log1DLayer(BaseLogLayer):

    def __init__(self, title='Log'):
        super().__init__(title)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) in [1, 2]

    def propagate(self, x):
        is_multislice = x.ndim == 2

        if is_multislice:
            for slice_no in range(x.shape[-1]):
                self.show_matrix(x[..., slice_no], slice_no)
        else:
            self.show_matrix(x)

        return x
