from abc import ABC

import numpy as np

from .base import Layer
from ..convolution import convolve
from ..exceptions import InvalidParameterException
from ..initializers import RandomUniformInitializer


class BaseConvLayer(Layer, ABC):

    def __init__(self, filters_count, kernel_size, stride, dilation, initializer):
        super().__init__()

        if not np.all(np.mod(kernel_size, 2) != 0):
            raise InvalidParameterException('kernel size must be an odd number in each dimension')

        self.filters_count = filters_count
        self.kernel_size = np.array(kernel_size)
        self.stride = np.array(stride)
        self.dilation = np.array(dilation)
        self.initializer = initializer
        self.kernels = None

    @property
    def dilated_kernel_size(self):
        return (self.kernel_size - 1) * self.dilation + 1

    @property
    def input_slices_count(self):
        return self.input_shape[-1]

    @property
    def input_slice_size(self):
        return self.input_shape[:-1]

    @property
    def output_slices_count(self):
        return self.filters_count

    @property
    def output_slice_size(self):
        diff = self.input_slice_size - 2 * (self.dilated_kernel_size // 2)
        return np.ceil(diff / self.stride).astype(int)

    def get_output_shape(self):
        return tuple((*self.output_slice_size, self.output_slices_count))

    def initialize(self):
        kwargs = {
            'fan_in': np.prod(self.kernel_size) * self.input_slices_count,
            'fan_out': np.prod(self.kernel_size) * self.output_slices_count
        }
        normalized_kernel_size = np.array(self.kernel_size).flatten()     # 3 -> (3,)
        shape = (*normalized_kernel_size, self.input_slices_count)
        kernels = [self.initializer(shape, **kwargs) for _ in range(self.output_slices_count)]
        self.kernels = np.array(kernels)


class Conv1DLayer(BaseConvLayer):

    def __init__(self, filters_count, kernel_size, stride=1, dilation=1,
                 initializer=RandomUniformInitializer()):
        super().__init__(filters_count, kernel_size, stride, dilation, initializer)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 2

    def propagate(self, x):
        slices = [self.__conv_with_kernel(x, kernel) for kernel in self.kernels]
        slices = np.array(slices)
        output = np.moveaxis(slices, [0, 1], [1, 0])
        return output

    def backpropagate(self, delta):
        raise NotImplementedError

    def __conv_with_kernel(self, x, kernel):
        output = np.zeros(self.output_slice_size, dtype=x.dtype)
        positions = [self.dilated_kernel_size // 2 + i * self.stride for i in range(self.output_slice_size[0])]
        for index, pos in enumerate(positions):
            output[index] = self.__conv(x, kernel, pos)
        return output

    def __conv(self, x, kernel, pos):
        kernel_half = self.kernel_size // 2
        result = 0
        x_indexes = [pos + self.dilation * i for i in range(-kernel_half, kernel_half+1)]
        for kernel_index, x_index in enumerate(x_indexes):
            for slice_index in range(self.input_slices_count):
                result += x[x_index][slice_index] * kernel[kernel_index][slice_index]
        return result


class Conv2DLayer(BaseConvLayer):

    def __init__(self, filters_count, kernel_size, stride=(1, 1), dilation=(1, 1),
                 initializer=RandomUniformInitializer()):
        super().__init__(filters_count, kernel_size, stride, dilation, initializer)
        self.x = None

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def propagate(self, x):
        self.x = x
        return convolve(x, self.kernels, self.stride, self.dilation, full=False)

    def backpropagate(self, delta):
        new_delta = self.__get_new_delta(delta)
        self.__update_weights(delta)
        return new_delta

    def __get_new_delta(self, delta):
        kernels = np.transpose(self.kernels, (3, 1, 2, 0))
        new_delta = convolve(delta, kernels, self.stride, self.dilation, full=True)
        return new_delta

    def __update_weights(self, delta):
        updates = []
        for slice_no in range(self.output_slices_count):
            slice_delta = delta[..., slice_no, np.newaxis]
            kernels = np.array([slice_delta for _ in range(self.input_slices_count)])
            kernels = np.transpose(kernels, (3, 1, 2, 0))
            update = convolve(self.x, kernels, self.stride, self.dilation, full=False)
            updates.append(update)

        updates = np.array(updates)
        self.kernels += self.nn.learning_rate * updates
