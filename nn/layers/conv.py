from abc import ABC

import numpy as np

from .base import Layer
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
        return (self.input_slice_size - self.dilated_kernel_size + 1) // self.stride

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

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def propagate(self, x):
        slices = [self.__conv_with_kernel(x, kernel) for kernel in self.kernels]
        slices = np.array(slices)
        output = np.moveaxis(slices, [0, 1, 2], [2, 0, 1])
        return output

    def backpropagate(self, delta):
        raise NotImplementedError

    def __conv_with_kernel(self, x, kernel):
        output = np.zeros(self.output_slice_size, dtype=x.dtype)
        positions0 = [self.dilated_kernel_size[0] // 2 + i * self.stride[0] for i in range(self.output_slice_size[0])]
        positions1 = [self.dilated_kernel_size[1] // 2 + i * self.stride[1] for i in range(self.output_slice_size[1])]
        for index0, pos0 in enumerate(positions0):
            for index1, pos1 in enumerate(positions1):
                output[index0][index1] = self.__conv(x, kernel, (pos0, pos1))
        return output

    def __conv(self, x, kernel, pos):
        kernel_half = self.kernel_size // 2
        result = 0
        x_indexes0 = [pos[0] + self.dilation[0] * i for i in range(-kernel_half[0], kernel_half[0]+1)]
        x_indexes1 = [pos[1] + self.dilation[1] * i for i in range(-kernel_half[1], kernel_half[1] + 1)]
        for kernel_index0, x_index0 in enumerate(x_indexes0):
            for kernel_index1, x_index1 in enumerate(x_indexes1):
                for slice_index in range(self.input_slices_count):
                    result += x[x_index0][x_index1][slice_index] * kernel[kernel_index0][kernel_index1][slice_index]
        return result
