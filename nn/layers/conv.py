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
        start = self.dilated_kernel_size // 2
        end = self.input_slice_size - start - 1
        diff = end - start + 1
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

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def propagate(self, x):
        image_sections = self.__get_sections(x)
        convoluted = self.__conv_sections(image_sections, self.kernels)
        return convoluted

    def backpropagate(self, delta):
        for slice_no in range(self.input_slices_count):
            slice_delta = delta[..., slice_no]

    def __conv_tensor(self, tensor, kernels):
        sections = self.__get_sections(tensor)
        flatten_kernels = kernels.reshape((len(kernels), -1))
        flatten_convoluted = sections @ flatten_kernels.T
        convoluted = flatten_convoluted.reshape((*self.output_slice_size, len(kernels)))

    def __conv_sections(self, sections, kernels):

        return convoluted

    def __get_sections(self, data):
        sections_count = np.prod(self.output_slice_size)
        kernel_count = np.prod(self.kernel_size)
        sections = np.zeros((sections_count, kernel_count * self.input_slices_count))

        centers0 = [self.dilated_kernel_size[0] // 2 + i * self.stride[0] for i in range(self.output_slice_size[0])]
        centers1 = [self.dilated_kernel_size[1] // 2 + i * self.stride[1] for i in range(self.output_slice_size[1])]

        linear_index = 0
        for center0 in centers0:
            for center1 in centers1:
                sections[linear_index] = self.__get_single_image_section(data, (center0, center1))
                linear_index += 1

        return sections

    def __get_single_image_section(self, data, pos):
        section = np.zeros((np.prod(self.kernel_size), self.input_slices_count))
        kernel_half = self.kernel_size // 2

        positions0 = [pos[0] + self.dilation[0] * i for i in range(-kernel_half[0], kernel_half[0]+1)]
        positions1 = [pos[1] + self.dilation[1] * i for i in range(-kernel_half[1], kernel_half[1] + 1)]

        linear_index = 0
        for pos0 in positions0:
            for pos1 in positions1:
                section[linear_index] = data[pos0, pos1, :]
                linear_index += 1

        return section.flatten()
