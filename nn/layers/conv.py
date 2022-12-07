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
        return Conv2DLayer.__conv_tensor(x, self.kernels, self.stride, self.dilation)

    def backpropagate(self, delta):
        updates = []
        for slice_no in range(self.output_slices_count):
            slice_delta = delta[..., slice_no, np.newaxis]
            kernels = np.array([slice_delta for _ in range(self.input_slices_count)])
            update = Conv2DLayer.__conv_tensor(self.x, kernels, self.stride, self.dilation)
            updates.append(update)
        updates = np.array(updates)
        print(updates.shape)

        kernels = np.transpose(self.kernels, (3, 1, 2, 0))
        new_delta = Conv2DLayer.__conv_tensor(delta, kernels, self.stride, self.dilation, full=True)
        return new_delta

    @staticmethod
    def __conv_tensor(tensor, kernels, stride, dilation, full=False):
        kernel_size = np.array(kernels.shape[1:-1])
        output_slice_size = Conv2DLayer.__get_conv_output_size(tensor.shape[:-1], kernel_size, stride, dilation, full)

        sections = Conv2DLayer.__get_sections(tensor, kernel_size, stride, dilation, full)
        flatten_kernels = kernels.reshape((len(kernels), -1))
        flatten_convoluted = sections @ flatten_kernels.T
        convoluted = flatten_convoluted.reshape((*output_slice_size, len(kernels)))
        return convoluted

    def __conv_sections(self, sections, kernels):
        flatten_kernels = kernels.reshape((len(kernels), -1))
        flatten_convoluted = sections @ flatten_kernels.T
        convoluted = flatten_convoluted.reshape((*self.output_slice_size, len(kernels)))
        return convoluted

    @staticmethod
    def __get_sections(data, kernel_size, stride, dilation, full=False):
        dilated_kernel_size = Conv2DLayer.__get_dilated_kernel_size(kernel_size, dilation)
        output_slice_size = Conv2DLayer.__get_conv_output_size(data.shape[:-1], kernel_size, stride, dilation, full)

        sections_count = np.prod(output_slice_size)
        kernel_length = np.prod(kernel_size)
        sections = np.zeros((sections_count, kernel_length * data.shape[-1]))

        offset = -dilated_kernel_size+1 if full else (0, 0)

        anchors0 = [i * stride[0] + offset[0] for i in range(output_slice_size[0])]
        anchors1 = [i * stride[1] + offset[1] for i in range(output_slice_size[1])]

        linear_index = 0
        for center0 in anchors0:
            for center1 in anchors1:
                sections[linear_index] = Conv2DLayer.__get_single_image_section(data, (center0, center1), kernel_size, dilation)
                linear_index += 1

        return sections

    @staticmethod
    def __get_single_image_section(data, pos, kernel_size, dilation):
        section = np.zeros((np.prod(kernel_size), data.shape[-1]))

        positions0 = [pos[0] + dilation[0] * i for i in range(kernel_size[0])]
        positions1 = [pos[1] + dilation[1] * i for i in range(kernel_size[1])]

        linear_index = 0
        for pos0 in positions0:
            for pos1 in positions1:
                index_valid = 0 <= pos0 < data.shape[0] and 0 <= pos1 < data.shape[1]
                section[linear_index] = data[pos0, pos1, :] if index_valid else np.zeros((data.shape[-1],))
                linear_index += 1

        return section.flatten()

    @staticmethod
    def __get_conv_output_size(data_size, kernel_size, stride, dilation, full=False):
        dilated_kernel_size = Conv2DLayer.__get_dilated_kernel_size(kernel_size, dilation)
        if full:
            tmp = np.ceil((data_size + 2*kernel_size - 2 - 2*(dilated_kernel_size // 2)) / stride).astype(int)
            return tmp
        else:
            return np.ceil((data_size - 2*(dilated_kernel_size // 2)) / stride).astype(int)

    @staticmethod
    def __get_dilated_kernel_size(kernel_size, dilation):
        return (kernel_size - 1) * dilation + 1
