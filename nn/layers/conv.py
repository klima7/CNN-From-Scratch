import numpy as np

from .base import Layer
from ..convolution import convolve, get_convolution_output_size, dilate
from ..exceptions import InvalidParameterException, InvalidShapeException
from ..initializers import GlorotUniformInitialization


class Conv2DLayer(Layer):

    def __init__(self, filters_count, kernel_size, stride=(1, 1), initializer=None):
        super().__init__()

        if not np.all(np.mod(kernel_size, 2) != 0):
            raise InvalidParameterException('kernel size must be an odd number in each dimension')

        self.filters_count = filters_count
        self.kernel_size = np.array(kernel_size)
        self.stride = np.array(stride)
        self.initializer = initializer or GlorotUniformInitialization()
        self.kernels = None
        self.__x = None

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
        return get_convolution_output_size(
            data_size=tuple(self.input_slice_size),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=np.array([1, 1]),
            full_conv=False
        )

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
        self.params_count = self.kernels.size

    def validate_input_shape(self):
        if len(self.input_shape) != 3:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 3D, but is {self.input_shape}')

    def propagate(self, x):
        self.__x = x
        return convolve(
            data=x,
            kernels=self.kernels,
            stride=self.stride,
            dilation=np.array([1, 1]),
            full_conv=False
        )

    def backpropagate(self, delta):
        new_delta = self.__get_new_delta(delta)
        self.__update_weights(delta)
        return new_delta

    def __get_new_delta(self, delta):
        kernels = np.transpose(self.kernels, (3, 1, 2, 0))
        delta = dilate(delta, self.stride)
        new_delta = convolve(
            data=delta,
            kernels=kernels,
            stride=np.array([1, 1]),
            dilation=np.array([1, 1]),
            full_conv=True
        )
        return new_delta

    def __update_weights(self, delta):
        updates = []
        for slice_no in range(self.output_slices_count):
            slice_delta = delta[..., slice_no, np.newaxis]
            slice_delta = dilate(slice_delta, self.stride)
            kernels = np.array([slice_delta for _ in range(self.input_slices_count)])
            kernels = np.transpose(kernels, (3, 1, 2, 0))
            update = convolve(
                data=self.__x,
                kernels=kernels,
                stride=np.array([1, 1]),
                dilation=np.array([1, 1]),
                full_conv=False
            )
            updates.append(update)

        updates = np.array(updates)
        self.kernels += self.nn.learning_rate * updates
