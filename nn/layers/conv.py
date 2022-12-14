import numpy as np

from .base import Layer
from ..convolution import convolve, get_convolution_output_size, get_dilated_kernel_size, dilate
from ..exceptions import InvalidParameterException, InvalidShapeException
from ..initializers import get_initializer


class Conv2DLayer(Layer):

    def __init__(self, filters_count, kernel_size, stride=(1, 1), initializer='glorot'):
        super().__init__()

        kernel_size = np.broadcast_to(kernel_size, (2,))
        stride = np.broadcast_to(stride, (2,))

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            msg = f'kernel size must be an odd, but it {kernel_size}'
            raise InvalidParameterException(msg)

        self.filters_count = filters_count
        self.kernel_size = np.array(kernel_size)
        self.stride = np.array(stride)
        self.initializer = get_initializer(initializer)
        self.kernels = None
        self.__x = None

    @property
    def input_slices_count(self):
        return self.input_shape[-1]

    @property
    def input_slice_size(self):
        return self.input_shape[:-1]

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
        return tuple((*self.output_slice_size, self.filters_count))

    def initialize(self):
        initializer_kwargs = {
            'fan_in': np.prod(self.kernel_size) * self.input_slices_count,
            'fan_out': np.prod(self.kernel_size) * self.filters_count
        }
        kernels_shape = (self.filters_count, *self.kernel_size, self.input_slices_count)
        self.kernels = self.initializer(kernels_shape, **initializer_kwargs)
        self.params_count = self.kernels.size

    def validate_input_shape(self):
        if len(self.input_shape) != 3:
            raise InvalidShapeException(f'{self.__class__.__name__} input must be 3D, but is {self.input_shape}')

        expected_input_size = get_convolution_output_size(
            data_size=tuple(get_dilated_kernel_size(self.output_slice_size, self.stride)),
            kernel_size=self.kernel_size,
            stride=np.array([1, 1]),
            dilation=np.array([1, 1]),
            full_conv=True
        )

        if not np.array_equal(expected_input_size, self.input_shape[:2]):
            msg = f'{self.__class__.__name__} parameters does not match to input data shape {self.input_shape}'
            raise InvalidShapeException(msg)

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
        kernels = np.flip(self.kernels, axis=(1, 2))
        kernels = np.transpose(kernels, (3, 1, 2, 0))
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
        updates = np.zeros_like(self.kernels)
        for slice_no in range(self.filters_count):
            kernel = delta[..., slice_no, np.newaxis]
            kernel = dilate(kernel, self.stride)
            kernels = np.repeat(kernel[np.newaxis, ...], self.input_slices_count, axis=0)
            kernels = np.transpose(kernels, (3, 1, 2, 0))
            update = convolve(
                data=self.__x,
                kernels=kernels,
                stride=np.array([1, 1]),
                dilation=np.array([1, 1]),
                full_conv=False
            )
            updates[slice_no, ...] = update

        self.kernels -= self.nn.learning_rate * updates
