from abc import ABC, abstractmethod

import numpy as np

from .exceptions import InvalidShapeError, InvalidParameterException
from .initializers import RandomUniformInitializer
from .utils import apply_padding


class Layer(ABC):

    def __init__(self):
        self.nn = None
        self.prev_layer = None
        self.next_layer = None
        self.output_shape = None

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @property
    def input_shape(self):
        return self.prev_layer.output_shape if self.prev_layer else None

    def connect(self, nn, prev_layer, next_layer):
        self.nn = nn
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        if not self.is_input_shape_valid(self.input_shape):
            raise InvalidShapeError(f'Inferred layer input shape is invalid {self.input_shape}')

        self.output_shape = np.array(self.get_output_shape())
        self.initialize()

    def propagate_with_validation(self, x):
        if not np.array_equal(x.shape, self.input_shape):
            raise InvalidShapeError(f'Array with invalid shape passed to propagate method. Should be {self.input_shape}, but is {x.shape}')
        propagated_data = self.propagate(x)
        if not np.array_equal(propagated_data.shape, self.output_shape):
            raise InvalidShapeError(f'Array with invalid shape returned from propagate method. Should be {self.output_shape}, but is {propagated_data.shape}')
        return propagated_data

    def backpropagate_with_validation(self, delta):
        if not np.array_equal(delta.shape, self.output_shape):
            raise InvalidShapeError(f'Array with invalid shape passed to backpropagate method. Should be {self.output_shape}, but is {delta.shape}')
        prev_delta = self.backpropagate(delta)
        if not np.array_equal(prev_delta.shape, self.input_shape):
            raise InvalidShapeError(f'Backpropagate method returned array with invalid shape. Should be {self.input_shape}, but is {prev_delta.shape}')
        return prev_delta

    def initialize(self):
        pass

    @abstractmethod
    def is_input_shape_valid(self, input_shape):
        pass

    @abstractmethod
    def get_output_shape(self):
        pass

    @abstractmethod
    def propagate(self, x):
        pass

    @abstractmethod
    def backpropagate(self, delta):
        pass


class InputLayer(Layer):

    def __init__(self, shape):
        super().__init__()
        self.shape = np.array(shape)

    def __repr__(self):
        return f'{self.__class__.__name__}(shape: {self.shape})'

    @property
    def input_shape(self):
        return self.output_shape

    def is_input_shape_valid(self, input_shape):
        return input_shape == self.input_shape

    def get_output_shape(self):
        return self.shape

    def propagate(self, x):
        return x

    def backpropagate(self, delta):
        return delta


class DenseLayer(Layer):

    def __init__(self, neurons_count, initializer=RandomUniformInitializer()):
        super().__init__()
        self.neurons_count = neurons_count
        self.initializer = initializer
        self.input_data = None
        self.weights = None

    def __repr__(self):
        return f'{self.__class__.__name__}(neurons_count: {self.neurons_count}, initializer: {self.initializer})'

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 1

    def get_output_shape(self):
        return tuple((self.neurons_count,))

    def initialize(self):
        shape = (self.neurons_count, self.input_shape[0])
        kwargs = {
            'fan_in': self.input_shape[0],
            'fan_out': self.output_shape[0]
        }
        self.weights = self.initializer(shape, **kwargs)

    def propagate(self, x):
        self.input_data = x
        return x @ self.weights.T

    def backpropagate(self, delta):
        next_delta = self.__get_next_delta(delta)
        self.__adjust_weights(delta)
        return next_delta.flatten()

    def __adjust_weights(self, delta):
        weights_delta = self.nn.learning_rate * delta.reshape(-1, 1) @ self.input_data.reshape(1, -1)
        self.weights += weights_delta

    def __get_next_delta(self, delta):
        next_delta = delta @ self.weights
        return next_delta


class ActivationLayer(Layer):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.state = None

    def __repr__(self):
        return f'{self.__class__.__name__}(activation: {self.activation})'

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return self.input_shape

    def propagate(self, x):
        self.state = self.activation.call(x)
        return self.state

    def backpropagate(self, delta):
        deriv = self.activation.deriv(self.state)
        return delta * deriv


class FlattenLayer(Layer):

    def __init__(self):
        super().__init__()

    def is_input_shape_valid(self, input_shape):
        return True

    def get_output_shape(self):
        return tuple((np.prod(self.input_shape),))

    def propagate(self, x):
        return x.flatten()

    def backpropagate(self, delta):
        return delta.reshape(*self.input_shape)


class BaseConvLayer(Layer, ABC):

    def __init__(self, filters_count, kernel_size, padding, stride, dilation, initializer):
        super().__init__()

        if not np.all(np.mod(kernel_size, 2) != 0):
            raise InvalidParameterException('kernel size must be an odd number in each dimension')

        self.filters_count = filters_count
        self.kernel_size = np.array(kernel_size)
        self.padding = padding
        self.stride = np.array(stride)
        self.dilation = np.array(dilation)
        self.initializer = initializer
        self.kernels = None

    def __repr__(self):
        return f'{self.__class__.__name__}(filters_count: {self.filters_count}, kernel_size: {self.kernel_size}, '\
               f'stride: {self.stride}, dilation: {self.dilation}, padding: {self.padding}, initializer: {self.initializer})'

    @property
    def dilated_kernel_size(self):
        return (self.kernel_size - 1) * self.dilation + 1

    @property
    def padding_size(self):
        return self.dilated_kernel_size // 2 if self.padding != 'valid' else np.zeros_like(self.dilated_kernel_size)

    @property
    def input_slices_count(self):
        return self.input_shape[-1]

    @property
    def input_slice_size(self):
        return self.input_shape[:-1]

    @property
    def adjusted_input_slice_size(self):
        return self.input_slice_size + 2*self.padding_size

    @property
    def output_slices_count(self):
        return self.filters_count

    @property
    def output_slice_size(self):
        return (self.adjusted_input_slice_size - self.dilated_kernel_size + 1) // self.stride

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

    def __init__(self, filters_count, kernel_size, padding='valid', stride=1, dilation=1,
                 initializer=RandomUniformInitializer()):
        super().__init__(filters_count, kernel_size, padding, stride, dilation, initializer)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 2

    def propagate(self, x):
        x_padded = apply_padding(x, [self.padding_size, 0], mode=self.padding)
        slices = [self.__conv_with_kernel(x_padded, kernel) for kernel in self.kernels]
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

    def __init__(self, filters_count, kernel_size, padding='valid', stride=(1, 1), dilation=(1, 1),
                 initializer=RandomUniformInitializer()):
        super().__init__(filters_count, kernel_size, padding, stride, dilation, initializer)

    def is_input_shape_valid(self, input_shape):
        return len(input_shape) == 3

    def propagate(self, x):
        x_padded = apply_padding(x, [*self.padding_size, 0], mode=self.padding)
        slices = [self.__conv_with_kernel(x_padded, kernel) for kernel in self.kernels]
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
