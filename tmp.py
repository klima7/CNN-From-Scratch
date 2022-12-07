import numpy as np


def __get_dilated_kernel_size(kernel_size, dilation):
    return (kernel_size - 1) * dilation + 1


def __get_conv_output_size(data_size, kernel_size, stride, dilation, full=False):
    dilated_kernel_size = __get_dilated_kernel_size(kernel_size, dilation)
    if full:
        return np.ceil((data_size + 2 * kernel_size - 2 - 2 * (dilated_kernel_size // 2)) / stride).astype(int)
    else:
        return np.ceil((data_size - 2 * (dilated_kernel_size // 2)) / stride).astype(int)
