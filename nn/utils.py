
import numpy as np

from .exceptions import InvalidParameterException


def apply_padding(array, axes_pad_width, mode):
    sym_pad_widths = [(val, val) for val in axes_pad_width]
    if mode == 'valid':
        return np.array(array)
    elif mode == 'same':
        return np.pad(array, sym_pad_widths, mode='constant', constant_values=0)
    elif mode == 'edge':
        return np.pad(array, sym_pad_widths, mode='edge')
    else:
        raise InvalidParameterException(f'Invalid padding mode: {mode}')
