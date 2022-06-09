# MIT License
# Copyright (c) 2022
# Erik Koynov
from typing import Sized

def calculate_padding_size(dimension_size: int, window_size: int, stride: int) -> int:
    """
    Calculate the needed padding to the right and lower ends of a 2-D array with an additional channel dim.
    The function is calculated by the following formula:
        pad = stride - pad%stride,  since:
       (a+x) mod b = 0 mod b where x=pad, a=(dimension_size-window_size) and b=stride
    The observation that led to creating this formula is that (dimension_size-window_size)%stride
    has to be an integer in order that the convolution covers the whole image.
    :param dimension_size:
    :param window_size:
    :param stride:
    :return:
    """

    if dimension_size==window_size:
        paddind_size = 0
    else:
        paddind_size = (dimension_size - window_size)
        paddind_size = paddind_size%stride
        if paddind_size>0:
            paddind_size = stride-paddind_size
    return paddind_size

def split_into_batches(dataset: Sized, batch_size):
    """Split a list into batches"""
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]
