# MIT License
# Copyright (c) 2022
# Erik Koynov
from typing import List, Tuple
import numpy as np

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

def split_into_batches(dataset: List[Tuple[np.ndarray]], batch_size)->List[np.array]:
    """Split a list into batches"""
    for i in range(0, len(dataset), batch_size):
        batch = list(zip(*dataset[i:i + batch_size])) # [(in11, in12,in13...),(in21,in22,...)]
        yield collate_fn(batch) # the patches for the different inputs will be separated

def collate_fn(batched_inputs: List[Tuple[np.array]])-> List[np.array]:
    collated_inputs = []
    for inputs in batched_inputs: # inputs is a Tuple of np.array (all inputs of one input category)
        # assuming the inputs do NOT have a batch dimension!
        collated_inputs.append(np.vstack([inputs]))
    return collated_inputs


