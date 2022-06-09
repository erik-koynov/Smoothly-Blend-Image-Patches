# MIT License
# Copyright (c) 2022
# Erik Koynov

import torch
from typing import Union
import numpy as np
from .utils import calculate_padding_size
from abc import ABC, abstractmethod
from math import ceil

class PaddingNotFitException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

class Padding(ABC):
    def __init__(self, mode: str = 'reflect'):
        self.mode = mode
        self._called = False

    @abstractmethod
    def fit_transform(self, *args, **kwargs)->np.array:
        """pad the image"""

    @abstractmethod
    def inverse_transform(self, *args, **kwargs)->np.array:
        """remove padding from the image"""

class AllAroundPadding(Padding):
    """
    Padding from all sides. The smaller the stride the larger the pad in order to mitigate the differences
    between the representation of the pixels in the middle and at the edges of the image.
    """
    def __init__(self,
                 mode: str = 'reflect',
                 pad_repeat: int = 1):
        super().__init__(mode=mode)
        self.pad_repeat = pad_repeat

    def fit_transform(self,
                      img: Union[torch.Tensor, np.array],
                      window_size: int,
                      stride: int) -> np.array:
        padded, self.aug_x, self.aug_y = AllAroundPadding.pad_img(img,
                                                                  window_size,
                                                                  stride,
                                                                  return_pad_size=True,
                                                                  pad_repeat=self.pad_repeat)
        self._called = True
        return padded

    def inverse_transform(self, img: Union[torch.Tensor, np.array]):
        return AllAroundPadding.unpad_img(img, self.aug_x, self.aug_y)

    @staticmethod
    def pad_img(img: Union[torch.Tensor, np.array],
                      window_size: int,
                      stride: int,
                pad_repeat: int = 2,
                return_pad_size: bool = False) -> np.array:
        subdivisions = ceil(window_size/float(stride))
        print(f"subdivisions: {subdivisions}")
        aug = int(round(window_size * (1 - 1.0 / subdivisions)))
        aug_x = pad_repeat*aug
        aug_y = pad_repeat*aug
        if len(img.shape) == 3:
            more_borders = ((aug_y, aug_y), (aug_x, aug_x), (0, 0))
        if len(img.shape) == 2:
            more_borders = ((aug_y, aug_y), (aug_x, aug_x))
        padded = np.pad(img, pad_width=more_borders, mode='reflect')
        if return_pad_size:
            return padded, aug_x, aug_y
        return padded

    @staticmethod
    def unpad_img(img: Union[torch.Tensor, np.array],
                  aug_x: int,
                  aug_y: int):
        """
        Undo what's done in the `_pad_img` function.
        Image is an np array of shape (x, y, nb_channels).
        """
        ret = img[
              aug_y:-aug_y,
              aug_x:-aug_x,
              ]
        # gc.collect()
        return ret


class LeftDownPadding(Padding):
    """
    Set the minimum padding needed (to the right and bottom ends) to the image so that a convolution operation
    can be applied over the whole image. Can be used in some settings of the smooth tiled prediction e.g.
    when the smoothing is applied by averaging and there is not the risk that at the unpadded edges of the
    image some unwanted effects may arise due to some polynomial filtering scheme.
    """
    def __init__(self,
                 mode: str = 'reflect'):
        super().__init__(mode=mode)


    def fit_transform(self,
            img: Union[torch.Tensor, np.array],
            window_size: int,
            stride: int,
            ):
        padded, self._aug_x, self._aug_y = LeftDownPadding.pad_img(img, window_size, stride, self.mode, return_pad_size=True)
        self._called = True
        return padded

    def inverse_transform(self,
                   img: Union[torch.Tensor, np.array]):
        return LeftDownPadding.unpad_img(img, self.aug_x, self.aug_y)

    @staticmethod
    def pad_img(img: Union[torch.Tensor, np.array],
                 window_size: int,
                 stride: int,
                 mode: str = "reflect",
                 return_pad_size: bool = False):
        """
        Add borders to img for a 'reflect' border pattern according to "window_size" and
        "stride". The padding is appended to the right and lower sides of the image.
        Image is an np array of shape (x, y, nb_channels).
        """
        aug_y = calculate_padding_size(img.shape[0], window_size, stride)
        aug_x = calculate_padding_size(img.shape[1], window_size, stride)

        print(f"Augmentation on Y: {aug_y}")
        print(f"Augmentation on X: {aug_x}")
        ret = np.pad(img, pad_width=((0, aug_y), (0, aug_x), (0,0)), mode=mode)
        if return_pad_size:
            return ret, aug_x, aug_y
        else:
            return ret

    @staticmethod
    def unpad_img(img: Union[torch.Tensor, np.array],
                   aug_x: int,
                   aug_y: int):
        """
        Undo what's done in the `_pad_img` function.
        Image is an np array of shape (x, y, nb_channels).
        """
        ret = img[
            0:-aug_y,
            0:-aug_x,
            :
        ]
        # gc.collect()
        return ret

    @property
    def aug_x(self):
        if self._called:
            return self._aug_x
        else:
            raise PaddingNotFitException("Cannot access self.aug_x before calling self.fit_transform!")

    @aug_x.setter
    def aug_x(self, aug_x):
        self._aug_x = aug_x

    @property
    def aug_y(self):
        if self._called:
            return self._aug_y
        else:
            raise PaddingNotFitException("Cannot access self.aug_x before calling self.fit_transform!")

    @aug_y.setter
    def aug_y(self, aug_y):
        self._aug_y = aug_y
