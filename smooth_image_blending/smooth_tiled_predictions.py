# MIT License
# Copyright (c) 2017 Vooban Inc.
# Coded by: Guillaume Chevalier, Fork by Erik Koynov.
# Source to original code and license:
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches
#     https://github.com/Vooban/Smoothly-Blend-Image-Patches/blob/master/LICENSE


"""Do smooth predictions on an image from tiled prediction patches."""


import numpy as np
import scipy.signal
from tqdm import tqdm
from typing import List, Callable, Iterable
from dataclasses import dataclass
from patchify import patchify, unpatchify
from .utils import split_into_batches
import gc
from .padding import Padding



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False






def spline_window(window_size, power=2, signal: Callable = scipy.signal.windows.triang):
    """
    Squared spline (power=2) window function:
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(signal(window_size))) ** power)/2

    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - ((2*(signal(window_size) - 1)) ** power)/2

    wind_inner[:intersection] = 0

    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer

    wind = wind / np.average(wind)

    return wind[:, None]



cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = spline_window(window_size, power)
        wind = wind*wind.T # new shape = window_size x window_size
        cached_2d_windows[key] = wind
    print("Window shape: ", wind.shape)
    return wind





def _rotate_mirror_do(im: np.array)->List[np.array]:
    """
    Duplicate an np array (image) of shape (x, y, nb_channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    """
    merges a list of 8 np arrays (images) of shape (x, y, nb_channels) generated
    from the `_rotate_mirror_do` function. Each images might have changed and
    merging them implies to rotated them back in order and average things out.

    It is the D_4 (D4) Dihedral group:
    https://en.wikipedia.org/wiki/Dihedral_group
    """
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def filter_predictions(patches: np.array, window_size: int):
    """
    Create tiled overlapping patches.

    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )

    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D: np.array = _window_2D(window_size=window_size, power=2)

    gc.collect()
    print("PATCHES: ",patches.shape)
    print("SLINE: ",WINDOW_SPLINE_2D[:,:,None].shape)
    if len(patches.shape)>3:
        WINDOW_SPLINE_2D = WINDOW_SPLINE_2D[..., None]
    patches = WINDOW_SPLINE_2D*patches
    gc.collect()

    return patches


def apply_averaging_unpatchify(patches: np.array, window_size: tuple, stride: int, reconstructed_shape: tuple):
    """
    Merge tiled overlapping patches smoothly.
    reconstructed_shape : shape of the padded image before patchify.
    patches: n_patches x (patch_dimensions)
    """
    subdivisions = np.ceil(window_size[0]/stride)
    height = reconstructed_shape[0]
    width = reconstructed_shape[1]

    window_height = window_size[0]
    window_width = window_size[1]

    y = np.zeros(reconstructed_shape)

    patch_idx = 0
    for i in range(0, width-window_height+1, stride):

        for j in range(0, height-window_width+1, stride):

            windowed_patch = patches[patch_idx]
            y[i:i+window_height, j:j+window_width] += windowed_patch
            patch_idx += 1

    return y / (subdivisions ** 2)

@dataclass
class TestTimeAugmentations:
    test_time_augmentations: List[Callable]

    def __call__(self, img: np.array):
        augmented = []
        for aumentation in self.test_time_augmentations:
            augmented.append(aumentation(img))
        return augmented

def predict_on_patches(patches: Iterable,
                       prediction_fn: Callable,
                       batch_size: int = 1) -> np.array:
    predictions = []
    for batch in split_into_batches(patches, batch_size):
        predictions.append(prediction_fn(batch).detach().to('cpu').numpy())
    return np.vstack(predictions)

def apply_patchify(image, window_size: tuple, stride: int):
    if len(image.shape) == 2:
        window_size = window_size[:-1]

    patches = patchify(image, window_size, stride)
    #  reshape into : n_batches x (window shape)
    patches = patches.reshape((-1,)+window_size)
    return patches


def predict_img_with_smooth_windowing(input_img,
                                      window_size,
                                      stride,
                                      padding: Padding,
                                      prediction_fn: Callable,
                                      batch_size = 1):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    
    Algorithm:
        1. tile image with overlap
        2. apply test-time augmentation
        3. make predictions on all patches (all their augmentations)
        4. apply filtering to each patch
        5. reconstruct the whole image from the processed patches
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    # 1. tile image with overlap
    pad: np.array = padding.fit_transform(input_img, window_size, stride)
    print("mean after padding: ", pad.mean())
    # 2. apply test-time augmentation
    pads = _rotate_mirror_do(pad)
    print("Means after mirroring: ", [p.mean() for p in pads])
    # For every rotation:
    for i, pad in tqdm(enumerate(pads)):
        shape = pad.shape
        reconstruction_shape = (shape[0], shape[1])
        # patchify image
        patches: np.array = apply_patchify(pad, (window_size, window_size, shape[-1]), stride)

        print("Means after patchify: ", [p.mean() for p in patches])

        # 3. make predictions on all patches (all their augmentations)
        patches = predict_on_patches(patches, prediction_fn, batch_size=batch_size)
        if len(patches.shape) > 3:
            output_channels = patches.shape[-1]
            reconstruction_shape += (output_channels,)
        print(f"Shape after prediction: {patches.shape}")
        print("Means after prediction: ", [p.mean() for p in patches])

        # 4. apply filtering to each patch
        patches = filter_predictions(patches, window_size)

        print("Means after filtering: ", [p.mean() for p in patches])

        # 5. reconstruct the whole image from the processed patches
        one_padded_result = apply_averaging_unpatchify(patches,
                                                       (window_size, window_size, shape[-1]),
                                                       stride,
                                                       reconstruction_shape)
        print(f"Shape after averaging: {one_padded_result.shape}")
        print("Mean after averaging: ", one_padded_result.mean())

        pads[i] = one_padded_result


    # Merge after rotations:
    padded_results = _rotate_mirror_undo(pads)

    prediction = padding.inverse_transform(padded_results)



    if PLOT_PROGRESS:
        plt.imshow(prediction)
        plt.title("Smoothly Merged Patches that were Tiled Tighter")
        plt.show()
    return prediction

