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
from smooth_image_blending.utils import split_into_batches
import gc
from smooth_image_blending.padding import Padding, AllAroundPadding
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import torch
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    PLOT_PROGRESS = True
    # See end of file for the rest of the __main__.
else:
    PLOT_PROGRESS = False


def ground_truth_classes(y_true: np.ndarray, background_lbl = 0):
    unique_labels = np.unique(y_true)
    for lbl in unique_labels:
        if lbl == background_lbl:
            continue
        yield lbl


def get_max_overlap_lbl_generator(a, b, retrieve_overlap=False, retrieve_list = False):
    for lbl in ground_truth_classes(a):
        lbl_mask = (a == lbl)
        print(np.unique(b[lbl_mask], return_counts=True))
        unique_pred_lbls, pred_lbls_counts = np.unique(b[lbl_mask], return_counts=True)

        print(unique_pred_lbls)

        if retrieve_overlap:
            if not retrieve_list:
                predicted_lbl = int(unique_pred_lbls[pred_lbls_counts.argmax()])
                if predicted_lbl == 0:
                    overlap = 0.
                else:
                    overlap = (lbl_mask * (b == predicted_lbl)).sum() / lbl_mask.sum()
                yield lbl, predicted_lbl, overlap
            else: # have to retrieve a list of all overlapping labels and their overlaps (because there might be multiple tiny dots as lone labels that will get missed
                unique_pred_lbls, pred_lbls_counts = list(
                    zip(*sorted(zip(unique_pred_lbls, pred_lbls_counts), key=lambda x: x[1])))

                predicted_lbls, overlaps = [],[]
                for predicted_lbl in unique_pred_lbls:
                    predicted_lbls.append(predicted_lbl)

                    if predicted_lbl == 0:
                        overlaps.append(0.)
                    else:
                        overlaps.append((lbl_mask * (b == predicted_lbl)).sum() / lbl_mask.sum())
                yield lbl, predicted_lbls, overlaps
        else:
            yield lbl, int(unique_pred_lbls[pred_lbls_counts.argmax()])


def max_overlap_labels(a, b, inverted=False, retrieve_list=False):
    if len(np.unique(a))==1 and np.unique(a)[0]==0:
        return {}
    overlap_dict = {}
    for lbl, predicted_lbl, overlap in get_max_overlap_lbl_generator(a, b,
                                                                     retrieve_overlap=True,
                                                                     retrieve_list=retrieve_list):
        if inverted:
            overlap_dict[predicted_lbl] = [lbl, overlap]
        else:
            overlap_dict[lbl] = [predicted_lbl, overlap]
    return overlap_dict

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

def identity_signal(window_size,_):
    return np.ones((window_size,1))

cached_2d_windows = dict()
def _window_2D(window_size, power=2, spline_window_fn: Callable = spline_window):
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
        wind = spline_window_fn(window_size, power)
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


def _rotate_mirror_undo(im_mirrs)->np.ndarray:
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


def filter_predictions(patches: List[np.array],
                       window_size: int,
                       exclude_from_filtering: List[bool] = None,
                       spline_window_fn: Callable = spline_window):
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
    WINDOW_SPLINE_2D: np.array = _window_2D(window_size=window_size,
                                            power=2,
                                            spline_window_fn=spline_window_fn)
    if exclude_from_filtering is None:
        exclude_from_filtering = [False]*len(patches)

    gc.collect()
    for i, (patch, exclude) in enumerate(zip(patches, exclude_from_filtering)):
        if exclude:
            patches[i] = patch
            continue
        print("PATCHES: ",patch.shape)
        print("SLINE: ",WINDOW_SPLINE_2D[:, :, None].shape)
        if len(patch.shape)>3:
            WINDOW_SPLINE_2D = WINDOW_SPLINE_2D[..., None]
        patch = WINDOW_SPLINE_2D*patch
        patches[i] = patch
    gc.collect()

    return patches


def next_patch_to_fill(window_size, stride, reconstructed_shape):
    height = reconstructed_shape[0]
    width = reconstructed_shape[1]

    window_height = window_size[0]
    window_width = window_size[1]

    patch_idx = -1
    for i in range(0, width-window_height+1, stride):

        for j in range(0, height-window_width+1, stride):
            patch_idx += 1
            yield patch_idx, i, i+window_height, j, j+window_width


def apply_instance_aware_unpatchify(patches: np.array,
                               window_size: tuple,
                               stride: int,
                               reconstructed_shape: tuple):
    """
    Merge tiled overlapping patches smoothly.
    reconstructed_shape : shape of the padded image before patchify.
    patches: n_patches x (patch_dimensions)
    """

    y = np.zeros(reconstructed_shape)

    available = 0
    for patch_idx, y_start, y_end, x_start, x_end in next_patch_to_fill(window_size, stride, reconstructed_shape):
        windowed_patch = patches[patch_idx]
        current_situation = y[y_start:y_end, x_start:x_end]

        # lbl_c: lbl_n, ovrlp , from the viewpoint of current (1.0 overlap -> lbl_c is INSIDE lbl_b)
        overlap_dict_current = max_overlap_labels(current_situation, windowed_patch)
        # lbl_n: lbl_c, ovrlp , from the viewpoint of new  (1.0 overlap -> lbl_b is INSIDE lbl_c)
        overlap_dict_new = max_overlap_labels(windowed_patch, current_situation)

        for lbl, (other, overlap) in overlap_dict_new.items():


            other_ = overlap_dict_current.get(other, None)
            if other_ is None: # the other is a 0 which is not included in the dict
                current_situation[(windowed_patch == lbl)] = available + 1
                available += 1
                continue
            other_lbl = other_[0]
            other_overlap = other_[1]
            if overlap > 0.7: # new is covered to 70% with current
                if other_overlap > 0.5:
                    current_situation[windowed_patch == lbl] = other
                else: # probably the other is a misclassified mix of two / more touching instances
                    current_situation[(windowed_patch == lbl)] = available + 1
                    available += 1

            # the highest amount of overlap pixels in other is covered by lbl
            elif other_lbl == lbl: # the top overlap of the other is current
                if other_overlap > 0.7:
                    current_situation[windowed_patch == lbl] = other
                # insufficient overlap on other -> create a separate instance
                else:
                    current_situation[(windowed_patch == lbl) & (current_situation != other)] = available+1
                    available += 1
            else: # if other is 0 or there is not enough overlap in both directions -> add a new instance
                current_situation[(windowed_patch == lbl) & (current_situation != other)] = available + 1
                available += 1

    return y



def apply_instance_aware_unpatchify_V2(patches: np.array,
                               window_size: tuple,
                               stride: int,
                               reconstructed_shape: tuple):
    """
    Merge tiled overlapping patches smoothly.
    reconstructed_shape : shape of the padded image before patchify.
    patches: n_patches x (patch_dimensions)
    """

    y = np.zeros(reconstructed_shape)

    available = 0
    for patch_idx, y_start, y_end, x_start, x_end in next_patch_to_fill(window_size, stride, reconstructed_shape):
        windowed_patch = patches[patch_idx]
        current_situation = y[y_start:y_end, x_start:x_end]
        # _, ax = plt.subplots(1,2)
        # ax[0].imshow(windowed_patch)
        # ax[0].set_title("NEW")
        # ax[1].imshow(current_situation)
        # ax[1].set_title("OLD, before update")
        #
        # plt.show()
        #lbl_c: lbl_n, ovrlp , from the viewpoint of current (1.0 overlap -> lbl_c is INSIDE lbl_n)
        overlap_dict_current = max_overlap_labels(current_situation, windowed_patch, retrieve_list=True)
        # lbl_n: lbl_c, ovrlp , from the viewpoint of new  (1.0 overlap -> lbl_n is INSIDE lbl_c)
        overlap_dict_new = max_overlap_labels(windowed_patch, current_situation, retrieve_list=True)

        for lbl, (others, overlaps) in overlap_dict_new.items():
            if len(others)==1 and others[0]==0:
                print("NEW IS NOT OVERLAPPED", overlaps[0], lbl)
                current_situation[(windowed_patch == lbl)] = available + 1
                available += 1
                continue

            changed_once = False

            for other, overlap in zip(others, overlaps):
                if other ==0:
                    continue
                other_ = overlap_dict_current.get(other, None)
                if other_ is None: # the other is a 0 which is not included in the dict
                    print("OLD IS ONLY BACKGROUND")
                    current_situation[(windowed_patch == lbl)] = available + 1
                    available += 1
                    continue
                other_lbl: list = other_[0]
                other_overlap: list = other_[1]



                for other_overlap_, other_lbl_ in zip(other_overlap, other_lbl):
                    if other_lbl_ != lbl:
                        continue
                    if overlap>0.7:
                        print("Way too much overlap: it has to be the same instance ")
                        current_situation[windowed_patch == lbl] = other

                    elif overlap > 0.5: # new is covered to 70% with current
                        if other_overlap_ <0.4 or other_overlap_ > 0.6: # if the new is come small insignificant dot, or is essentially the same object but of smaller scale
                            print("NEW IS COVERED 70% with current and ")
                            current_situation[windowed_patch == lbl] = other
                        else: # probably the other is a misclassified mix of two / more touching instances
                            print("Overlap is above 0.5 but the other's overlap is between 40% and 60%")
                            if changed_once:
                                new_lbl = available
                            else:
                                available += 1
                                new_lbl = available

                            current_situation[(windowed_patch == lbl)] = new_lbl
                            break # the overlaps are sorted in descending order, if the other's overlap is <=0.5 then there can be no further overlap > 0.5
                    elif other_overlap_>0.7:
                        if changed_once:
                            new_lbl = available
                        else:
                            new_lbl = available+1
                            available+=1
                        current_situation[(windowed_patch == lbl)&((current_situation == other)|(current_situation == 0))] = new_lbl
                        continue
                    else: # if other is 0 or there is not enough overlap in both directions -> add a new instance
                        if changed_once:
                            new_lbl = available
                        else:
                            available += 1
                            new_lbl = available

                        current_situation[(windowed_patch == lbl) & (current_situation != other)] = new_lbl

                        break
        # _, ax = plt.subplots(1,2)
        # ax[0].imshow(current_situation)
        # ax[0].set_title("OLD, after update")
        # ax[1].imshow(y[y_start:y_end, x_start:x_end])
        # ax[1].set_title("OLD")
        # plt.show()
    return y





def apply_averaging_unpatchify(patches: np.array,
                               window_size: tuple,
                               stride: int,
                               reconstructed_shape: tuple):
    """
    Merge tiled overlapping patches smoothly.
    reconstructed_shape : shape of the padded image before patchify.
    patches: n_patches x (patch_dimensions)
    """
    subdivisions = np.ceil(window_size[0]/stride)

    y = np.zeros(reconstructed_shape)


    for patch_idx, y_start, y_end, x_start, x_end in next_patch_to_fill(window_size, stride, reconstructed_shape):
        windowed_patch = patches[patch_idx]
        y[y_start:y_end, x_start:x_end] += windowed_patch


    return y / (subdivisions ** 2)

def predict_on_patches(patches: List[tuple],
                       prediction_fn: Callable,
                       batch_size: int = 1) -> List[np.array]:
    """

    :param patches: tuple containing a patch for all input images
    :param prediction_fn:
    :param batch_size:
    :return:
    """
    predictions = []
    for batch in split_into_batches(patches, batch_size):
        prediction = prediction_fn(*batch)#.detach().to('cpu').numpy()

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().to('cpu').numpy(),
        # plt.imshow(prediction.squeeze())
        # plt.show()
        predictions.append(prediction)
    predictions = list(zip(*predictions)) # if multiple predictions-> [(1,1,1,1,1),(2,2,2,2,2,)]
    return [np.vstack(pred) for pred in predictions]

def apply_patchify(image, window_size: tuple, stride: int) -> list:
    """
    return a list of all patches created from the image
    :param image:
    :param window_size:
    :param stride:
    :return:
    """
    if len(image.shape) != len(window_size):
        raise Exception("For each dimension of the image there should be a window size. Currently"
                        f"image has the following shape: {image.shape}, window size is : {window_size}")

    patches = patchify(image, window_size, stride)
    #  reshape into : n_batches x (window shape)
    patches = patches.reshape((-1,)+window_size)
    return list(patches)

def predict_img_with_smooth_windowing(input_img: Union[Tuple[np.ndarray], np.ndarray],
                                      window_size,
                                      stride,
                                      prediction_fn: Callable,
                                      padding: Padding = None,
                                      batch_size=1,
                                      spline_window_fn: Callable = spline_window,
                                      apply_test_time_aug=True,
                                      exclude_from_filtering: List[bool] = None,
                                      averaging_functions_list: List[Callable] = None):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.

    Algorithm:
        1. tile image with overlap
        2. apply test-time augmentation
        for each augmented image (pad variable in the for loop):
            2a. patchify
            3. make predictions on all patches (all their augmentations)
            4. apply filtering to each patch
        5. reconstruct the whole image from the processed patches
    See 6th, 7th and 8th idea here:
    http://blog.kaggle.com/2017/05/09/dstl-satellite-imagery-competition-3rd-place-winners-interview-vladimir-sergey/
    """
    if padding is None:
        padding = AllAroundPadding(pad_repeat=2)

    if isinstance(input_img, np.ndarray):
        input_img = (input_img,)

    # 1. tile image with overlap
    pads: list = [padding.fit_transform(img, window_size, stride) for img in input_img] # N for the n images
    #print("mean after padding: ", [p.mean() for p in pads])
    # 2. apply test-time augmentation
    if apply_test_time_aug:
        augmented_images = list(zip(*[_rotate_mirror_do(img) for img in pads])) # [(i1.1,i2.1,...), (i1.2,i2.2,...)(i1.3,i2.3,...)]
    else:
        augmented_images = list(zip(*[[img] for img in pads]))
    #print("Means after mirroring: ", [p[0].mean() for p in augmented_images])

    # For every rotation:
    for i, pad in tqdm(enumerate(augmented_images)):
        shape = pad[0].shape # pads have shave H x W x C
        reconstruction_shape = (shape[0], shape[1])

        # patchify image
        if len(shape) == 3:
            window_size_ = (window_size, window_size, shape[-1])
        elif len(shape) == 2:
            window_size_ = (window_size, window_size)
        else:
            raise Exception(f"Unexpected shape of the padded image: {pad.shape}")
        # 2a. patchify the test-time augmented image
        patches: list = list(zip(*[apply_patchify(pad_, window_size_, stride) for pad_ in pad]))

        #print("Means after patchify: ", [p[0].mean() for p in patches])

        # 3. make predictions on all patches (all their augmentations) POSSIBLE MULTIPLE OUTPUTS PER PATCH!
        patches: List[np.ndarray] = predict_on_patches(patches, prediction_fn, batch_size=batch_size) # n outputs per augmented input images*

        if len(patches[0].shape) > 3:
            output_channels = patches[0].shape[-1]
            reconstruction_shape += (output_channels,)
        print(f"Shape after prediction: {patches[0].shape}")
        #print("Means after prediction: ", [p.mean() for p in patches])

        # 4. apply filtering to each patch
        patches: List[np.ndarray] = filter_predictions(patches, window_size, exclude_from_filtering, spline_window_fn=spline_window_fn, )


        #print("Means after filtering: ", [p.mean() for p in patches])
        if averaging_functions_list is None:
            averaging_functions_list = [apply_averaging_unpatchify]*len(patches)
        # 5. reconstruct the whole image from the processed patches
        one_padded_result: List[np.ndarray] = [func(patch,
                                                       (window_size, window_size, shape[-1]),
                                                       stride,
                                                       reconstruction_shape) for patch, func in zip(patches, averaging_functions_list)]


        print(f"Shape after averaging: {one_padded_result[0].shape}")
        print(f"Shape after averaging: {one_padded_result[1].shape}")
        print(f"Shape after averaging: {one_padded_result[2].shape}")

        augmented_images[i] = one_padded_result

    # Merge after rotations:
    if apply_test_time_aug:
        padded_result = []
        for aug_images in augmented_images:
            padded_result.append([_rotate_mirror_undo(aug_image) for aug_image in aug_images]) # the avg of all predictions on the different tt augmentations
    else:
        padded_result = augmented_images
    for i, prediction in enumerate(padded_result):
        padded_result[i] = [padding.inverse_transform(pred) for pred in prediction ]

    return padded_result
