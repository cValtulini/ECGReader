import os
from sys import argv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


_string_mult = 100


def show(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def loadDataset(img_gen, img_shape, path, batch_size=1, seed=42,
                color_mode='grayscale', name=None):
    # We need images.shape and images.dtype as parameters of
    # `tf.data.Dataset.from_generator`
    images = next(
        img_gen.flow_from_directory(
            path, target_size=img_shape, class_mode=None, seed=seed,
            color_mode=color_mode, batch_size=batch_size
            )
        )

    print('-' * _string_mult)
    print(f'Loading dataset from {path}:')

    spec = tf.TensorSpec(images.shape, dtype=images.dtype, name=name)

    print(f'TensorSpec: {spec}')

    # Creating the data set from the ImageDataGenerator object.
    data_set = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(
            path, target_size=img_shape, class_mode=None, seed=seed,
            color_mode=color_mode, batch_size=batch_size
            ),
        output_signature=spec
        )

    print(tf.data.experimental.cardinality(data_set))

    print('Sets loaded.')
    print('-' * _string_mult)

    return data_set


if __name__ == '__main__':
    # Load images and masks as tf.data.Dataset loading from ImageDataGenerator
    _, path = argv
    ecg_path = os.path.join(path, 'png/matches/img')
    masks_path = os.path.join(path, 'png/matches/masks')

    ecg_train_path = os.path.join(ecg_path, 'train')
    ecg_val_path = os.path.join(ecg_path, 'val')
    ecg_test_path = os.path.join(ecg_path, 'train')

    ecg_rows = 6
    ecg_cols = 2

    # Number of patches for each lead on the time axis (width)
    time_patch_lead = 10

    # TODO: Substitute masks values
    original_mask_shape = (3053, 8001)
    original_ecg_shape = (4410, 9082)

    mask_subs_coeff = 5
    ecg_subs_coeff = 5

    # Roughly, the track length is not exactly half the width of the image but this is
    # already considered in the creation of the mask adding proper white spaces
    mask_lead_shape = (original_mask_shape[0] // mask_subs_coeff // ecg_rows,
                       original_mask_shape[1] // mask_subs_coeff // ecg_cols)
    ecg_lead_shape = (original_ecg_shape[0] // ecg_subs_coeff // ecg_rows,
                      original_ecg_shape[1] // ecg_subs_coeff // ecg_cols)

    mask_shape = (mask_lead_shape[0] * ecg_rows, mask_lead_shape[1] * ecg_cols)
    ecg_shape = (ecg_lead_shape[0] * ecg_rows, ecg_lead_shape[1] * ecg_cols)

    mask_stride = (mask_lead_shape[0], mask_lead_shape[1] // time_patch_lead)
    ecg_stride = (ecg_lead_shape[1], ecg_lead_shape[1] // time_patch_lead)

    mask_patch_shape = (mask_lead_shape[0], mask_stride[1] * 2)
    ecg_patch_shape = (ecg_lead_shape[0] * 2, ecg_stride[1] * 2)

    n_patches_row = ((mask_shape[1] - mask_patch_shape[1]) // mask_patch_shape[1]) + 1
    n_patches_col = ((mask_shape[0] - mask_patch_shape[0]) // time_patch_lead) + 1
    n_patches_image = n_patches_row * n_patches_row

    ecg_pad = ecg_lead_shape[0] // 2

    # TODO: Check after train/val/test split
    train_set_card = 48
    val_set_card = 15
    test_set_card = 14

    n_train_patches = n_patches_image * train_set_card
    n_val_patches = n_patches_image * val_set_card
    n_test_patches = n_patches_image * test_set_card

    # Creating a generator to load images for the Dataset
    image_gen = ImageDataGenerator(
        rescale=1. / 255
        )

    ecg_set = loadDataset(image_gen, ecg_shape, ecg_path, name='ecg')

    # Prints information about the elements of the data_set, in particular dtype and
    # shape, a note about this:
    # According to the guide (cited later) if I understood correctly all the element in
    # a dataset (batches?) have the same characteristic, element_spec return the
    # specification of
    # a single one.
    # print('ECGs:')
    # print(ecg_set.element_spec)
    # print('Masks: ')
    # print(masks_set.element_spec)

    # Divide into patches
    # ecg_set = ecg_set.map(
    #     lambda x: tf.reshape(tf.image.extract_patches(
    #             x, sizes=[1, patch_size, patch_size, 1],
    #             strides=[1, stride_size, stride_size, 1],
    #             rates=[1, 1, 1, 1],
    #             padding='VALID'
    #             ),
    #         [-1, patch_size, patch_size, 1]
    #         )
    #     )
    # masks_set = masks_set.map(
    #     lambda x: tf.reshape(tf.image.extract_patches(
    #             x, sizes=[1, patch_size, patch_size, 1],
    #             strides=[1, stride_size, stride_size, 1],
    #             rates=[1, 1, 1, 1],
    #             padding='VALID'
    #             ),
    #         [-1, patch_size, patch_size, 1]
    #         )
    #     )
    #
    # masks_set = masks_set.apply(
    #     tf.data.experimental.assert_cardinality(np.int64(n_patches))
    #     )
    # mask_count = masks_set.cardinality().numpy()
    # print(mask_count)

    # Create a single dataset from the two sets
    # ecg_set = ecg_set.unbatch()
    # masks_set = masks_set.unbatch()
    # ecg_masks_set = tf.data.Dataset.zip((ecg_set, masks_set))
    #
    # print(ecg_masks_set.element_spec)

    # We find the average number of nonzero pixels in the masks here
    # somma=0
    # for ecg, mask in ecg_masks_set.take(mask_count):
    #     somma+=np.count_nonzero(mask)
    # average_nonzero_pixels=somma/mask_count

    # Select patches that have a certain amount of signal in it,
    # 351 was found as one third of the average of nonzero values in the masks dataset
    # ecg_masks_set = ecg_masks_set.filter(
    #     lambda x, y: tf.math.greater(tf.math.count_nonzero(y), 0)
    #     )
    # ecg_masks_set = ecg_masks_set.batch(batch_size=1)

    # Check on the empty masks
    # for ecg, mask in ecg_masks_set:
    #     if np.count_nonzero(mask)==0:
    #         print("Empty mask found")

    # Unpack dataset to have back the masks and ecgs datasets filtered
    # ecg_set_filtered = ecg_masks_set.map(lambda a, b: a)
    # mask_set_filtered = ecg_masks_set.map(lambda a, b: b)

    # Here we perform the dataset division in train and validation, by slicing it
    # so that we have a 3/1 train/validation split.
    # Meaning 3 records will go to training, then 1 record to validation, then repeat.
    # The flat_map(lambda ds: ds) is because window() returns the results in batches,
    # which we don't want. So we flatten it back out.
    # split = 3
    # ecg_train = ecg_set_filtered.window(split, split + 1).flat_map(lambda ds: ds)
    # mask_train = mask_set_filtered.window(split, split + 1).flat_map(lambda ds: ds)
    # ecg_validation = ecg_set_filtered.skip(split).window(1, split + 1).flat_map(
    #     lambda ds: ds
    #     )
    # mask_validation = mask_set_filtered.skip(split).window(1, split + 1).flat_map(
    #     lambda ds: ds
    #     )

    # I think it will be best to apply transformations after selection if we apply them
    # through tf.data.Dataset.map(), I've seen there are a bunch of tf.image functions
    # (it's better to use those for speed and memory reasons) for preprocessing but
    # haven't looked up if/how to apply them randomly (with random parameters as
    # ImageDataGenerator does) to patches/images

    # Check some patches (visualize)

    # Nothing to add from here on, just jotted down the comments to eventually expand,
    # feel free to add modify and do every kind of thing on this code (or do anything
    # else at all) and tell me if I have to merge some commit tomorrow.
    # I was trying to do all this with info from the
    # tf.keras.preprocessing.image.ImageDataGenerator page of the tensorflow API + the
    # tf.data.Dataset page + the "Build Tensorflow input pipelines" guide that you can
    # find linked in both the previous pages ( surely on the Dataset one) see the
    # Preprocessing section of this one for the visualization part that I just started.

    # Load model

    # Train model

    # Evaluate model on test
