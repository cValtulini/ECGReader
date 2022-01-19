import os
from sys import argv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import imgaug
from imgaug import augmenters as iaa


_string_mult = 100

imgaug.seed(42)

def show(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def loadDataset(img_gen, img_shape, img_path, n_images, batch_size=1, seed=42, name=None):
    """

    Parameters
    ----------
    img_gen
    img_shape
    img_path
    n_images
    batch_size
    seed
    name

    Returns
    -------

    """
    # We need images.shape and images.dtype as parameters of
    # `tf.data.Dataset.from_generator`
    images = next(
        img_gen.flow_from_directory(
            img_path, target_size=img_shape, class_mode=None, seed=seed,
            batch_size=batch_size
            )
        )

    print('-' * _string_mult)
    print(f'Loading dataset from {path}:')

    spec = tf.TensorSpec(images.shape, dtype=tf.uint8, name=name)

    print(f'TensorSpec: {spec}')

    # Creating the data set from the ImageDataGenerator object.
    data_set = tf.data.Dataset.from_generator(
        lambda: img_gen.flow_from_directory(
            img_path, target_size=img_shape, class_mode=None, seed=seed,
            batch_size=batch_size
            ),
        output_signature=spec
        )

    data_set = data_set.apply(
        tf.data.experimental.assert_cardinality(n_images)
        )

    print('Spec:')
    print(data_set.element_spec)

    print('Loaded.')
    print('-' * _string_mult)

    return data_set


def augmentPatch():
    augmenter = iaa.SomeOf((1, None),[
            iaa.Add([-30, 30]),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2), per_channel=True),
            iaa.Multiply((0.4, 1.6)),
            #iaa.SaltAndPepper((0.01, 0.02), per_channel=True),
            iaa.GaussianBlur(sigma=(0.01, 1.0)),
            iaa.MotionBlur(k=(3, 5)),
            #iaa.imgcorruptlike.DefocusBlur(severity=1),  #imgcorruptlike works only on uint8 images
            #iaa.imgcorruptlike.ZoomBlur(severity=1),
            #iaa.imgcorruptlike.Saturate(severity=1),
            #iaa.imgcorruptlike.Spatter(severity=1),
            ])
    return augmenter


def createPatchesSet(data_set, patch_shape, stride_shape, augment=False,
                     grayscale=True, color_invert=True):
    """

    Parameters
    ----------
    data_set
    patch_shape
    stride_shape
    augment
    grayscale
    color_invert

    Returns
    -------

    """

    original_card = tf.data.experimental.cardinality(data_set)
    original_shape = (data_set.element_spec.shape[1], data_set.element_spec.shape[2])

    data_set = data_set.map(
        lambda x: tf.reshape(
            tf.image.extract_patches(
                x, sizes=[1, patch_shape[0], patch_shape[1], 1],
                strides=[1, stride_shape[0], stride_shape[1], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
                ),
            [-1, patch_shape[0], patch_shape[1], 1]
            )
        )

    n_patches_row = ((original_shape[1] - patch_shape[1]) // stride_shape[1]) + 1
    n_patches_col = ((original_shape[0] - patch_shape[0]) // stride_shape[0]) + 1
    n_patches = n_patches_row * n_patches_col

    data_set = data_set.apply(
        tf.data.experimental.assert_cardinality(original_card * n_patches)
        )

    if augment:
        data_set = data_set.map(
            lambda x : tf.numpy_function(func=augmentPatch().augment_images, inp=[(x)], Tout=tf.float32)
        )

    if grayscale:
        data_set = data_set.map(tf.image.grayscale_to_rgb)

    data_set = data_set.map(lambda x: tf.cast(x, tf.float32)).map(lambda x: x / 255)

    if color_invert:
        data_set = data_set.map(lambda x: 1 - x)

    return data_set


if __name__ == '__main__':
    # Load images and masks as tf.data.Dataset loading from ImageDataGenerator
    _, path = argv

    ecg_path = os.path.join(path, 'png/matches/img')

    mask_path = os.path.join(path, 'png/matches/masks')

    ecg_train_path = os.path.join(ecg_path, 'train')
    ecg_val_path = os.path.join(ecg_path, 'val')
    ecg_test_path = os.path.join(ecg_path, 'train')

    mask_train_path = os.path.join(mask_path, 'train')
    mask_val_path = os.path.join(mask_path, 'val')
    mask_test_path = os.path.join(mask_path, 'train')

    ecg_rows = 6
    ecg_cols = 2

    # Number of patches for each lead on the time axis (width)
    t_patch_lead = 10

    original_mask_shape = (3149, 6102)
    original_ecg_shape = (4410, 9082)

    mask_subs_coeff = 4
    ecg_subs_coeff = 5

    # We define mask and ecg overall shape based on patches parameters
    mask_lead_shape = (original_mask_shape[0] // mask_subs_coeff // ecg_rows,
                       original_mask_shape[1] // mask_subs_coeff // ecg_cols)
    ecg_lead_shape = (original_ecg_shape[0] // ecg_subs_coeff // ecg_rows,
                      original_ecg_shape[1] // ecg_subs_coeff // ecg_cols)

    mask_stride = (mask_lead_shape[0], mask_lead_shape[1] // t_patch_lead)
    ecg_stride = (ecg_lead_shape[0], ecg_lead_shape[1] // t_patch_lead)

    mask_patch_shape = (mask_lead_shape[0], mask_stride[1] * 2)
    ecg_patch_shape = (ecg_lead_shape[0] * 2, ecg_stride[1] * 2)

    mask_shape = (mask_lead_shape[0] * ecg_rows, mask_stride[1] * t_patch_lead * ecg_cols)
    ecg_shape = (ecg_lead_shape[0] * ecg_rows, ecg_stride[1] * t_patch_lead * ecg_cols)

    print(f'mask patches: {mask_patch_shape}')
    print(f'ecg patches: {ecg_patch_shape}')

    ecg_pad = ecg_lead_shape[0] // 2

    train_set_card = 48
    val_set_card = 15
    test_set_card = 14

    # Creating a generator to load images for the Dataset
    image_gen = ImageDataGenerator()

    ecg_set = loadDataset(image_gen, ecg_shape, ecg_path, 77, name='ecg')
    mask_set = loadDataset(image_gen, mask_shape, mask_path, 77, name='mask')

    # Pad ECG
    ecg_set = ecg_set.map(
        lambda x: tf.image.pad_to_bounding_box(
            x, ecg_pad, 0,
            x.shape[1] + 2 * ecg_pad, x.shape[2]
            )
        )

    ecg_set = createPatchesSet(ecg_set, ecg_patch_shape, ecg_stride, augment=True)
    mask_set = createPatchesSet(mask_set, mask_patch_shape, mask_stride)

    for image in ecg_set.take(10):
        i = 0
        while i < 19:
            show(image[i, :, :, 0])
            i += 1

    # We find the average number of nonzero pixels in the masks
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
