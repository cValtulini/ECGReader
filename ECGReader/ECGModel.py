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


if __name__ == '__main__':
    # Load images and masks as tf.data.Dataset loading from ImageDataGenerator
    _, path = argv
    ecg_path = os.path.join(path, 'png/matches/img')
    masks_path = os.path.join(path, 'png/matches/masks')

    original_width = 9082
    original_height = 4410

    patch_size = 256
    stride_size = 64

    new_shape = (patch_size*int((original_height//2)//patch_size),
                 patch_size*int((original_width//2)//patch_size))

    n_images = 77     
    n_patches = (((new_shape[0] - patch_size) / stride_size) + 1) * (((new_shape[1] - patch_size) / stride_size) + 1) * n_images

    # Creating a generator to load images for the Dataset
    ecg_gen = ImageDataGenerator(
        rescale=1. / 255
        )
    masks_gen = ImageDataGenerator(
        rescale=1. / 255
        )

    # We need train_images.shape and train_images.dtype (same for test_images) as
    # parameters of `tf.data.Dataset.from_generator`
    ecg_images = next(
        ecg_gen.flow_from_directory(
            ecg_path, target_size=new_shape, class_mode=None, seed=42,
            color_mode='grayscale', batch_size=1
            )
        )
    masks_images = next(
        masks_gen.flow_from_directory(
            masks_path, target_size=new_shape, class_mode=None, seed=42,
            color_mode='grayscale', batch_size=1
            )
        )

    print('-' * _string_mult)
    print('Shapes:')
    print(ecg_images.shape)
    print(masks_images.shape)

    print('-' * _string_mult)
    print('Loading sets')

    # Creating the data set from the ImageDataGenerator objects we previously
    # initialized.
    ecg_set = tf.data.Dataset.from_generator(
        lambda: ecg_gen.flow_from_directory(
            ecg_path, target_size=new_shape,
            class_mode=None, seed=42, color_mode='grayscale', batch_size=1
            ),
        output_types=ecg_images.dtype,
        output_shapes=ecg_images.shape,
        name='ecg'
        )
    masks_set = tf.data.Dataset.from_generator(
        lambda: masks_gen.flow_from_directory(
            masks_path, target_size=new_shape,
            class_mode=None, seed=42, color_mode='grayscale', batch_size=1
            ),
        output_types=masks_images.dtype,
        output_shapes=masks_images.shape,
        name='mask'
        )

    print('Sets loaded.')
    print('-' * _string_mult)

    # Prints information about the elements of the data_set, in particular dtype and
    # shape, a note about this:
    # According to the guide (cited later) if I understood correctly all the element in
    # a dataset (batches?) have the same characteristic, element_spec return the
    # specification of
    # a single one.
    print('ECGs:')
    print(ecg_set.element_spec)
    print('Masks: ')
    print(masks_set.element_spec)

    # Divide into patches
    ecg_set = ecg_set.map(
        lambda x: tf.reshape(tf.image.extract_patches(
                x, sizes=[1, patch_size, patch_size, 1],
                strides=[1, stride_size, stride_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
                ),
            [-1, patch_size, patch_size, 1]
            )
        )
    masks_set = masks_set.map(
        lambda x: tf.reshape(tf.image.extract_patches(
                x, sizes=[1, patch_size, patch_size, 1],
                strides=[1, stride_size, stride_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
                ),
            [-1, patch_size, patch_size, 1]
            )
        )


    masks_set = masks_set.apply(tf.data.experimental.assert_cardinality(np.int64(n_patches)))
    mask_count = masks_set.cardinality().numpy()
    print(mask_count)

    # Create a single dataset from the two sets
    ecg_set = ecg_set.unbatch()
    masks_set = masks_set.unbatch()
    ecg_masks_set = tf.data.Dataset.zip((ecg_set, masks_set))

    print(ecg_masks_set.element_spec)

    ## We find the average number of nonzero pixels in the masks here
    # somma=0
    # for ecg, mask in ecg_masks_set.take(mask_count):
    #     somma+=np.count_nonzero(mask)
    # average_nonzero_pixels=somma/mask_count

    # Select patches
    ecg_masks_set = ecg_masks_set.filter(
        lambda x, y: tf.math.greater(tf.math.count_nonzero(y), 351) #351 was found as one third of the average of nonzero values in the masks dataset
        )
    ecg_masks_set = ecg_masks_set.batch(batch_size=1)


    # Check on the empty masks
    # for ecg, mask in ecg_masks_set:
    #     if np.count_nonzero(mask)==0:
    #         print("Empty mask found")

    ecg_set_filtered = ecg_masks_set.map(lambda a, b: a)
    mask_set_filtered = ecg_masks_set.map(lambda a, b: b)



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
