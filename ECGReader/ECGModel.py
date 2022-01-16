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
    # Pass the data folder as an argument when calling the script, later it can be the
    # argument of a function call
    _, path = argv
    images_path = os.path.join(path, 'png/matches')

    # Creating a generator to load images for the Dataset
    # For now I'm just trying to set up thing but we have a decision to make that will
    # impact the way we set up this process: how do we divide train and test set
    # If we divide them through the folders then it will be pretty straight forward,
    # we can create separate ImageDataGenerator with different parameters (this is true
    # for the subdivision of train and validation too). Otherwise we have to look up (
    # or find a way to) how to properly divide data when loading them this way.
    # Validation split divides the data that are loaded, I'm actually using it now for
    # what I'm calling test_set, planning to furtherly divide train_set into train and
    # validation (even though I still don't know how I'm going to do it.
    images_gen = ImageDataGenerator(
        rescale=1. / 255, validation_split=0.2
        )

    # I just need these two objects to set up output_shapes and output_types of the
    # function that creates the datasets later.
    # As target size I've decided to subsample with a factor 1/3 the original size of
    # images (we may want to look up how this subsampling is performed). class_mode
    # None specifies we don't want labels based on the folder in which data are
    # contained to be returned when calling flow_from_directory, seed is the usual
    # parameter that initializes random things and subset selects which of the two
    # between train and validation set we want (from validation_split parameter)
    train_images = next(
        images_gen.flow_from_directory(
            images_path, target_size=(1408, 3072), class_mode=None, seed=42,
            subset='training'
            )
        )
    test_images = next(
        images_gen.flow_from_directory(
            images_path, target_size=(1408, 3072), class_mode=None, seed=42,
            subset='validation'
            )
        )

    print('-' * _string_mult)
    print('Loading sets')

    # Creating the data set from the ImageDataGenerator objects I previously
    # initialized, I pass flow_from_directory as a lambda function to avoid passing it
    # as an object because I would have needed to pass its parameters as an additional
    # list parameter of from_generator (haven't tried so I don't actually know how to
    # properly set that up)
    train_set = tf.data.Dataset.from_generator(
        lambda: images_gen.flow_from_directory(
            images_path, target_size=(1408, 3072),
            class_mode=None, seed=42,
            subset='training'
            ),
        output_types=train_images.dtype,
        output_shapes=train_images.shape
        )
    test_set = tf.data.Dataset.from_generator(
        lambda: images_gen.flow_from_directory(
            images_path, target_size=(1408, 3072),
            class_mode=None, seed=42,
            subset='validation'
            ),
        output_types=test_images.dtype,
        output_shapes=test_images.shape
        )

    print('Sets loaded.')
    print('-' * _string_mult)

    # The map method applies a function passed as parameter to all the objects of the
    # set, in this case turning images into grayscale: I haven't found (or looked for,
    # actually) a way to do this automatically through ImageDataGenerator but I don't
    # think it will be a problem...
    train_set = train_set.map(tf.image.rgb_to_grayscale)
    test_set = test_set.map(tf.image.rgb_to_grayscale)

    # Prints information about the elements of the data_set, in particular dtype and
    # shape, a note about this:
    # For the train_set it now returns (32, height, width, 1) where 1 is for grayscale
    # and 32 is for the number of elements, I think this is related to batches division
    # of the set but I still haven't had the time to look up how to use/manage this.
    print('Train: ')
    print(train_set.element_spec)
    print('Test:')
    print(test_set.element_spec)

    # Shows examples from the set, still have to understand how take works
    for image in train_set.take(3):
        show(image[0, :, :, 0])

    # Divide into patches
    # I found this link and wanted to try this
    # https://stackoverflow.com/questions/64326029/load-tensorflow-images-and-create-patches

    # Select patches
    # I don't know if it's possible (and don't think it is) to just select them through
    # indexing

    # I think it will be best to apply transformations after selection if we apply them
    # through tf.data.Dataset.map(), I've seen there are a bunch of tf.image functions
    # (it's better to use those for speed and memory reasons) for preprocessing but
    # haven't looked up if/how to apply them randomly (with random parameters as
    # ImageDataGenerator does) to patches/images

    # Check some patches (visualize)

    # Nothing to add from here on, just jotted down the comments to eventually expand,
    # feel free to add modify and do every kind of thing on this code (or do anything
    # else at all) and tell me if I have to merge some commit tomorrow.
    # I was trying to do all this with info from the tf.keras.preprocessing.image.ImageDataGenerator
    # page of the tensorflow API + the tf.data.Dataset page + the "Build Tensorflow
    # input pipelines" guide that you can find linked in both the previous pages (
    # surely on the Dataset one) see the Preprocessing section of this one for the
    # visualization part that I just started.

    # Load model

    # Train model

    # Evaluate model on test
