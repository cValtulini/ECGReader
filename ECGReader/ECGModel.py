import os
from sys import argv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
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

    spec = tf.TensorSpec(images.shape, dtype=images.dtype, name=name)

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


def createAugmenter():
    augmenter = iaa.SomeOf(
        (1, None), [
                iaa.OneOf(
                    [
                            iaa.Add([-50, 50]),
                            iaa.Multiply((0.6, 1.4))
                            ]
                    ),
                iaa.OneOf(
                    [
                            iaa.OneOf(
                                [
                                        iaa.AdditiveGaussianNoise(
                                            scale=(0, 0.2 * 255), per_channel=True
                                            ),
                                        iaa.SaltAndPepper((0.01, 0.2), per_channel=True)
                                        ]
                                ),
                            iaa.GaussianBlur(sigma=(0.01, 1.0))
                            ]
                    ),
                iaa.imgcorruptlike.DefocusBlur(severity=1),
                iaa.imgcorruptlike.Saturate(severity=1)
                ]
        )
    return augmenter


def createPatchesSet(data_set, patch_shape, stride_shape, pad_horizontal=False,
                     pad_horizontal_size=None, augment=False, grayscale=True,
                     color_invert=True):
    """

    Parameters
    ----------
    data_set
    patch_shape
    stride_shape
    pad_horizontal
    pad_horizontal_size
    augment
    grayscale
    color_invert

    Returns
    -------

    """

    original_card = tf.data.experimental.cardinality(data_set)
    original_shape = (data_set.element_spec.shape[1], data_set.element_spec.shape[2])

    if color_invert:
        data_set = data_set.map(lambda x: 255.0 - x)

    if pad_horizontal:
        if isinstance(pad_horizontal_size, type(None)):
            print('No pad size set. Padding not added.')
        else:
            data_set = data_set.map(
                lambda x: tf.image.pad_to_bounding_box(
                    x, pad_horizontal_size, 0, x.shape[1] + 2 * pad_horizontal_size,
                    x.shape[2]
                    )
                )

    data_set = data_set.map(
        lambda x: tf.reshape(
            tf.image.extract_patches(
                x, sizes=[1, patch_shape[0], patch_shape[1], 1],
                strides=[1, stride_shape[0], stride_shape[1], 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
                ),
            [-1, patch_shape[0], patch_shape[1], 3]
            )
        )

    n_patches_row = ((original_shape[1] - patch_shape[1]) // stride_shape[1]) + 1
    n_patches_col = ((original_shape[0] - patch_shape[0]) // stride_shape[0]) + 1
    n_patches = n_patches_row * n_patches_col

    data_set = data_set.apply(
        tf.data.experimental.assert_cardinality(original_card * n_patches)
        )

    if augment:
        in_type = data_set.element_spec.dtype
        in_shape = data_set.element_spec.shape

        augmenter = createAugmenter()
        data_set = data_set.map(
            lambda x: tf.numpy_function(
                func=augmenter.augment_images, inp=[tf.cast(x, tf.uint8)], Tout=tf.uint8
                )
            )

        # Shape is lost after applying imgaug's augmentations but it's still the same
        # as the input's shape
        data_set = data_set.map(lambda x: tf.reshape(x, in_shape))
        data_set = data_set.map(lambda x: tf.cast(x, in_type))

    if grayscale:
        data_set = data_set.map(lambda x: tf.image.rgb_to_grayscale(x))

    data_set = data_set.map(lambda x: x / 255)

    return data_set


def getBaseModel(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    """
    [First half of the network: down-sampling inputs]
    """

    # Entry block
    x = layers.Conv2D(
        32, 3,  # strides=2,
        padding="same"
        )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    previous_block_activation = [x]
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        previous_block_activation.append(x)  # Set aside next residual
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    """
    [Second half of the network: up-sampling inputs]
    """

    print(previous_block_activation)
    for i, filters in enumerate([128, 64, 32]):
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation[-(i + 1)])
        residual = previous_block_activation[-(i + 1)]
        x = layers.add([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    # Load images and masks as tf.data.Dataset loading from ImageDataGenerator
    _, path = argv

    ecg_path = os.path.join(path, 'png/matches/img')

    mask_path = os.path.join(path, 'png/matches/masks')

    ecg_train_path = os.path.join(ecg_path, 'train')
    ecg_val_path = os.path.join(ecg_path, 'val')
    ecg_test_path = os.path.join(ecg_path, 'test')

    mask_train_path = os.path.join(mask_path, 'train')
    mask_val_path = os.path.join(mask_path, 'val')
    mask_test_path = os.path.join(mask_path, 'test')

    ecg_rows = 6
    ecg_cols = 2

    # Number of patches for each lead on the time axis (width)
    t_patch_lead = 8

    original_mask_shape = (3149, 6102)
    original_ecg_shape = (4410, 9082)

    # We define mask and ecg overall shape based on patches parameters
    mask_patch_shape = (160, 160)
    ecg_patch_shape = (360, 200)

    mask_stride = (mask_patch_shape[0], mask_patch_shape[1] // 2)
    ecg_stride = (ecg_patch_shape[0] // 2, ecg_patch_shape[1] // 2)

    mask_shape = (mask_patch_shape[0] * ecg_rows,
                  mask_stride[1] * t_patch_lead * ecg_cols)
    ecg_shape = (ecg_stride[0] * ecg_rows, ecg_stride[1] * t_patch_lead * ecg_cols)

    print(f'mask patches: {mask_patch_shape}')
    print(f'ecg patches: {ecg_patch_shape}')

    ecg_pad = ecg_stride[0] // 2

    train_set_card = 48
    val_set_card = 15
    test_set_card = 14

    # Creating a generator to load images for the Dataset
    image_gen = ImageDataGenerator()

    ecg_set = loadDataset(image_gen, ecg_shape, ecg_path, 77, name='ecg')
    mask_set = loadDataset(image_gen, mask_shape, mask_path, 77, name='mask')

    ecg_set = createPatchesSet(ecg_set, ecg_patch_shape, ecg_stride, pad_horizontal=True,
                               pad_horizontal_size=ecg_pad, augment=True)
    mask_set = createPatchesSet(mask_set, mask_patch_shape, mask_stride)

    mask_set = mask_set.map(lambda x: tf.math.greater(x, 1e-5))

    print(ecg_set.element_spec)
    print(mask_set.element_spec)

    # Seems that we can pass x and y as a single dataset
    train_set = tf.data.Dataset.zip((ecg_set, mask_set))

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    basicUNet = getBaseModel(ecg_shape, num_classes=2)
    basicUNet.summary()
    #
    # # TODO: Just copy pasted this
    # # Configure the model for training.
    # # We use the "sparse" version of categorical_crossentropy
    # # because our target data is integers.
    # basicUNet.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    #
    # callbacks = [
    #     keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
    # ]
    #
    # # Train the model, doing validation at the end of each epoch.
    # epochs = 15
    # model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)