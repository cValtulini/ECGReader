import os
from sys import argv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import imgaug
from ECGDataset import ECGDataset


_string_mult = 100


def show(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


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
        if i == 0:
            x = layers.UpSampling2D((1, 2))(x)
        else:
            x = layers.UpSampling2D(2)(x)

        # Project residual
        crop_shape = ((previous_block_activation[-(i + 1)].shape[1] - x.shape[1]) // 2,
                      (previous_block_activation[-(i + 1)].shape[2] - x.shape[2]) // 2)
        residual = layers.Cropping2D(crop_shape)(previous_block_activation[-(i + 1)])
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

    imgaug.seed(42)

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
    ecg_patch_shape = (320, 200)

    mask_stride = (mask_patch_shape[0], mask_patch_shape[1] // 2)
    ecg_stride = (ecg_patch_shape[0] // 2, ecg_patch_shape[1] // 2)

    mask_shape = (mask_patch_shape[0] * ecg_rows,
                  mask_stride[1] * t_patch_lead * ecg_cols)
    ecg_shape = (ecg_stride[0] * ecg_rows, ecg_stride[1] * t_patch_lead * ecg_cols)

    ecg_pad = ecg_stride[0] // 2

    train_set_card = 48
    val_set_card = 15
    test_set_card = 14

    # Creating a generator to load images for the Dataset
    image_gen = ImageDataGenerator()

    # Creating train val test sets
    train_ecg_set = ECGDataset(
        image_gen, ecg_shape, ecg_train_path, train_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, binarize_patches=False
        )
    train_mask_set = ECGDataset(
        image_gen, mask_shape, mask_train_path, train_set_card, mask_patch_shape,
        mask_stride
        )
    train_set = tf.data.Dataset.zip((train_ecg_set.patches_set,
                                     train_mask_set.patches_set))
    val_ecg_set = ECGDataset(
        image_gen, ecg_shape, ecg_val_path, val_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, binarize_patches=False
        )
    val_mask_set = ECGDataset(
        image_gen, mask_shape, mask_val_path, val_set_card, mask_patch_shape,
        mask_stride
        )
    val_set = tf.data.Dataset.zip(
        (val_ecg_set.patches_set, val_mask_set.patches_set)
        )

    test_ecg_set = ECGDataset(
        image_gen, ecg_shape, ecg_test_path, test_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, binarize_patches=False
        )
    test_mask_set = ECGDataset(
        image_gen, mask_shape, mask_test_path, test_set_card, mask_patch_shape,
        mask_stride
        )
    test_set = tf.data.Dataset.zip(
        (test_ecg_set.patches_set, test_mask_set.patches_set)
        )

    print(train_ecg_set.patches_set.element_spec)
    print(train_mask_set.patches_set.element_spec)

    # Seems that we can pass x and y as a single dataset

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    basicUNet = getBaseModel(ecg_patch_shape, num_classes=2)
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
