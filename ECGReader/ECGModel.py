import os
from system import argv
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    # Load images and masks as tf.data.Dataset loading from ImageDataGenerator
    _, path = argv
    images_path = os.path.join(path, 'png/matches')

    images_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_images = next(
        images_gen.flow_from_directory(
            images_path, target_size=(4480, 2176), class_mode=None, seed=42,
            subset='training'
            )
        )
    test_images = next(
        images_gen.flow_from_directory(
            images_path, target_size=(4480, 2176), class_mode=None, seed=42,
            subset='validation'
            )
        )

    train_set = tf.data.Dataset.from_generator(
        lambda: images_gen.flow_from_directory(
            images_path, target_size=(4480, 2176),
            class_mode=None, seed=42,
            subset='training'
            ),
        output_signature=tf.type_spec_from_value(train_images.shape),
        name='train'
        )
    test_set = tf.data.Dataset.from_generator(
        lambda: images_gen.flow_from_directory(
            images_path, target_size=(4480, 2176),
            class_mode=None, seed=42,
            subset='validation'
            ),
        output_signature=tf.type_spec_from_value(test_images.shape),
        name='test'
        )

    # Divide into patches

    # Divide into train test

    # Select patches

    # Check some patches (visualize)

    # Load model

    # Train model

    # Evaluate model on test
