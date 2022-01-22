"""
Defines the main utilities for loading the ECGDataset
"""
import tensorflow as tf
from ECGPreProcessing import createAugmenter


_string_mult = 100


class ECGDataset(object):

    def __init__(self, img_shape, path_to_img, n_images, patch_shape,
                 stride_shape, batch_size=None, seed=42, pad_horizontal=False,
                 pad_horizontal_size=None, augment_patches=False, color_invert=True,
                 one_hot_encode=True, binarize_threshold=1e-5):
        """

        Parameters
        ----------
        img_shape
        path_to_img
        n_images
        patch_shape
        stride_shape
        batch_size
        seed
        pad_horizontal
        pad_horizontal_size
        augment_patches
        color_invert
        one_hot_encode
        binarize_threshold
        """

        self.shape = img_shape
        self.n_images = n_images

        self.data_set = self._loadDataset(path_to_img, seed)

        self.patch_shape = patch_shape
        self.stride_shape = stride_shape

        self.patches_set = self._createPatchesSet(pad_horizontal, pad_horizontal_size,
                                                  augment_patches, batch_size,
                                                  color_invert, one_hot_encode,
                                                  binarize_threshold)


    def _loadDataset(self, path_to_img, seed):
        """

        Parameters
        ----------
        img_gen
        path_to_img
        seed

        Returns
        -------

        """

        print('-' * _string_mult)
        print(f'Loading dataset from {path_to_img}:')

        # spec = tf.TensorSpec(images.shape, dtype=images.dtype)

        # Creating the data set from the ImageDataGenerator object.
        data_set = tf.keras.utils.image_dataset_from_directory(
            path_to_img, labels=None, label_mode=None, color_mode='grayscale',
            batch_size=1, image_size=self.shape, shuffle=False, seed=seed,
            interpolation='nearest'
            )
        data_set = data_set.unbatch()
        data_set = data_set.batch(1, drop_remainder=True)

        print('Loaded.')
        print('-' * _string_mult)

        return data_set


    def _createPatchesSet(self, pad_horizontal, pad_horizontal_size, augment_patches,
                          batch_size, color_invert, one_hot_encode, binarize_threshold):
        """

        Parameters
        ----------
        pad_horizontal
        pad_horizontal_size
        augment_patches
        color_invert
        one_hot_encode
        binarize_threshold

        Returns
        -------

        """
        patches_set = self.data_set.take(-1)

        if color_invert:
            patches_set = patches_set.map(
                lambda x: 255 - x,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        if pad_horizontal:
            if isinstance(pad_horizontal_size, type(None)):
                print('Padding requested but no pad size set. Padding not added.')
            else:
                patches_set = patches_set.map(
                    lambda x: tf.image.pad_to_bounding_box(
                        x, pad_horizontal_size, 0,
                        self.shape[0] + 2 * pad_horizontal_size, self.shape[1]
                        ),
                    num_parallel_calls=tf.data.AUTOTUNE
                    )

        patches_set = patches_set.map(
            lambda x: tf.reshape(
                tf.image.extract_patches(
                    x, sizes=[1, self.patch_shape[0], self.patch_shape[1], 1],
                    strides=[1, self.stride_shape[0], self.stride_shape[1], 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                    ),
                [-1, self.patch_shape[0], self.patch_shape[1], 1]
                ),
            num_parallel_calls=tf.data.AUTOTUNE
            )

        self.patches_per_image = patches_set.element_spec.shape[0]

        if not isinstance(batch_size, type(None)):
            self.batch_size = batch_size
            patches_set = patches_set.unbatch()
            patches_set = patches_set.batch(self.batch_size, drop_remainder=True)

        self.n_patches = self.n_images * self.patches_per_image

        if augment_patches:
            in_shape = patches_set.element_spec.shape

            augmenter = createAugmenter()
            patches_set = patches_set.map(
                lambda x: tf.numpy_function(
                    func=augmenter.augment_images, inp=[x], Tout=tf.uint8
                    ),
                num_parallel_calls=tf.data.AUTOTUNE
                )

            # Shape is lost after applying numpy_function
            patches_set = patches_set.map(
                lambda x: tf.reshape(x, in_shape),
                num_parallel_calls=tf.data.AUTOTUNE
                )

        patches_set = patches_set.map(
            lambda x: tf.cast(x, dtype=tf.float32) / 255.0,
            num_parallel_calls=tf.data.AUTOTUNE
            )

        if one_hot_encode:
            patches_set = patches_set.map(
                lambda x: tf.cast(
                    tf.math.greater(x, binarize_threshold), dtype=tf.float32
                    ),
                num_parallel_calls=tf.data.AUTOTUNE
                )

        return patches_set

