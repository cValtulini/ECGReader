"""
Defines the main utilities for loading the ECGDataset
"""
import tensorflow as tf
from ECGPreProcessing import createAugmenter


_string_mult = 100


class ECGDataset(object):

    def __init__(self, img_gen, img_shape, path_to_img, n_images, patch_shape,
                 stride_shape, batch_size=None, seed=42, pad_horizontal=False,
                 pad_horizontal_size=None, augment_patches=False, grayscale=True,
                 color_invert=True, one_hot_encode=True, binarize_threshold=1e-5):
        """

        Parameters
        ----------
        img_gen
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
        grayscale
        color_invert
        one_hot_encode
        binarize_threshold
        """

        self.shape = img_shape
        self.n_images = n_images

        self.data_set = self._loadDataset(img_gen, path_to_img, seed)

        self.patch_shape = patch_shape
        self.stride_shape = stride_shape

        self.patches_set = self._createPatchesSet(pad_horizontal, pad_horizontal_size,
                                                  augment_patches, grayscale, batch_size,
                                                  color_invert, one_hot_encode,
                                                  binarize_threshold)


    def _loadDataset(self, img_gen, path_to_img, seed):
        """

        Parameters
        ----------
        img_gen
        path_to_img
        seed

        Returns
        -------

        """
        # We need images.shape and images.dtype as parameters of
        # `tf.data.Dataset.from_generator`
        images = next(
            img_gen.flow_from_directory(
                path_to_img, target_size=self.shape, class_mode=None, seed=seed,
                batch_size=1
                )
            )

        print('-' * _string_mult)
        print(f'Loading dataset from {path_to_img}:')

        spec = tf.TensorSpec(images.shape, dtype=images.dtype)

        # Creating the data set from the ImageDataGenerator object.
        data_set = tf.data.Dataset.from_generator(
            lambda: img_gen.flow_from_directory(
                path_to_img, target_size=self.shape, class_mode=None, seed=seed,
                batch_size=1
                ),
            output_signature=spec
            )

        data_set = data_set.apply(
            tf.data.experimental.assert_cardinality(self.n_images)
            )

        print('Loaded.')
        print('-' * _string_mult)

        return data_set


    def _createPatchesSet(self, pad_horizontal, pad_horizontal_size, augment_patches,
                          grayscale, batch_size, color_invert, one_hot_encode,
                          binarize_threshold):
        """

        Parameters
        ----------
        pad_horizontal
        pad_horizontal_size
        augment_patches
        grayscale
        color_invert
        one_hot_encode
        binarize_threshold

        Returns
        -------

        """
        patches_set = self.data_set

        if color_invert:
            patches_set = patches_set.map(lambda x: 255.0 - x)

        if pad_horizontal:
            if isinstance(pad_horizontal_size, type(None)):
                print('Padding requested but no pad size set. Padding not added.')
            else:
                patches_set = patches_set.map(
                    lambda x: tf.image.pad_to_bounding_box(
                        x, pad_horizontal_size, 0,
                        self.shape[0] + 2 * pad_horizontal_size, self.shape[1]
                        )
                    )

        patches_set = patches_set.map(
            lambda x: tf.reshape(
                tf.image.extract_patches(
                    x, sizes=[1, self.patch_shape[0], self.patch_shape[1], 1],
                    strides=[1, self.stride_shape[0], self.stride_shape[1], 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                    ),
                [-1, self.patch_shape[0], self.patch_shape[1], 3]
                )
            )

        self.patches_per_image = patches_set.element_spec.shape[0]

        if isinstance(batch_size, type(None)):
            self.batch_size = self.patches_per_image // 2
        else:
            self.batch_size = batch_size

        patches_set = patches_set.unbatch()
        patches_set = patches_set.batch(self.batch_size)

        self.n_patches = self.n_images * self.patches_per_image

        patches_set = patches_set.apply(
            tf.data.experimental.assert_cardinality(self.n_patches)
            )

        if augment_patches:
            in_type = patches_set.element_spec.dtype
            in_shape = patches_set.element_spec.shape

            augmenter = createAugmenter()
            patches_set = patches_set.map(
                lambda x: tf.numpy_function(
                    func=augmenter.augment_images, inp=[tf.cast(x, tf.uint8)],
                    Tout=tf.uint8
                    )
                )

            # Shape is lost after applying imgaug's augmentations but it's still the same
            # as the input's shape
            patches_set = patches_set.map(lambda x: tf.reshape(x, in_shape))
            patches_set = patches_set.map(lambda x: tf.cast(x, in_type))

        if grayscale:
            patches_set = patches_set.map(lambda x: tf.image.rgb_to_grayscale(x))

        patches_set = patches_set.map(lambda x: x / 255)

        if one_hot_encode:
            patches_set = patches_set.map(
                lambda x: tf.math.greater(x, binarize_threshold)
                )
            patches_set = patches_set.map(lambda x: tf.cast(x, dtype=tf.float32))

        return patches_set

