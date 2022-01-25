"""
Defines the main utilities for loading the ECGDataset
"""
import tensorflow as tf
from ECGPreProcessing import createAugmenter


_string_mult = 100


class ECGDataset(object):

    def __init__(self, img_shape, path_to_img, n_images, patch_shape,
                 stride_shape, mask=False, batch_size=None, seed=42, pad_horizontal=False,
                 pad_horizontal_size=None, augment_patches=False, color_invert=True,
                 binarize=True, binarize_threshold=1e-6):
        """
        Initializes an ECGDataset object. Loads a tf.data.Dataset object into the
        `data_set` attribute and then divides it into patches in `patches_set`.

        Parameters
        ----------
        img_shape : Tuple[int]
            The shape of one image in the dataset.

        path_to_img : string
            Path to the images folder.

        n_images : int
            Total number of images in the folder.

        patch_shape : Tuple[int]
            Shape of patches into which images are divided.

        stride_shape : Tuple[int]
            Strides into the height and width directions when dividing images into
            patches.

        mask : bool = False

        batch_size : int = None
            Size of a batch of `patches_set` if None a batch will be formed by the number
            of patches obtained by an image.

        seed : int = 42
            The seed for operation involving randomness applied on the datasets (
            loading and augmentation).

        pad_horizontal : bool = False
            Flag to indicate if the images are to be padded in height before creating
            patches.

        pad_horizontal_size : int = None
            The amount of pixel to add as padding if pad_horizontal is True. the same
            amount will be added on the top and on the bottom of the image.

        augment_patches : bool = False
            Flag to indicate if augmentation has to be applied after patches have been
            divided.

        color_invert : bool = True
            Flag to indicate if patches need to be color inverted or not.

        binarize : bool = True
            Flag to indicate whether patches have to be binarized or not. After
            binarization patches are casted to float.

        binarize_threshold : float = 1e-6
            Threshold to identify a pixel as True

        """

        self.shape = img_shape
        self.n_images = n_images

        self.data_set = self._loadDataset(path_to_img, mask, seed)

        self.patch_shape = patch_shape
        self.stride_shape = stride_shape

        self.patches_set = self._createPatchesSet(mask, pad_horizontal,
                                                  pad_horizontal_size, augment_patches,
                                                  batch_size, color_invert, binarize,
                                                  binarize_threshold)


    def _loadDataset(self, path_to_img, mask, seed):
        """
        Creates a tf.data.Dataset object loading images from path_to_img

        Parameters
        ----------
        path_to_img : string
            The location of images to be loaded

        seed : int
            The seed for image shuffling

        Returns
        -------
        data_set : tf.data.Dataset
            A dataset containing the images.

        """

        print('-' * _string_mult)
        print(f'Loading dataset from {path_to_img}:')

        # spec = tf.TensorSpec(images.shape, dtype=images.dtype)

        # Creating the data set from the ImageDataGenerator object.
        color = 'grayscale' if map else 'rgb'
        data_set = tf.keras.utils.image_dataset_from_directory(
            path_to_img, labels=None, label_mode=None, color_mode=color,
            batch_size=1, image_size=self.shape, shuffle=False, seed=seed
            )

        data_set = data_set.map(
            lambda x: tf.cast(x, dtype=tf.uint8),
            num_parallel_calls=tf.data.AUTOTUNE
            )

        # Executed otherwise the first dimension of the tensor is undetermined and
        # following operation will launch exceptions
        data_set = data_set.unbatch()
        data_set = data_set.batch(1, drop_remainder=True)

        print('Loaded.')
        print('-' * _string_mult)

        return data_set


    def _createPatchesSet(self, mask, pad_horizontal, pad_horizontal_size,
                          augment_patches,
                          batch_size, color_invert, binarize, binarize_threshold):
        """
        Creates a tf.data.Dataset of patches from self.data_set.

        Parameters
        ----------
        mask : bool


        pad_horizontal : bool = False
            Flag to indicate if the images are to be padded in height before creating
            patches.

        pad_horizontal_size : int = None
            The amount of pixel to add as padding if pad_horizontal is True. the same
            amount will be added on the top and on the bottom of the image.

        augment_patches : bool = False
            Flag to indicate if augmentation has to be applied after patches have been
            divided.

        color_invert : bool = True
            Flag to indicate if patches need to be color inverted or not.

        binarize : bool = True
            Flag to indicate whether patches have to be binarized or not. After
            binarization patches are casted to float.

        binarize_threshold : float = 1e-6
            Threshold to identify a pixel as True

        Returns
        -------
        patches_set : tf.data.Dataset

        """

        # Copies the data_set
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

        channels = 1 if map else 3
        patches_set = patches_set.map(
            lambda x: tf.reshape(
                tf.image.extract_patches(
                    x, sizes=[1, self.patch_shape[0], self.patch_shape[1], 1],
                    strides=[1, self.stride_shape[0], self.stride_shape[1], 1],
                    rates=[1, 1, 1, 1],
                    padding='VALID'
                    ),
                [-1, self.patch_shape[0], self.patch_shape[1], channels]
                ),
            num_parallel_calls=tf.data.AUTOTUNE
            )

        # Saves the number of elements in a batch, i.e. the number of patches in an image
        self.patches_per_image = patches_set.element_spec.shape[0]

        if not isinstance(batch_size, type(None)):
            self.batch_size = batch_size
            patches_set = patches_set.unbatch()
            patches_set = patches_set.batch(self.batch_size, drop_remainder=True)

        self.n_patches = self.n_images * self.patches_per_image

        if augment_patches:
            # After the numpy function the shape of an element of the dataset is
            # unknown, we save it to restore it later
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

        # Normalizes images to 0.0 - 1.0
        patches_set = patches_set.map(
            lambda x: tf.cast(x, dtype=tf.float32) / 255.0,
            num_parallel_calls=tf.data.AUTOTUNE
            )

        # Computes the mask from mask patches: since the image is downsampled we don't
        # have values equal to 0 or 255 anymore, then we create the binary image
        # setting a threshold and then casting to float.
        if binarize:
            patches_set = patches_set.map(
                lambda x: tf.cast(
                    tf.math.greater(x, binarize_threshold), dtype=tf.float32
                    ),
                num_parallel_calls=tf.data.AUTOTUNE
                )

        return patches_set

