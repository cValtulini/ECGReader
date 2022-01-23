import os
from sys import argv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, metrics, losses
import segmentation_models
from matplotlib import pyplot as plt
import imgaug
from ECGDataset import ECGDataset


_string_mult = 100


class ECGModel(object):

    def __init__(self, train_ecgs, train_masks, test_ecgs, test_masks,
                 val_ecgs, val_masks, from_saved=False, saved_model_path=None):

        self.patch_shape = train_ecgs.patch_shape

        self.ecg_sets = {train_ecgs: "train", test_ecgs: "test", val_ecgs: "validation"}
        self.mask_sets = {train_masks: "train", test_masks: "test", val_masks:
            "validation"}

        self.train_set = tf.data.Dataset.zip(
            (train_ecgs.patches_set, train_masks.patches_set)
            )
        self.test_set = tf.data.Dataset.zip(
            (test_ecgs.patches_set, test_masks.patches_set)
            )
        self.val_set = tf.data.Dataset.zip(
            (val_ecgs.patches_set, val_masks.patches_set)
            )

        self.callbacks = []

        if from_saved:
            self.model = keras.models.load_model(saved_model_path)
        else:
            self.model = self._getModel()

        self.to_be_compiled = True


    def _getModel(self):
        inputs = keras.Input(shape=self.patch_shape + (1,))

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
            # crop_shape = (
            #     (previous_block_activation[-(i + 1)].shape[1] - x.shape[1]) // 2,
            #     (previous_block_activation[-(i + 1)].shape[2] - x.shape[2]) // 2
            #     )
            # residual = layers.Cropping2D(crop_shape)(
            #   previous_block_activation[-(i + 1)]
            #   )

            residual = previous_block_activation[-(i + 1)]
            x = layers.add([x, residual])  # Add back residual
            # previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        # Define the model
        return keras.Model(inputs, outputs)


    def fitModel(self, epochs=1, learning_rate=1e-3):
        if self.to_be_compiled:
            keras.backend.clear_session()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=segmentation_models.losses.DiceLoss(class_weights=[0.75]),
                #loss=losses.BinaryCrossentropy(),
                metrics=[metrics.Precision(), metrics.Recall()]
                )
            self.callbacks.append(
                    keras.callbacks.ModelCheckpoint(
                        'basic_unet.ckpt', save_best_only=True
                        )
                )
            self.to_be_compiled = False

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

        self.model.fit(
            self.train_set, epochs=epochs, callbacks=self.callbacks, shuffle=True,
            validation_data=self.val_set
            )

    def evaluateAndVisualize(self, visualize=True, save=False, save_path=None):
        if self.to_be_compiled:
            print("Model hasn't been compiled yet")
            return

        self.model.evaluate(self.test_set)

        if visualize:
            figure_number = 0
            for (ecg, mask) in self.test_set:
                predicted = self.model.predict(ecg)

                fig, ax = plt.subplots(3, 5, figsize=(50 / 2.54, 25 / 2.54))
                plt.subplots_adjust(wspace=0.05, hspace=0.05)
                for i in range(5):
                    t = np.random.randint(0, ecg.numpy().shape[0])

                    ax[0, i].imshow(ecg[t, :, :, 0], cmap='gray')
                    ax[0, i].title.set_text(f'ECG PATCH {t}')
                    ax[0, i].axis('off')

                    ax[1, i].imshow(predicted[t, :, :, 0], cmap='gray')
                    ax[1, i].title.set_text(f'PREDICTED PATCH {t}')
                    ax[1, i].axis('off')

                    ax[2, i].imshow(mask[t, :, :, 0], cmap='gray')
                    ax[2, i].title.set_text(f'MASK PATCH {t}')
                    ax[2, i].axis('off')

                plt.show()

                if save:
                    plt.savefig(save_path+f'figure_number_{figure_number}')
                    figure_number += 1


    def plotModelHistory(self):
        pass


def show(img, title=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    if not isinstance(title, type(None)):
        plt.title(title)
    plt.axis('off')
    plt.show()


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

    # original_mask_shape = (3149, 6102)
    # original_ecg_shape = (4410, 9082)

    # We define mask and ecg overall shape based on patches parameters
    mask_patch_shape = (200, 120)
    ecg_patch_shape = (200, 120)

    mask_stride = (mask_patch_shape[0], mask_patch_shape[1] // 2)
    ecg_stride = (ecg_patch_shape[0] // 2, ecg_patch_shape[1] // 2)

    mask_shape = (mask_patch_shape[0] * ecg_rows,
                  mask_stride[1] * t_patch_lead * ecg_cols)
    ecg_shape = (ecg_stride[0] * ecg_rows, ecg_stride[1] * t_patch_lead * ecg_cols)

    ecg_pad = ecg_stride[0] // 2

    train_set_card = 48
    val_set_card = 15
    test_set_card = 14

    # Creating train val test sets
    train_ecg_set = ECGDataset(
        ecg_shape, ecg_train_path, train_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, one_hot_encode=False
        )
    train_mask_set = ECGDataset(
        mask_shape, mask_train_path, train_set_card, mask_patch_shape,
        mask_stride
        )

    val_ecg_set = ECGDataset(
        ecg_shape, ecg_val_path, val_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, one_hot_encode=False
        )
    val_mask_set = ECGDataset(
        mask_shape, mask_val_path, val_set_card, mask_patch_shape,
        mask_stride
        )

    test_ecg_set = ECGDataset(
        ecg_shape, ecg_test_path, test_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, one_hot_encode=False
        )
    test_mask_set = ECGDataset(
        mask_shape, mask_test_path, test_set_card, mask_patch_shape,
        mask_stride
        )

    basicUNet = ECGModel(train_ecg_set, train_mask_set,
                         test_ecg_set, test_mask_set,
                         val_ecg_set, val_mask_set
                        )

    basicUNet.fitModel(epochs=1, learning_rate=1e-3)

    basicUNet.evaluateAndVisualize()

