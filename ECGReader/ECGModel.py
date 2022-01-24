import os
from sys import argv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, metrics, losses
import segmentation_models
from matplotlib import pyplot as plt
import imgaug
from tqdm import tqdm
from ECGDataset import ECGDataset


_string_mult = 100


class ECGModel(object):

    def __init__(self, train_ecgs, train_masks, test_ecgs, test_masks,
                 val_ecgs, val_masks, from_saved=False, saved_model_path=None):
        """

        Parameters
        ----------
        train_ecgs
        train_masks
        test_ecgs
        test_masks
        val_ecgs
        val_masks
        from_saved
        saved_model_path
        """

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

        self.weights = self._computeWeights(train_masks.patches_set)

        self._compileModel()

        self.history = None


    def _getModel(self):
        """

        Returns
        -------

        """
        layers_activation = "relu"

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
        x = layers.Activation(layers_activation)(x)
        previous_block_activation = [x]
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128]:
            x = layers.Activation(layers_activation)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation(layers_activation)(x)
            x = layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            previous_block_activation.append(x)  # Set aside next residual
            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        """
        [Second half of the network: up-sampling inputs]
        """

        print(previous_block_activation)
        for i, filters in enumerate([128, 64, 32]):
            x = layers.Activation(layers_activation)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation(layers_activation)(x)
            x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D(2)(x)

            # Project residual
            residual = previous_block_activation[-(i + 1)]
            x = layers.add([x, residual])  # Add back residual
            # previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

        # Define the model
        return keras.Model(inputs, outputs)


    def _computeWeights(self, patches):
        """

        Parameters
        ----------
        patches

        Returns
        -------

        """
        print('-' * _string_mult)
        print('Computing weigths')

        mask_pixel_mean = 0
        mask_count = 0
        for patch in tqdm(patches.take(-1)):
            mask_pixel_mean += np.sum(patch.numpy()) / patch.numpy().size
            mask_count += 1

        print(f'Weight: {1 - (mask_pixel_mean / mask_count)}')
        print('-' * _string_mult)
        return 1 - (mask_pixel_mean / mask_count)


    def _compileModel(self):
        """

        Returns
        -------

        """
        keras.backend.clear_session()
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=segmentation_models.losses.DiceLoss(class_weights=[self.weights]),
            metrics=[metrics.Precision(), metrics.Recall()]
            )
        self.callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    'unet_model', save_best_only=True
                    )
            )


    def fitModel(self, epochs=1, learning_rate=1e-3, validation_frequency=1):
        """

        Parameters
        ----------
        epochs
        learning_rate
        validation_frequency

        Returns
        -------

        """

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

        self.history = self.model.fit(
            self.train_set, epochs=epochs, callbacks=self.callbacks, shuffle=True,
            validation_data=self.val_set, validation_freq=validation_frequency
            )


    def evaluateAndVisualize(self, visualize=True, save=False, save_path=None):

        self.model.evaluate(self.test_set)

        if visualize:
            figure_number = 0
            for ecg_number, (ecg, mask) in enumerate(self.test_set):
                predicted = self.model.predict(ecg)

                fig, ax = plt.subplots(3, 5, figsize=(60 / 2.54, 30 / 2.54))
                plt.subplots_adjust(wspace=0.05, hspace=0.15)
                for i in range(5):
                    t = np.random.randint(0, ecg.numpy().shape[0])

                    ax[0, i].imshow(ecg[t, :, :, 0], cmap='gray')
                    ax[0, i].title.set_text(f'ECG {ecg_number} PATCH {t}')
                    ax[0, i].axis('off')

                    ax[1, i].imshow(predicted[t, :, :, 0], cmap='gray')
                    ax[1, i].title.set_text(f'PREDICTED {ecg_number} PATCH {t}')
                    ax[1, i].axis('off')

                    ax[2, i].imshow(mask[t, :, :, 0], cmap='gray')
                    ax[2, i].title.set_text(f'MASK {ecg_number} PATCH {t}')
                    ax[2, i].axis('off')

                plt.show()

                if save:
                    plt.savefig(save_path+f'figure_number_{figure_number}')
                    figure_number += 1


    def visualizePatch(self, ecg_number, patch_number):

        for index, (ecg, mask) in enumerate(self.test_set.take(ecg_number + 1)):
            if index == ecg_number:
                predicted = self.model.predict(ecg)

                fig, ax = plt.subplots(1, 3, figsize=(30 / 2.54, 30 / 2.54))
                plt.subplots_adjust(wspace=0.05, hspace=0.15)

                ax[0].imshow(ecg[patch_number, :, :, 0], cmap='gray')
                ax[0].title.set_text(f'ECG {ecg_number} PATCH {patch_number}')
                ax[0].axis('off')

                ax[1].imshow(predicted[patch_number, :, :, 0], cmap='gray')
                ax[1].title.set_text(f'PREDICTED {ecg_number} PATCH {patch_number}')
                ax[1].axis('off')

                ax[2].imshow(mask[patch_number, :, :, 0], cmap='gray')
                ax[2].title.set_text(f'MASK {ecg_number} PATCH {patch_number}')
                ax[2].axis('off')

                plt.show()


    def visualizeHistory(self, save=False):
        """

        Parameters
        ----------
        save

        Returns
        -------

        """

        loss = self.history.history['loss'][-1]
        acc = self.history.history['precision'][-1]
        rec = self.history.history['recall'][-1]
        print(f'Loss: {loss}')
        print(f'Accuracy: {acc}')
        print(f'Recall: {rec}')

        df = pd.DataFrame(
            self.history.history, index=self.history.epoch
            )  # create a pandas dataframe
        plt.figure(figsize=(8, 6))
        df.plot(ylim=(0, max(1, df.values.max())))  # plot all the metrics

        if save:
            plt.savefig('history.png')


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
    mask_patch_shape = (220, 120)
    ecg_patch_shape = (220, 120)

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

    UNetModel = ECGModel(train_ecg_set, train_mask_set,
                         test_ecg_set, test_mask_set,
                         val_ecg_set, val_mask_set
                        )
