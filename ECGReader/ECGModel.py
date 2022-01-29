import os
from sys import argv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import metrics
import segmentation_models as sm
from matplotlib import pyplot as plt
import imgaug
from tqdm import tqdm
from ECGDataset import ECGDataset


_string_mult = 100


class ECGModel(object):

    def __init__(self, train_ecgs, train_masks, test_ecgs, test_masks,
                 val_ecgs, val_masks, from_saved=False, saved_model_path=None):
        """
        Creates an ECGModel object, containing the patch shape for ecg,
        two dictionaries of ECGDataset objects for ecgs and masks, with keys `train`,
        `test`, and `set`, and three tf.data.Dataset containing ecg and masks patches
        for train test and validation. Also contains a list of callbacks for a keras
        model, the keras model, weights for the loss function and a history attribute.

        Parameters
        ----------
        train_ecgs : ECGDataset
            Contains images and patches for the training set.

        train_masks : ECGDataset
            Contains images and patches for the ground truth of the training set.

        test_ecgs : ECGDataset
            Contains images and patches for the test set.

        test_masks : ECGDataset
            Contains images and patches for the ground truth of the test set.

        val_ecgs : ECGDataset
            Contains images and patches for the validation set.

        val_masks : ECGDataset
            Contains images and patches for the ground truth of the validation set.

        from_saved : bool = False
            Flag, if true loads the model from a saved tensorflow model instead of
            creating it from scratch.

        saved_model_path : string
            The path of the tensorflow model to be loaded.

        """

        self.patch_shape = train_ecgs.patch_shape
        self.img_batch_size = train_ecgs.patches_set.element_spec.shape[0]
        
        self.ecg_sets = {train_ecgs: "train", test_ecgs: "test", val_ecgs: "validation"}
        self.mask_sets = {train_masks: "train", test_masks: "test", val_masks:
            "validation"}

        self.weights = self._computeWeights(train_masks.patches_set)

        self.train_set = tf.data.Dataset.zip(
            (
                train_ecgs.patches_set,
                train_masks.patches_set
                # If weights are to be passed to model.fit() instead that to the loss
                # function. This is the case when using losses from keras.losses
                # train_masks.patches_set.map(
                #     lambda x: tf.math.add(
                #         (1 - x) * (1 - self.weights), x * self.weights
                #         )
                #     )
                )
            )
        self.test_set = tf.data.Dataset.zip(
            (test_ecgs.patches_set, test_masks.patches_set)
            )
        self.val_set = tf.data.Dataset.zip(
            (val_ecgs.patches_set, val_masks.patches_set)
            )

        self.callbacks = []

        sm.set_framework('tf.keras')
        sm.framework()

        if from_saved:
            self.model = keras.models.load_model(saved_model_path)
        else:
            self.model = self._getModel()

        self._compileModel(from_saved)

        self.histories = []
        self.val_frequencies = []


    def _getModel(self):
        """
        Returns a tf.keras.Model object resembling a UNet architecture.

        Returns
        -------
        : tf.keras.Model
            The model, to be compiled.

        """
        model = sm.Unet(
            input_shape=(self.patch_shape[0], self.patch_shape[1], 3),
            encoder_weights=None, decoder_filters=(256, 128, 64, 32, 16)
            )

        return model


    def _computeWeights(self, patches):
        """
        Compute class weights based on a tf.data.Dataset object contents.

        Parameters
        ----------
        patches : tf.data.Dataset
            A Dataset containing binary patches (or images) indicating the class of the
            corresponding pixels.

        Returns
        -------
        : float
            The weights of the class to be identified, to be passed to the loss function.

        """
        print('-' * _string_mult)
        print('Computing weights')

        mask_pixel_mean = 0
        mask_count = 0

        # Computes the mean number of "1" pixels over the patches in the dataset
        for patch in tqdm(patches.take(-1)):
            mask_pixel_mean += tf.reduce_sum(patch).numpy() / tf.size(
                patch, out_type=tf.int64
                ).numpy()
            mask_count += 1

        print(f'Weight: {1 - (mask_pixel_mean / mask_count)}')
        print('-' * _string_mult)
        return 1 - (mask_pixel_mean / mask_count)


    def _compileModel(self):
        """
            Clears the keras session and compiles a model with Adam optimizer,
            Dice loss and Precision and Recall as metrics.

        Returns
        -------
        None

        """
        keras.backend.clear_session()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=sm.losses.DiceLoss(class_weights=[self.weights]),
            metrics=[metrics.MeanSquaredError()]
            )

        self.callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    'unet_model', save_best_only=True
                    )
            )


    def fitModel(self, epochs=1, learning_rate=1e-3, validation_frequency=1,
                 batch_size=None):
        """
            Sets the learning rate for the model and calls the fit function,
            saving history results in the history attribute.

        Parameters
        ----------
        epochs : int = 1
            The number of epochs of training.

        learning_rate : float = 1e-3
            The learning rate for the optimizer.

        validation_frequency : int = 1
            Number of training epochs before performing validation.

        batch_size : int = None
            Size of a training batch

        Returns
        -------
        None

        """

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

        # If batch_size is None uses the Dataset as is, otherwise
        if isinstance(batch_size, type(None)):
            history = self.model.fit(
                self.train_set, epochs=epochs, callbacks=self.callbacks, shuffle=True,
                validation_data=self.val_set, validation_freq=validation_frequency
                )
        else:
            history = self.model.fit(
                self.train_set.unbatch().batch(batch_size, drop_remainder=True),
                epochs=epochs, callbacks=self.callbacks, shuffle=True,
                validation_data=self.val_set, validation_freq=validation_frequency
                )

        self.histories.append(history)
        self.val_frequencies.append(validation_frequency)


    def evaluateAndVisualize(self, visualize=True, save=False, save_path=None):
        """
        Calls the evaluate method on the test_set and visualize five random patches
        from each ECGDataset, showing the ECG's patch, the predicted patch and the mask's
        patch.
p
        Parameters
        ----------
        visualize : bool = True
            Flag, indicates if patches have to be visualized or if the method just has
            to carry out evaluation.

        save : bool = False
            Flag, indicates if figures has to be saved or just to be plotted.

        save_path : string
            The path where figures are saved.

        Returns
        -------

        """

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
                    plt.savefig(save_path+f'/figure_number_{figure_number}')
                    figure_number += 1


    def visualizePatch(self, ecg_number, patch_number):
        """
        Show a figure for a single patch.

        Parameters
        ----------
        ecg_number : int
            The index of the ECG in the dataset.

        patch_number : int
            The patch number in the ECG

        Returns
        -------
        None

        """

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


    def visualizeSingleTrainingHistory(self, save=False, save_path=None):
        """
        Visualize the training history for the last training session of the model.

        Parameters
        ----------
        save : bool = True
            Flag, indicates if the results have to be saved instead of only shown.

        save_path : string
            The path where the figure is saved.

        Returns
        -------
        : None

        """

        loss = self.histories[-1].history['loss'][-1]
        mse = self.histories[-1].history['mean_squared_error'][-1]
        print(f'Loss: {loss}')
        print(f'Mean Squared Error: {mse}')

        plot_loss = self.histories[-1].history['loss']
        plot_mse = self.histories[-1].history['mean_squared_error']

        plot_val_loss = self.histories[-1].history['val_loss']
        plot_val_mse = self.histories[-1].history['val_mean_squared_error']

        epoch_n = self.histories[-1].epoch[-1] + 1

        historyPlot(
            plot_loss, plot_val_loss, 'loss', range(0, epoch_n, self.val_frequencies[-1]),
            save, save_path
            )
        historyPlot(
            plot_mse, plot_val_mse, 'mean squared error',
            range(0, epoch_n, self.val_frequencies[-1]), save, save_path
            )


    def visualizeHistory(self, save=False, save_path=None):
        """
        Visualizes overall training history when calling fitModel multiple times.

        Parameters
        ----------
        save : bool = False
            Flag, indicates if the resulting plots have to be saved or not.

        save_path : string = None
            The path to the save location for the plots.

        Returns
        -------
        : None

        """
        if len(self.histories) > 1:
            loss_overall = []
            mse_overall = []
            val_loss_overall = []
            val_mse_overall = []
            epoch_val_axis = [np.array([1])]

            for val_freq, history in zip(self.val_frequencies, self.histories):
                loss_overall.append(np.array(history.history['loss']))
                mse_overall.append(np.array(history.history['mean_squared_error']))
                val_loss_overall.append(np.array(history.history['val_loss']))
                val_mse_overall.append(
                    np.array(history.history['val_mean_squared_error'])
                    )
                epoch_val_axis.append(
                    np.arange(
                        epoch_val_axis[-1][-1] + val_freq,
                        # We'd want epoch + 1 but history.epoch starts from 0 and goes
                        # to epochs - 1, we add one more.
                        epoch_val_axis[-1][-1] + history.epoch[-1] + 2,
                        val_freq
                        )
                    )
            
            loss_overall = np.concatenate(loss_overall)
            mse_overall = np.concatenate(mse_overall)
            val_loss_overall = np.concatenate(val_loss_overall)
            val_mse_overall = np.concatenate(val_mse_overall)
            epoch_val_axis = np.concatenate(epoch_val_axis)[1:] - 1

            historyPlot(
                loss_overall, val_loss_overall, 'loss', epoch_val_axis, save,
                save_path
                )
            historyPlot(
                mse_overall, val_mse_overall, 'mean squared error', epoch_val_axis,
                save, save_path
                )

        else:
            self.visualizeSingleTrainingHistory(save, save_path)


def show(img, title=None):
    plt.figure()
    plt.imshow(img, cmap='gray')
    if not isinstance(title, type(None)):
        plt.title(title)
    plt.axis('off')
    plt.show()


def historyPlot(training_metric, validation_metric, name, val_frequency_array,
                save=False, save_path=None):
    """
    Plots a training metric and a validation metric on a single graphic, evaluating the
    range for the y-axis from the minimum value between the two, with a maximum value
    based on the training metrics maximum.

    Parameters
    ----------
    training_metric : List or numpy.ndarray
        The values of the tracked training metric to show

    validation_metric : List or numpy.ndarray
        The values of the tracked validation metric to show

    name : string
        The label for the y-axis

    val_frequency_array : List or numpy.ndarray
        x-axis for the validation metric

    save : bool = False
        Flag, indicates if the resulting plots have to be saved or not.

    save_path : string = None
        The path to the save location for the plots.

    Returns
    -------
    : None

    """

    plt.figure(figsize=(8, 6))
    plt.plot(training_metric, label='training')
    plt.plot(
        val_frequency_array, validation_metric, label='validation'
        )
    plt.legend()

    plt.xlabel('epochs')
    plt.ylabel(name)
    ax = plt.gca()

    if isinstance(training_metric, type(list())):
        y_min = np.array(
            [np.array(training_metric).min(), np.array(validation_metric).min()]
            ).min()
        y_max = np.array(training_metric).max()
    else: # assumes that otherwise it's a numpy.ndarray
        y_min = np.array([training_metric.min(), validation_metric.min()]).min()
        y_max = training_metric.max()

    ax.set_ylim((y_min, y_max))

    if save:
        plt.savefig(save_path + f'/history_{name}.png')


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
    t_patch_lead = 10

    # original_mask_shape = (3149, 6102)
    # original_ecg_shape = (4410, 9082)

    # We define mask and ecg overall shape based on patches parameters
    mask_patch_shape = (256, 256)
    ecg_patch_shape = (256, 256)

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
        augment_patches=True, binarize=False
        )
    train_mask_set = ECGDataset(
        mask_shape, mask_train_path, train_set_card, mask_patch_shape,
        mask_stride, mask=True
        )

    val_ecg_set = ECGDataset(
        ecg_shape, ecg_val_path, val_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, binarize=False
        )
    val_mask_set = ECGDataset(
        mask_shape, mask_val_path, val_set_card, mask_patch_shape,
        mask_stride, mask=True
        )

    test_ecg_set = ECGDataset(
        ecg_shape, ecg_test_path, test_set_card, ecg_patch_shape,
        ecg_stride, pad_horizontal=True, pad_horizontal_size=ecg_pad,
        augment_patches=True, binarize=False
        )
    test_mask_set = ECGDataset(
        mask_shape, mask_test_path, test_set_card, mask_patch_shape,
        mask_stride, mask=True
        )

    UNetModel = ECGModel(train_ecg_set, train_mask_set,
                         test_ecg_set, test_mask_set,
                         val_ecg_set, val_mask_set
                        )
