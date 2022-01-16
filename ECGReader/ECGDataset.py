"""
Defines the main utilities for loading the ECGDataset
"""
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import os


_string_mult = 100


class ECGDataset(object):
    # class ECGDataset(tf.data.Dataset):

    # TODO: Should call load or something like that to load ecgs and masks, should also
    #  implement "distortion" parameters as function parameters
    def __init__(self):
        # TODO: they will be keras.preprocessing.ImageDataGenerator object(s)
        # TODO: when loading masks check if it's needed to modify their creation to use
        #  values in 0-255
        # TODO: we should definitely implement a way to divide images into patches:
        #  check cv2 or skimage
        self.ECGs= None
        #self.masks = None


    def foo(self, bar):
        pass


    def loadDataset(self):



        pass
