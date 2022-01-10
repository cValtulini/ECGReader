"""
Defines the main utilities for loading the ECGDataset
"""


_string_mult = 100


class ECGDataset(object):

    # TODO: Should call load or something like that to load ecgs and masks, should also
    #  implement "distortion" parameters as function parameters
    def __init__(self, parameters):
        # TODO: they will be keras.preprocessing.ImageDataGenerator object(s)
        self.ECGs = None
        self.masks = None

    def foo(self, bar):
        pass
    
    def loadDataset(self):
        pass
