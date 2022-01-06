"""
Creates mask etc.
"""
import cv2
import SPxml
import numpy as np

# Multiplies the '-' character that separates sections of outputs
# for some functions
_string_mult = 100


def loadXML(path_to_file):
    """
    Loads a single XML
    """
    pass

def loadPNG(path_to_file):
    """
    Loads a single PNG
    """
    pass


def loadData(path_to_png, path_to_xml):
    """

    """
    # Reads folders

    # Check if there are files in the unmatched folders

    # Loops over file calling loadPNG loadXML 
    # ! Remember to use sorted on both and check if they are equally ordered
    # or keep only file name without extension and load file adding it
    #############
    # Need to handle unmatched in another way!
    #############

    # return array of png and array of xml
    # images[number_of_matches, number_of_channels, width, height]
    # ecg_leads[number_of_matches, number_of_leads, number_of_time_samples]
    # unmatched_images
    # unmatched_ecg_leads
    pass


def preprocessData(images, ecg_leads, unmatched_images=None, 
                unmatched_ecg_leads=None, rgb_to_grey=False):
    """

    """
    # Remember to handle unmatched not None

    # Reacts to rgb_to_grey

    # Computes parameters to create waveform masks
    # Creates waveform masks from leads
    # ecg_leads[:, :, :] -> ecg_leads[:, :, :, height]
    
    # Process image per image with waveform mask
    # (for comparison)
    # waveform mask has to be transformed in different ways
    # check article / code
    
    # Find best match


    pass
