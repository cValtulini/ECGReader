"""
Creates mask etc.
"""
import os
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


def loadMatches(file_names):
    """
    Loads file in matches folder, requires a list of file names without 
    file extensions
    """

    # return array of png and array of xml
    # images[number_of_matches, number_of_channels, width, height]
    # ecg_leads[number_of_matches, number_of_leads, number_of_time_samples]
    pass


def loadUnmatched(path_to_files, extension):
    """
    !!! Input is a list of paths, not names as in matches
    """
    
    assert extension == 'png' or extension == 'xml', 'Extension not recognised'

    if extension == 'png':
        pass
    elif extension == 'xml':
        pass


def loadData(path_to_png, path_to_xml):
    """
    Expect the two paths to only contain folders `matches` and `unmatched`, and at least `matches` to contain elements
    Returns a list containing the proper number of arrays based on folders and folders content ordered as png_matches xml_matches 
    """

    assert sorted([_.name for _ in os.scandir(path_to_png) if _.is_dir()]) == sorted(['matches', 'unmatched']), 'PNG folder content is not as expected.'
    assert sorted([_.name for _ in os.scandir(path_to_xml) if _.is_dir()]) == sorted(['matches', 'unmatched']), 'XML folder content is not as expected.'

    png_matches_path = f'{path_to_png}/matches'
    xml_matches_path = f'{path_to_xml}/matches'
    png_unmatched_path = f'{path_to_png}/unmatched'
    xml_unmatched_path = f'{path_to_xml}/unmatched'

    # Check if there are files in the unmatched folders
    if len([_ for _ in os.scandir(png_unmatched_path) if _.is_file()]):
        _existing_unmatched_png = True
    if len([_ for _ in os.scandir(xml_unmatched_path) if _.is_file()]):
        _existing_unmatched_xml = True
    
    # Reads matches folders
    png_matches = sorted(
        [_.name.split('.')[0] for _ in os.scandir(png_matches_path)
        if len(_.name.split('.')) == 2 and _.name.split[1] == 'png']
        )
    xml_matches = sorted(
        [_.name.split('.')[0] for _ in os.scandir(xml_matches_path)
        if len(_.name.split('.')) == 2 and _.name.split[1] == 'xml']
        )
    matches = set(png_matches).intersection(xml_matches)
    # set: png_matches - xml_matches
    unmatched_matches_png = set(png_matches).difference(xml_matches)
    # set: xml_matches - png_matches
    unmatched_matches_xml = set(xml_matches).difference(png_matches) 
    
    # Checks if there are unmatched files in the matches folders, tells the 
    # user if so
    if len(unmatched_matches_png) or len(unmatched_matches_xml):
        print('You have unmatched files in the unmatched folders!')
        print(unmatched_matches_png)
        print(unmatched_matches_xml)
        print('They\'ll be ignored')

    print('-' * _string_mult)
    print('Loading files...')

    data = []
    # Loops over file in matches calling loadPNG loadXML
    # extend() -> append() but for multiple elements
    data.extend([loadMatches(matches)])

    # Load unmatched files if it's the case to do so
    if _existing_unmatched_png:
        png_unmatched = sorted(
            [_.path for _ in os.scandir(png_unmatched_path)
            if len(_.name.split('.')) == 2 and _.name.split[1] == 'png']
            )
        data.append(loadUnmatched(png_unmatched, 'png'))
    if _existing_unmatched_xml:
        xml_unmatched = sorted(
            [_.path for _ in os.scandir(xml_unmatched_path)
            if len(_.name.split('.')) == 2 and _.name.split[1] == 'xml']
            )
        data.append(loadUnmatched(png_unmatched, 'xml'))

    print('-' * _string_mult)
    print('Completed.')

    return data


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
