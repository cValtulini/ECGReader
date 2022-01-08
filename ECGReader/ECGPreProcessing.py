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

    ecg = SPxml.getLeads(path_to_file)
    return np.array([ecg[i]['data'] for i in range(len(ecg))])#[:, :5450]


def loadPNG(path_to_file):
    """
    Loads a single PNG
    """

    image = cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)

    return 1 - (image / 255)


def loadMatches(file_names, path_to_png_matches, path_to_xml_matches):
    """
    Loads file in matches folder, requires a list of file names without 
    file extensions and the paths for the two folders for the files
    to match.
    """

    #return pngs, xmls generator
    for file in file_names:
        png_file = f'{path_to_png_matches}/{file}.png'
        xml_file = f'{path_to_xml_matches}/{file}.xml'

        yield (loadPNG(png_file), loadXML(xml_file))

"""
def loadUnmatched(path_to_files, extension):
    """ """
    !!! Input is a list of paths, not names as in matches
    """ """
    
    assert extension == 'png' or extension == 'xml', 'Extension not recognised'

    if extension == 'png':
        return np.stack([loadPNG(png) for png in path_to_files], axis=0)
    elif extension == 'xml':
        return np.stack([loadXML(xml) for xml in path_to_files], axis=0)
"""


# XML Loading is fast we could load them just to find these params
def findMaxRange(xml_array):
    """
    
    """

    return (xml_array.max(axis=-1) - xml_array.min(axis=-1)).max()


def loadData(path_to_png, path_to_xml):
    """
    Expect the two paths to only contain folders `matches` and `unmatched`, and at least `matches` to contain elements
    Returns a list containing the proper number of arrays based on folders and folders content ordered as png_matches xml_matches 
    """

    assert sorted([_.name for _ in os.scandir(path_to_png) if _.is_dir()]) == sorted(['matches', 'unmatched']), 'PNG folder content is not as expected.'
    assert sorted([_.name for _ in os.scandir(path_to_xml) if _.is_dir()]) == sorted(['matches', 'unmatched']), 'XML folder content is not as expected.'

    png_matches_path = f'{path_to_png}/matches'
    xml_matches_path = f'{path_to_xml}/matches'
    #png_unmatched_path = f'{path_to_png}/unmatched'
    #xml_unmatched_path = f'{path_to_xml}/unmatched'
    #_existing_unmatched_png=False
    #_existing_unmatched_xml=False

    # Check if there are files in the unmatched folders
    """
    if len([_ for _ in os.scandir(png_unmatched_path) if _.is_file()]):
        _existing_unmatched_png = True
    if len([_ for _ in os.scandir(xml_unmatched_path) if _.is_file()]):
        _existing_unmatched_xml = True
    """

    # Reads matches folders
    png_matches = sorted(
        [_.name.split('.')[0] for _ in os.scandir(png_matches_path)
        if len(_.name.split('.')) == 2 and _.name.split('.')[1] == 'png'])
    xml_matches = sorted(
        [_.name.split('.')[0] for _ in os.scandir(xml_matches_path)
        if len(_.name.split('.')) == 2 and _.name.split('.')[1] == 'xml'])

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

    data = dict()

    # matched_pngs, matched_xmls = 
    matches_generator = loadMatches(matches,
        png_matches_path,
        xml_matches_path
        )
    data.update({"matches": matches_generator})
    #data.update({"matches_png_files":matched_pngs})
    #data.update({"matches_xml_files":matched_xmls})

    """
    # Load unmatched files if it's the case to do so
    if _existing_unmatched_png:
        png_unmatched = sorted(
            [_.path for _ in os.scandir(png_unmatched_path)
            if len(_.name.split('.')) == 2 and _.name.split('.')[1] == 'png']
            )
        #data.append(loadUnmatched(png_unmatched, 'png'))
        data.update({"unmatched_png_files":loadUnmatched(png_unmatched, 'png')})
    if _existing_unmatched_xml:
        xml_unmatched = sorted(
            [_.path for _ in os.scandir(xml_unmatched_path)
            if len(_.name.split('.')) == 2 and _.name.split('.')[1] == 'xml']
            )
        #data.append(loadUnmatched(xml_unmatched, 'xml'))
        data.update({"unmatched_xml_files":loadUnmatched(xml_unmatched, 'xml')})
    """

    print('-' * _string_mult)
    print('Completed.')

    matches_number = len(matches)

    
    ecg_max_lead_range = findMaxRange(
        np.stack(
            [loadXML(f'{png_matches_path}/{file}.xml') for file in matches],
            axis=0)
            )

    return data, matches_number, ecg_max_lead_range 


# Should we return a 0-255 (?)
def createWaveformMask(ecg_lead, span):
    """
    
    """
    # Creates the mask image background with width = the number of time samples
    # and height = span*10 since we know the resolution is 0.1 -> 1/0.1 is the 
    # expansion factor we need to map values to integers (pixel's positions)
    mask = np.zeros(shape=(int(span*10), ecg_lead.shape[-1]), dtype=np.uint8)

    # Creates the vector of indexes to map the digital data onto the image 
    # (mask)
    # We have a range of values going from negative to positive, but in 
    # the image the top left position is 0 and indexes grow moving towards the 
    # bottom left -> we subtract out lead's maximum value to map it to 0 and # invert the sign, we then multiply by the expansion coefficient and cast 
    # to int for indexing 
    indexes = (-(ecg_lead-ecg_lead.max())*10).astype(np.uint16)

    # index over the computed indexes coupled with a "time axis" to set to 1 
    # pixel corresponding to waveform points
    mask[indexes, np.arange(0, ecg_lead.shape[-1])] = 1

    return mask


#, unmatched_images=None, unmatched_ecg_leads=None):
def findECGBestMatch(matches):
    """

    """

    # Computes parameters to create waveform masks
    

        # Creates waveform masks from leads
        
        # waveform mask has to be transformed in different ways
        
        # Compare image per image with waveform mask(s)
        
        # Find best match


    pass
