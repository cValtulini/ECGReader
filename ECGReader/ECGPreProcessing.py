"""
Creates mask etc.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm


# Multiplies the '-' character that separates sections of outputs
# for some functions
_string_mult = 100


def loadPNG(path_to_file, binary=False):
    """
    Loads a single PNG, returns it grayscale color-inverted or unchanged
    """

    if binary: return cv2.imread(path_to_file, cv2.IMREAD_UNCHANGED)
    else: return 255 - cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)


def loadMatches(file_paths):
    """
    Loads file in matches folder, requires a list of file names without 
    file extensions and the paths for the two folders for the files
    to match.
    """

    #return a pngs generator
    for file in file_paths:
        yield loadPNG(file)


def loadData(path_to_png):
    """
    Expect the two paths to only contain folders `matches` and `unmatched`, and at least `matches` to contain elements
    Returns a list containing the proper number of arrays based on folders and folders content ordered as png_matches xml_matches 
    """

    assert sorted([_.name for _ in os.scandir(path_to_png) if _.is_dir()]) == sorted(['matches', 'unmatched']), 'PNG folder content is not as expected.'

    png_matches_path = f'{path_to_png}/matches'

    # Reads matches folders
    png_matches = sorted(
        [_.path for _ in os.scandir(png_matches_path)
        if len(_.name.split('.')) == 2 and _.name.split('.')[1] == 'png']
        )

    print('-' * _string_mult)
    print('Loading files...')

    data = dict()

    matches_generator = loadMatches(png_matches)
    data.update({"matches": matches_generator})

    print('Completed.')
    print('-' * _string_mult)
    
    return data, png_matches


def gridRemoval(img, std_coeff):
    """
    
    """

    while np.sum(img > img.mean() + std_coeff*img.std()) == 0:
        std_coeff -= 0.05

    return (img > img.mean() + std_coeff*img.std()).astype(np.uint8)


def adjustTemplate(template):
    """
    Cut additional background from a binary template
    """

    limits = np.where(temp > 0)
    return template[limits[0].min():limits[0].max()+1,
                limits[1].min():limits[1].max()+1]


def extractWaveformMasks(path_to_ecgs, path_to_templates, path_for_masks):
    """

    N.B. : this version of the function assumes only one template
    """

    data, matches_path = loadData(path_to_ecgs)
    gen = data["matches"]

    temp = adjustTemplate(loadPNG(f'{path_to_templates}/triangle_temp.png'))

    print('-'* _string_mult)

    for match in tqdm(matches_path, desc='Creating masks'):
        mask = gridRemoval(next(gen), 5)

        #eventually this could be removed from both image and mask
        mask[:, -321:] = 0 
        mask[:, 4338:4428] = 0

        try:
            res = cv2.matchTemplate(mask[:100], temp, cv2.TM_CCORR_NORMED)
            
            template_locations = np.where(res > res.max()*0.9)
            row = template_locations[0]
            col = template_locations[1]
            
            prev_x, prev_y = (10000, 10000)
            for x, y in zip(row, col):
                # Should avoid removing pixels unnecessarily, we haven't move
                # at least two pixels far from the previous location doesn't 
                # remove pixels
                if (prev_x-x) + (prev_y-y) > 2:
                    mask[:100][x:x+temp.shape[0], y:y+temp.shape[1]] = 0
                
                prev_x, prev_y = (x, y)

        except Exception:
            continue

        cv2.imwrite(f'{path_for_masks}/{match}.png', mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
