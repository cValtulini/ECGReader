"""
Creates mask etc.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm


# Multiplies the '-' character that separates sections of outputs
_string_mult = 100


def loadPNG(path_to_file, binary=False):
    """
    loadPNG loads a png with opencv `imread` given the path.
    
    Parameters
    ----------
    path_to_file: String
        The path to the image to be loaded, comprehensive of filename and extension.
        
    binary: bool
        If True loads a binary image without any changes; if False loads a color image 
        converting it to grayscale and color inverting it (255 - grayscale)
        Default to False

    Returns
    -------
    : numpy.ndarray
        A numpy array representing the image

    """

    if binary:
        return cv2.imread(path_to_file, cv2.IMREAD_UNCHANGED)
    else:
        return 255 - cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)


def loadPNGsFromFolder(file_paths):
    """
    loadPNGsFromFolder returns an iterator yielding PNG images of the ECGs, grayscale
    and color inverted.

    Parameters
    ----------
    file_paths:
        List of PNG files paths, comprehensive of filename and extension

    Returns
    -------
    : generator
        A generator yielding PNG files

    """
    """
    Loads file in matches folder, requires a list of file names without file extensions
    and the paths for the two folders for the files to match.
    """

    for file in file_paths:
        yield loadPNG(file)


def loadData(path_to_png):
    """
    loadData returns a dictionary containing generators of PNG files and a list of sorted
    file paths

    Parameters
    ----------
    path_to_png: String
        The folder to the path expected to contain sub-folders `matches` and
        `unmatched` containing (at least matches) PNG files.
        Note: the current version just returns files from the matches folder

    Returns
    -------
    data: Dictionary
        Containing keys `matches` and `unmatched` indexing the generators for the
        corresponding folder's sorted PNG files
    png_matches: List[String]
        The list of PNG file's paths. comprehensive of filename and extension

    """
    # TODO: Implement loading from the `unmatched` folder

    assert sorted([_.name for _ in os.scandir(path_to_png) if _.is_dir()]) == sorted([
            'matches', 'unmatched']), 'PNG folder content is not as expected.'

    png_matches_path = f'{path_to_png}/matches'

    # Reads `matches` folders
    png_matches = sorted(
        [_.path for _ in os.scandir(png_matches_path)
         if len(_.name.split('.')) == 2 and _.name.split('.')[1] == 'png']
        )

    print('-' * _string_mult)
    print('Loading files...')

    data = dict()

    matches_generator = loadPNGsFromFolder(png_matches)
    data.update({"matches": matches_generator})

    print('Completed.')
    print('-' * _string_mult)

    return data, png_matches


def gridRemoval(img, std_coeff):
    """
    gridRemoval removes the grid from color inverted PNG files of ECGs based on the
    image grayscale values mean and standard deviation (multiplied by a coefficient).
    While the output binary image sum is 0 progressively reduces the coefficient
    multiplying the standard deviation.

    Parameters
    ----------
    img: numpy.ndarray, shape=(W, H)
        The grayscale image, assuming it is color inverted

    std_coeff: float
        The coefficient multiplying the standard deviation

    Returns
    -------
    : numpy.ndarray, shape=(W, H), dtype=np.uint8
        An array of 0/1 values with 0 for background and 1 for the signal

    """
    """
    
    """

    while np.sum(img > img.mean() + std_coeff * img.std()) == 0:
        std_coeff -= 0.05

    return (img > img.mean() + std_coeff * img.std()).astype(np.uint8)


def adjustTemplate(template):
    """
    adjustTemplate given an image as a numpy array resize it removing eventual border
    where
    all pixels
    on the border's line are 0.

    Parameters
    ----------
    template: numpy.ndarray
        The image to cut

    Returns
    -------
    : numpy.ndarray
        The input array without external border where all the border's line is made of
        0 values

    """

    limits = np.where(template > 0)
    return template[limits[0].min():limits[0].max() + 1, limits[1].min():limits[1].max() + 1]


def extractWaveformMasks(path_to_ecgs, path_to_templates, path_for_masks):
    """
    extractWaveformMasks saves PNG files of waveform masks extracted from PNG of ECGs
    extracted from digitally printed PDFs.

    Parameters
    ----------
    path_to_ecgs: String
        The path to the folder containing PNG images to load and from which to extract
        waveform masks, have to be divided into `matches` and `unmatched`
    path_to_templates: String
        Path to the folder containing templates of element to be removed from the
        waveform mask after grid removal
    path_for_masks: String
        Path to the folder in which to save the generated waveform mask

    Returns
    -------
    Nothing

    """
    # TODO: if needed modify to use more than just one template

    data, matches_path = loadData(path_to_ecgs)
    gen = data["matches"]

    temp = adjustTemplate(loadPNG(f'{path_to_templates}/triangle_temp.png', binary=True))

    print('-' * _string_mult)

    for match in tqdm(matches_path, desc='Creating masks'):
        # Asks the generator to return the next element and directly passes it to the
        # gridRemoval function
        mask = gridRemoval(next(gen), 5)

        # TODO: [If necessary] remove the second part completely from images and masks
        #  instead of just setting it to 0
        mask[:, -321:] = 0
        mask[:, 4338:4428] = 0

        try:
            # Looks for the template shape inside the image
            res = cv2.matchTemplate(mask[:100], temp, cv2.TM_CCORR_NORMED)

            # Get regions where the value of the correlation image is greater than a
            # certain threshold based on the maximum value of res image (which is
            # normalized)
            # TODO: [If necessary] since `res` is normalized it could be better to use
            #  a coefficient between 0 and 1 alone instead of multiplying it by res.max()
            template_locations = np.where(res > res.max() * 0.9)
            row = template_locations[0]
            col = template_locations[1]

            prev_x, prev_y = (2 * mask.shape[0], 2 * mask.shape[1])
            for x, y in zip(row, col):
                # Should avoid removing pixels unnecessarily, if it hasn't moved at
                # least two pixels far from the previous location doesn't remove pixels
                if (prev_x - x) + (prev_y - y) > 2:
                    mask[:100][x:x + temp.shape[0], y:y + temp.shape[1]] = 0

                prev_x, prev_y = (x, y)

        except Exception:
            continue

        cv2.imwrite(f'{path_for_masks}/{match}.png', mask, [cv2.IMWRITE_PNG_BILEVEL, 1])

    print('Completed.')
    print('-' * _string_mult)
