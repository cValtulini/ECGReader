"""
Creates mask etc.
"""
import os
import cv2
import SPxml
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


# Multiplies the '-' character that separates sections of outputs
_string_mult = 100


def loadXML(path_to_file):
    """
    Loads a single XML
    """

    ecg = SPxml.getLeads(path_to_file)
    return np.array([ecg[i]['data'] for i in range(len(ecg))])


def loadPNG(path_to_file, mask=False):
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

    if mask:
        return cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)
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


def extractWaveformMasks(path_to_xml, path_to_save):
    """
    extractWaveformMasks saves PNG files of waveform masks extracted from PNG of ECGs
    extracted from digitally printed PDFs.

    Parameters
    ----------
    path_to_xml :

    path_to_save :


    Returns
    -------
    Nothing

    """

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


def plotXML(tracks,fullname,sim,destination_path):  
    """
    Plots a single xml given its tracks and based on the fact that is a simultaneous record or sequential.

    Parameters
    ----------
    tracks : xml tracks to plot

    fullname: full path to xml file

    sim : boolean variable that tells if file is simultaneous

    destination_path: for saving the plot


    Returns
    -------
    Nothing
    """
    plt.style.use('classic')
    fig, ax = plt.subplots(6, 2, figsize=(100/2.54, 50/2.54))
    plt.subplots_adjust(wspace=0, hspace=0)

    t = 0
    row = 6
    col = 2
    # range_min = np.array(tracks).min() - 1
    # range_max = np.array(tracks).max() + 1
    range_min=-95.0
    range_max=88.0
    #major_ticks_x = np.arange(0, 2640, 100)
    #minor_ticks_x = np.arange(0, 2500, 20)

    for i in range(col):
        for j in range(row):
            
            if i>0:
                major_ticks_x = np.arange(0, 2660, 100)
                minor_ticks_x = np.arange(0, 2660, 20)
            else:
                major_ticks_x = np.arange(0, 2500, 100)
                minor_ticks_x = np.arange(0, 2500, 20)
            
            ax[j][i].plot(tracks[t],'k', linewidth=1.2,)
            ax[j][i].set_xlim([0, 2500])
            ax[j][i].set_ylim([range_min, range_max])
            ax[j][i].set_xticks(major_ticks_x)
            ax[j][i].set_xticks(minor_ticks_x, minor=True)
            #major_ticks_y = np.arange(range_min, range_max, 10)
            #minor_ticks_y = np.arange(range_min, range_max, 2)
            #ax[j][i].set_yticks(major_ticks_y)
            #ax[j][i].set_yticks(minor_ticks_y, minor=True)
            #ax[j][i].grid(which='minor', alpha=0.2)
            #ax[j][i].grid(which='major', alpha=0.5)
            #ax[j][i].set_aspect(10, 'box')

            ax[j][i].spines['top'].set_visible(False)
            ax[j][i].spines['right'].set_visible(False)
            ax[j][i].spines['bottom'].set_visible(False)
            ax[j][i].spines['left'].set_visible(False)
            plt.setp(ax[j][i].get_xticklabels(), visible=False)
            plt.setp(ax[j][i].get_yticklabels(), visible=False)
            ax[j][i].get_xaxis().set_visible(False)
            ax[j][i].get_yaxis().set_visible(False)

            t += 1
            
    if sim==True:
        plt.savefig(destination_path+fullname.split("/")[-1].split(".")[0]+".png", dpi=200, bbox_inches='tight',pad_inches=0)
    else:
        plt.savefig(destination_path+fullname.split("/")[-1].split(".")[0]+".png", dpi=200, bbox_inches='tight',pad_inches=0)


def masksPlotterXML(source_png_path,source_xml_path,destination_path):
    """
    Plots multiple xml files from a source folder that must contain the seq or sim subfolder
    for discriminating the plotting process.

    Parameters
    ----------
    source_png_path : source path for getting the file names of pdf captures

    source_xml_path: source path for getting the corresponding xml files

    destination_path: path for saving the plots

    Returns
    -------
    Nothing
    """

    sim=False
    xml_names=[]
    seqs = sorted(
            [_.name for _ in os.scandir(source_png_path)]
            )

    for seq in seqs:
        name=seq.split(".")[0]+".xml"
        xml_names.append(source_xml_path+name)
        
    if source_png_path.split("/")[-1]=="sim":
        sim=True

    for xml in xml_names:

        datas=SPxml.getLeads(xml)
        tracks = []

        if sim:
            for i in range(len(datas)):
                tracks.append(np.array(datas[i]['data']))
                tracks[i] = tracks[i][0:2460] - np.around(tracks[i].mean(), 1)

        else:
            for i in range(len(datas)):
                tracks.append(np.array(datas[i]['data']))
                if i < 6:
                    tracks[i] = tracks[i][0:2500] - np.around(tracks[i].mean(), 1)
                else:
                    tracks[i] = tracks[i][2500:5000]
                    tracks[i] -= np.around(tracks[i].mean(), 1)

        plotXML(tracks,xml,sim,destination_path)
