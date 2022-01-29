"""
Creates mask etc.
"""
import os
# import SPxml
import numpy as np
from matplotlib import pyplot as plt
from imgaug import augmenters as iaa


_string_mult = 100


def plotXML(tracks, fullname, sim, destination_path):
    """
    Plots a single xml given its tracks and based on the fact that is a simultaneous record or sequential.

    Parameters
    ----------
    tracks : list 
        xml tracks to plot

    fullname: string
        full path to xml file

    sim : bool
        boolean variable that tells if file is simultaneous

    destination_path: string 
        for saving the plot


    Returns
    -------
    Nothing
    """

    # This width ratio adjustment is made to match mask patches with png from pdfs patches
    gs_kw = dict(width_ratios=[48.638, 51.362])

    plt.style.use('classic')
    fig, ax = plt.subplots(
        6, 2, figsize=(100 / 2.54, 50 / 2.54), sharex=False, squeeze=False,
        gridspec_kw=gs_kw
        )
    plt.subplots_adjust(wspace=0, hspace=0)

    t = 0
    row = 6
    col = 2
    # range_min = np.array(tracks).min() - 1
    # range_max = np.array(tracks).max() + 1

    # These ranges for the amplitudes were found as the minimum and the maximum from
    # the entire dataset
    range_min = -55
    range_max = 42

    for i in range(col):
        for j in range(row):

            if i > 0:
                # major_ticks_x = np.arange(0, 2640, 100)
                # minor_ticks_x = np.arange(0, 2640, 20)
                ax[j][i].set_xlim([0, 2640])
            else:
                # major_ticks_x = np.arange(0, 2500, 100)
                # minor_ticks_x = np.arange(0, 2500, 20)
                ax[j][i].set_xlim([0, 2500])

            ax[j][i].plot(tracks[t], 'k', linewidth=1.2)
            ax[j][i].set_ylim([range_min, range_max])
            # ax[j][i].set_xticks(major_ticks_x)
            # ax[j][i].set_xticks(minor_ticks_x, minor=True)
            # major_ticks_y = np.arange(range_min, range_max, 10)
            # minor_ticks_y = np.arange(range_min, range_max, 2)
            # ax[j][i].set_yticks(major_ticks_y)
            # ax[j][i].set_yticks(minor_ticks_y, minor=True)
            # ax[j][i].grid(which='minor', alpha=0.2)
            # ax[j][i].grid(which='major', alpha=0.5)
            # ax[j][i].set_aspect(10, 'box')
            ax[j][i].spines['top'].set_visible(False)
            ax[j][i].spines['right'].set_visible(False)
            ax[j][i].spines['bottom'].set_visible(False)
            ax[j][i].spines['left'].set_visible(False)
            plt.setp(ax[j][i].get_xticklabels(), visible=False)
            plt.setp(ax[j][i].get_yticklabels(), visible=False)
            ax[j][i].get_xaxis().set_visible(False)
            ax[j][i].get_yaxis().set_visible(False)

            t += 1

    if sim:
        plt.savefig(
            destination_path + fullname.split("/")[-1].split(".")[0] + ".png", dpi=200,
            bbox_inches='tight', pad_inches=0
            )
    else:
        plt.savefig(
            destination_path + fullname.split("/")[-1].split(".")[0] + ".png", dpi=200,
            bbox_inches='tight', pad_inches=0
            )


def masksPlotterXML(source_png_path, source_xml_path, destination_path):
    """
    Plots multiple xml files from a source folder that must contain the seq or sim
    subfolder for discriminating the plotting process.

    Parameters
    ----------
    source_png_path : string
        source path for getting the file names of pdf captures
        that must contain the seq or sim subfolder

    source_xml_path: string
        source path for getting the corresponding xml files

    destination_path: string
        path for saving the plots

    Returns
    -------
    Nothing
    """

    sim = False
    xml_names = []
    seqs = sorted(
        [_.name for _ in os.scandir(source_png_path)]
        )

    for seq in seqs:
        name = seq.split(".")[0] + ".xml"
        xml_names.append(source_xml_path + name)

    if source_png_path.split("/")[-1] == "sim":
        sim = True

    for xml in xml_names:

        datas = SPxml.getLeads(xml)
        tracks = []

        if sim:
            for i in range(len(datas)):
                tracks.append(np.array(datas[i]['data']))
                tracks[i] = tracks[i][:2460] - np.around(tracks[i][:2460].mean(), 1)

        else:
            for i in range(len(datas)):
                tracks.append(np.array(datas[i]['data']))
                if i < 6:
                    tracks[i] = tracks[i][0:2500] - np.around(tracks[i][0:2500].mean(), 1)
                else:
                    tracks[i] = tracks[i][2500:5000] - np.around(
                        tracks[i][2500:5000].mean(), 1
                        )

        plotXML(tracks, xml, sim, destination_path)


def createAugmenter():
    """
    Create an augmenter to add noise to images.

    Returns
    -------
    augmenter : imgaug.augmenters.Augmenter

    """
    augmenter = iaa.SomeOf(
        (1, None), [
                iaa.OneOf(
                    [
                            iaa.Add([-50, 50]),
                            iaa.Multiply((0.6, 1.4))
                            ]
                    ),
                iaa.OneOf(
                    [
                            iaa.OneOf(
                                [
                                        iaa.AdditiveGaussianNoise(
                                            scale=(0, 0.2 * 255)
                                            ),
                                        iaa.SaltAndPepper((0.01, 0.2))
                                        ]
                                ),
                            iaa.GaussianBlur(sigma=(0.01, 1.0))
                            ]
                    ),
                iaa.imgcorruptlike.DefocusBlur(severity=1),
                iaa.imgcorruptlike.Saturate(severity=1)
                ]
        )
    return augmenter
