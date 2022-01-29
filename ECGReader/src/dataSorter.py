"""
This module/script handles the organization of PDF and XML files in order to rename
them properly to check for correspondences and sort them into the proper folder after
the matches search.
"""
from sys import argv
import os
import random
import pdfplumber
from PyPDF2 import PdfFileWriter, PdfFileReader
import PIL
from PIL import Image
from tqdm import tqdm

# Multiplies the '-' character that separates sections of outputs for some functions
_string_mult = 100


def renameXMLFiles(path):
    """
    renameXMLFiles loops through XML files to rename them properly into the name format:
    initials_patientID_ddmmaa.xml
    Note: it assumes the initial name to have the format:
    patientID,surname,name,aaaa-mm-dd_hh-mm-ss.xml

    Parameters
    ----------
    path: String
        The path to the folder containing the xml files to rename

    Returns
    -------
    Nothing

    """

    # Creates iterator over directory content
    directory_list = os.scandir(path)

    print('-' * _string_mult)
    for file in tqdm(directory_list, desc='Renaming XML files'):
        if file.is_file():
            # Produces a list containing
            # ['patient ID', 'surname', 'name', 'aaaa-mm-dd_hh-mm-ss.xml']
            # surname and name may have the structure 'sur1_sur2' 'name1_name2'
            parse_filename = file.name.split(',')

            if len(parse_filename) != 4:
                continue

            # Creates a list containing the elements for the new filename
            new_filename = []

            name_initials = []

            date = []
            for substring in parse_filename[3].split('_')[0].split('-'):
                date.insert(0, substring)
            date[2] = date[2][2] + date[2][3]

            if date[0] == '30':
                patient_surname = parse_filename[1].upper()
                patient_name = parse_filename[2].upper()
                name_initials.append(patient_surname[0])
                name_initials.append(patient_name[0])
            else:
                for substring in parse_filename[1].split('_'):
                    if len(substring):
                        name_initials.append(substring[0].upper())
                for substring in parse_filename[2].split('_'):
                    if len(substring):
                        name_initials.append(substring[0].upper())

            new_filename.append(''.join(name_initials))
            new_filename.append(parse_filename[0].upper())
            new_filename.append(''.join(date))

            os.rename(file.path, path + '/' + '_'.join(new_filename) + '.xml')

        elif file.is_dir():
            print(f'Found subdirectory: {file.path}')
            print(f'Ignored.')

    print('Completed.')
    print('-' * _string_mult)


def pdfPatientIDExtractor(path):
    """
    padPatientIDExtractor extracts patient codes from pdf files of ECGs.

    Parameters
    ----------
    path: String
        Full file path, comprehending file extension `.pdf`

    Returns
    -------
    : String
        The patient ID extracted from the file

    """

    with pdfplumber.open(path) as pdf:
        first_page = pdf.pages[0]

        # Returns a list of dictionaries with various parameters, including the extracted
        # text
        list_pdf = first_page.extract_words(vertical_ttb=False)

        for dictionary in list_pdf:
            # The patient code is found in the dictionary with a specific value of the
            # `bottom` key
            if dictionary['bottom'] == 793.0860326260371:
                return dictionary['text']

        print(f'patient ID not found for: {path}')
        return 'NotFound' + str(random.randint(0, 1000))


def multipagesPdfPatientIDNameExtractor(path,
                                        code_bottom=793.0860326260371,
                                        break_bottom=366.121624373963):
    """
    multipagesPdfPatientIDNameExtractor patientIDs and names from pdf files of ECG
    composed of many pages

    Parameters
    ----------
    path: String
        The path to the pdf file
    code_bottom: float
        Coordinates at which the patientID should be found
    break_bottom: float
        pdf coordinates at which to stop looking for the patient name

    Returns
    -------
    : List[String]
        A list containing the patientIDs
    : List[String]
        A list containing the patients' names

    """

    patient_codes = []
    names = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            list_pdf = page.extract_words(vertical_ttb=False)
            code = False
            name = str()
            for dictionary in list_pdf:

                # Here we look for the dictionary containing the patient code 
                if dictionary['bottom'] == code_bottom:
                    patient_codes.append(dictionary['text'])
                    code = True

                # Here we check if we reached the end of the name field in the pdf
                if dictionary['bottom'] == break_bottom:
                    break

                # Here we compose the patient name that will be present, if we found 
                # the patient code, in the dictionaries right after it
                if code and dictionary['bottom'] != code_bottom:
                    name = name + dictionary['text']

            # We check that the name is present in the pdf
            if name != str():
                names.append(name)

    return patient_codes, names


def renamePDFFiles(path):
    """
    renamePDFFiles renames pdf into the name format
    initials_patientID_ddmmaa.pdf
    Assumes that the directory found at `path` is divided into sub-folders named
    `ECG DATE`

    Parameters
    ----------
    path: String
        The path to the directory containing

    Returns
    -------
    Nothing

    """

    # Creates iterator over directory content
    directory_list = os.scandir(path)

    print('-' * _string_mult)
    for sub_directory in tqdm(directory_list, desc='Renaming PDF files'):
        # We know that files are organized into sub-folders identified by dates
        if sub_directory.is_dir():
            # We need to parse the pdfs in 'ECG 301221' differently as they contain
            # multiple pages
            if sub_directory.name == 'ECG 301221':
                sub_dir_list = os.scandir(sub_directory.path)
                for file in sub_dir_list:
                    # The pdf "15 ECG.pdf" has different coordinates for initials and
                    # patientID
                    if file.name == '15 ECG.pdf':
                        patient_ids, names = multipagesPdfPatientIDNameExtractor(
                            file.path, 793.0859926260371, 366.121584373963
                            )
                    else:
                        patient_ids, names = multipagesPdfPatientIDNameExtractor(
                            file.path
                            )

                    input_pdf = PdfFileReader(open(file.path, 'rb'))
                    for i in range(input_pdf.numPages):
                        # We have a file with a blank page at number 14, that is
                        # skipped as follows
                        if i < 13:
                            output = PdfFileWriter()
                            output.addPage(input_pdf.getPage(i))
                            split_name = names[i].split(',')
                            new_filename = split_name[0][0] + split_name[1][0] + '_' \
                                           + patient_ids[i] + '_' + \
                                           sub_directory.name.split(' ')[1] + '.pdf'

                            # PDFs are split and pages written in the "path" folder
                            with open(path + '/' + new_filename, 'wb') as outputStream:
                                output.write(outputStream)

                        if i > 13:
                            output = PdfFileWriter()
                            output.addPage(input_pdf.getPage(i))
                            split_name = names[i - 1].split(',')
                            new_filename = split_name[0][0] + split_name[1][0] + '_' + \
                                           patient_ids[i - 1] + '_' + \
                                           sub_directory.name.split(' ')[1] + '.pdf'
                            with open(path + '/' + new_filename, 'wb') as outputStream:
                                output.write(outputStream)

                    # We need to delete the multi pages pdfs to delete the folder later
                    os.remove(file.path)

            else:
                sub_dir_list = os.scandir(sub_directory.path)
                for file in sub_dir_list:
                    if file.is_file():
                        # Original filename is 'initials birthdate.pdf'
                        parse_filename = file.name.split(' ')

                        new_filename = [parse_filename[0].upper()]
                        new_filename.append(
                            pdfPatientIDExtractor(file.path).upper()
                            )
                        # subdirectory name is 'ECG DATE' we only keep 'DATE'
                        new_filename.append(sub_directory.name.split(' ')[1])

                        os.rename(file.path, path + '/' + '_'.join(new_filename) + '.pdf')

                    elif file.is_dir():
                        print(f'Found subdirectory: {file.path}')
                        print(f'Ignored')

            os.rmdir(sub_directory.path)

        # We ignore everything that is not a sub-folder in the main folder
        elif sub_directory.is_file():
            print(f'File found: {sub_directory.path}')
            print('Ignored')

    print('-' * _string_mult)


def matchesFinder(path_to_png, path_to_xml):
    """
    matchesFinder finds matches between file names in the two folders, excluding file
    extensions. Reorganize files into `matches` and `unmatched` sub-folders.

    Parameters
    ----------
    path_to_png: String
        The path to the folder containing PNG files
    path_to_xml: String
        The path to the folder containing XML files

    Returns
    -------
    Nothing

    """

    # List files in the two directories keeping only the filename without the extension
    png_list = [file.name.split('.')[0] for file in os.scandir(path_to_png) if
                file.is_file()]
    xml_list = [file.name.split('.')[0] for file in os.scandir(path_to_xml) if
                file.is_file()]

    print('-' * _string_mult)
    print('Finding matches:')
    # Finds the elements in both lists
    matches = set(png_list).intersection(xml_list)

    print(f'There are {len(matches)} matches in data.')
    print(f'There are {len(png_list)} png files.')
    print(f'There are {len(xml_list)} xml files.')

    # Creates folders to put matches and unmatched files into
    os.mkdir(f'{path_to_png}/matches')
    os.mkdir(f'{path_to_png}/unmatched')
    os.mkdir(f'{path_to_xml}/matches')
    os.mkdir(f'{path_to_xml}/unmatched')

    # Move matches into the proper folder
    for filename in matches:
        png_src = f'{path_to_png}/{filename}.png'
        png_dst = f'{path_to_png}/matches/{filename}.png'
        os.rename(png_src, png_dst)

        xml_src = f'{path_to_xml}/{filename}.xml'
        xml_dst = f'{path_to_xml}/matches/{filename}.xml'
        os.rename(xml_src, xml_dst)

    #  Moves unmatched files, ignores subdirectories
    for file in os.scandir(path_to_png):
        if file.is_file():
            dst = f'{path_to_png}/unmatched/{file.name}'
            os.rename(file.path, dst)
    for file in os.scandir(path_to_xml):
        if file.is_file():
            dst = f'{path_to_xml}/unmatched/{file.name}'
            os.rename(file.path, dst)

    print(
        f'{len([_ for _ in os.scandir(path_to_png) if _.is_file()])} png files '
        f'remaining in path_to_png'
        )
    print(
        f'{len([_ for _ in os.scandir(path_to_xml) if _.is_file()])} xml files '
        f'remaining in path_to_png'
        )
    print('Completed.')
    print('-' * _string_mult)


def convertPdfToPng(path_to_data):
    """
    convertPdfToPng converts the PDF files into `path_to_data/pdf` and saves them into
    `path_to_data/png`
    Expects to find a `pdf` folder inside `path_to_data`

    Parameters
    ----------
    path_to_data: String
        The folder where to find the `pdf` folder and where to put the `png` folder
        into which png files are saved

    Returns
    -------
    Nothing

    """

    out_path = f'{path_to_data}/png'

    # Gets folder's file list
    pdf_list = os.scandir(f'{path_to_data}/pdf')

    # Creates folder for png files at the same level of the pdf folder
    if not os.path.isdir(out_path):
        os.makedirs(f'{out_path}')

    print('-' * _string_mult)
    print('Converting from PDF to PNG...')

    # Opens PDF and save PNG images
    for file in pdf_list:
        with pdfplumber.open(file.path) as pdf:
            filename = file.name.split('.')[0]
            pdf.pages[0].to_image(resolution=900).save(
                f'{out_path}/{filename}.png', format='PNG'
                )

    print('Conversion completed.')
    print('-' * _string_mult)


def rotateImage(img, angle, expand=True):
    """
    rotateImage rotates an image using PIL

    Parameters
    ----------
    img: PIL.Image
        The image to rotate
    angle: float
        How much the image needs to be rotated (degrees)
    expand: bool
        If the canvas has to be expanded around the rotated image or (if not) to
        maintain the original format

    Returns
    -------
    : PIL.Image
        The image, rotated
    """

    return img.rotate(angle, expand=expand)


def cropImage(img, vertices):
    """
    cropImage crops an image using PIL
    Parameters
    ----------
    img: PIL.Image
        The image to be cropped
    vertices: tuple(Int)
        A tuple of integers identifying coordinate of the crop box in the order Left,
        Up, Bottom, Right

    Returns
    -------
    : PIL.Image
        The cropped image
    """

    return img.crop(vertices)


def imagePreProcess(path):
    """
    Rotates and crop the images contained in the `path` folder, which is expected to be
    divided into `matches` and `unmatched` sub-folders

    Parameters
    ----------
    path: String
        The path of the folder containing png files organized into `matches` and
        `unmatched` sub-folders

    Returns
    -------
    Nothing

    """

    matches_files = os.scandir(f'{path}/matches')
    unmatched_files = os.scandir(f'{path}/unmatched')

    print('-' * _string_mult)

    # Left/Upper Right/Lower coordinates of the cropping box
    crop_vertices = (481, 2545, 9583, 6985)

    for file in tqdm(matches_files, desc='Pre-processing matched images'):
        if file.is_file():
            with Image.open(file.path) as img:
                cropImage(rotateImage(img, 270, expand=True), crop_vertices).save(
                    file.path
                    )
    for file in tqdm(unmatched_files, desc='Pre-processing unmatched images'):
        if file.is_file():
            with Image.open(file.path) as img:
                cropImage(rotateImage(img, 270, expand=True), crop_vertices).save(
                    file.path
                    )

    print('Completed.')
    print('-' * _string_mult)


if __name__ == '__main__':
    _, data_path = argv

    # Copy data into local content folder
    os.system(f'cp -r {data_path} /content/')

    # Rename XML files
    renameXMLFiles('/content/data/xml')

    # Rename PDF files
    renamePDFFiles('/content/data/pdf')

    # Convert PDF files to PNG
    convertPdfToPng(f'/content/data')

    # Find matches between xml / png and organize files
    matchesFinder('/content/data/png', '/content/data/xml')

    # Rotates PNG and crop to PNG
    imagePreProcess('/content/data/png')
