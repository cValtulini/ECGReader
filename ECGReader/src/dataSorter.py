"""
This module is useful to handle our files to make sure that pdf and xml name corresponds
"""
from sys import argv
import os
import random
import pdfplumber
from PyPDF2 import PdfFileWriter, PdfFileReader
from PIL import Image
import PIL


# Multiplies the '-' character that separates sections of outputs
# for some functions
_string_mult = 40


def renameXMLFiles(path):
    """
    Renames XML files into the new name format
    initials_patientID_ddmmaa.xml
    """

    #assert isinstance(path, str), 'Warning: path is not a string'
    
    # Creates iterator over directory content
    directory_list = os.scandir(path)
    
    print('-' * _string_mult)
    print(f'Iterating over {path} to rename XML files...')
    for file in directory_list:
        if file.is_file():
            # Produces a list containing
            # ['patient ID', 'surname', 'name', 'aaaa-mm-dd_hh-mm-ss']
            # surname and name may have the structure 'sur1_sur2' 'name1_name2'
            parse_filename = file.name.split(',')

            # Creates a list containing the elements for the new filename
            new_filename = []

            name_initials = []
            for substring in parse_filename[1].split('_'):
                if len(substring): name_initials.append(substring[0].upper())
            for substring in parse_filename[2].split('_'):
                if len(substring): name_initials.append(substring[0].upper())

            new_filename.append(''.join(name_initials))
            new_filename.append(parse_filename[0].upper())
            
            date = []
            for substring in parse_filename[3].split('_')[0].split('-'):
                date.insert(0, substring)
            date[2] = date[2][2]+date[2][3]
            new_filename.append(''.join(date))

            os.rename(file.path, path+'/'+'_'.join(new_filename)+'.xml')  

        elif file.is_dir():
            print(f'Found subdirectory: {file.path}')
            print(f'Ignored')
    
    print('All XML files renamed.')
    print('-' * _string_mult)


def pdfPatientIDExtractor(path):
    """
    pdfPatientIDExtractor extracts patient codes from '.pdf' files of ECGs.

    :param path: should be a full file path comprehensive of '.pdf'
    :return: the patient code of the file passed as parameter
    """ 

    with pdfplumber.open(path) as pdf:
        first_page = pdf.pages[0]
        # Returns a list of dictionaries with various parameters, including the
        # extracted text
        list_pdf = first_page.extract_words(vertical_ttb=False)
         
        for dictionary in list_pdf:
            # the patient code is found in the dictionary with a specific
            # value of the 'bottom' key
            if dictionary['bottom'] == 793.0860326260371:
                return dictionary['text']

        print(f'pID not found for: {path}')
        return 'NotFound'+str(random.randint(0, 100))


def multipagesPdfPatientIDNameExtractor(path,
                                        code_bottom=793.0860326260371,
                                        break_bottom=366.121624373963):
    """
    multipagesPdfPatientIDNameExtractor extracts patient codes from '.pdf' files of ECGs composed by many pages.

    :param path: should be a full file path comprehensive of '.pdf'
    :param code_bottom: is the pdf coordinate at which the function finds the patient code
    :param break_bottom: is the pdf coordinate at which the function stops to read the patient name
    :return: the patient codes and names as two different lists
    """
    patient_codes = []
    names = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            list_pdf = page.extract_words(vertical_ttb=False)
            code = False
            name = str()
            for dictionary in list_pdf:
                if dictionary['bottom'] == code_bottom:
                    patient_codes.append(dictionary['text'])
                    code=True
                if dictionary['bottom'] == break_bottom:
                    break
                if code and dictionary['bottom'] != code_bottom:
                    name=name+dictionary['text']
            if name != str():
                names.append(name)
    return patient_codes, names


def renamePDFFiles(path):
    """
    Renames PDF files into the new name format
    initials_patientID_ddmmaa.pdf
    """
    #assert isinstance(path, str), 'Warning: path is not a string'
    
    # Creates iterator over directory content
    
    directory_list = os.scandir(path)

    print('-' * _string_mult)
    print(f'Iterating over {path} to rename PDF files...')
    for sub_directory in directory_list:

        # We know that files are organized into subfolder identified by dates
        if sub_directory.is_dir():
            if sub_directory.name == 'ECG 301221':
                sub_dir_list = os.scandir(sub_directory.path)
                for file in sub_dir_list:
                    patient_ids, names= multipagesPdfPatientIDNameExtractor(file.path)
                    
                    inputpdf = PdfFileReader(open(file.path, "rb"))
                    for i in range(inputpdf.numPages):
                        output = PdfFileWriter()
                        output.addPage(inputpdf.getPage(i))
                        with open("document-page%s.pdf" % i, "wb") as outputStream:
                            output.write(outputStream)

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
                        # subdirectory name is 'ECG DATE' we only keep 'DATE'
                        new_filename.append(sub_directory.name.split(' ')[1])

                        os.rename(file.path, path+'/'+'_'.join(new_filename)+'.pdf')

                    elif file.is_dir():  
                        print(f'Found subdirectory: {file.path}')
                        print(f'Ignored')
            
            os.rmdir(sub_directory.path)

        # We ignore everything that is not a subfolder in the main folder
        elif sub_directory.is_file():
            print(f'File found: {sub_directory.path}')
            print('Ignored')
        
    print('-' * _string_mult)


def matchesFinder(path_to_png, path_to_xml):
    """
    Find matches between files in the two folder, excluding file extensions.
    Reorganize files into `matches` and `unmatched` folders.
    """
    # List files in the two directories keeping only the filename without
    # extension
    png_list = [file.path.split('.')[0].split('/')[-1]
                for file in os.scandir(path_to_png)]
    xml_list = [file.path.split('.')[0].split('/')[-1]
                for file in os.scandir(path_to_xml)]

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

    # Moves matches into the proper folder
    for filename in matches:
        png_src = f'{path_to_png}/{filename}.png'
        png_dst = f'{path_to_png}/matches/{filename}.png'
        os.rename(png_src, png_dst)

        xml_src = f'{path_to_xml}/{filename}.xml'
        xml_dst = f'{path_to_xml}/matches/{filename}.xml'
        os.rename(xml_src, xml_dst)

    # Moves unmatched files, ignores subdirectories
    for file in os.scandir(path_to_png):
        if file.is_file():
            dst = f'{path_to_png}/unmatched/{file.name}'
            os.rename(file.path, dst)
    for file in os.scandir(path_to_xml):
        if file.is_file():
            dst = f'{path_to_xml}/unmatched/{file.name}'
            os.rename(file.path, dst)

    print('Matches found and files moved')
    print(f'{len([_ for _ in os.scandir(path_to_png) if _.is_file()])} png files remaining')
    print(f'{len([_ for _ in os.scandir(path_to_xml) if _.is_file()])} xml files remaining')
    print('-' * _string_mult)


def convertPdfToPng(path_to_data, remove_pdf_folder=False):
    """
    Expects to find a pdf folder, divided into `/matches` and `/unmatched`
    converts pdf to png and puts into `/png`
    """
    out_path = f'{path_to_data}/png'

    # Gets folders file list
    matches_pdf = os.scandir(f'{path_to_data}/pdf/matches')
    unmatched_pdf = os.scandir(f'{path_to_data}/pdf/unmatched')

    # Creates folder with new format at the same level of the pdf folder
    if not  os.path.isdir(out_path):
        os.makedirs(f'{out_path}/matches')
        os.makedirs(f'{out_path}/unmatched')

    print('-' * _string_mult)
    print('Converting from PDF to PNG...')
    # Saves PDF as PNG images
    for file in matches_pdf:
        with pdfplumber.open(file.path) as pdf:
            filename = file.name.split('.')[0]
            pdf.pages[0].to_image().save(f'{out_path}/matches/{filename}.png',
                                        format='PNG')
    for file in unmatched_pdf:
        with pdfplumber.open(file.path) as pdf:
            filename = file.name.split('.')[0]
            pdf.pages[0].to_image().save(f'{out_path}/unmatched/{filename}.png',
                                        format='PNG')

    print('Conversion completed.')
    print('-' * _string_mult)

    # Removes the pdf folder
    if remove_pdf_folder:
        os.removedirs(f'{path_to_data}/pdf')


def rotateImage(img, angle, expand=True):
    """
    Simple function to rotate images using PIL
    """
    return img.rotate(angle, expand=expand)


def cropImage(img, vertices):
    """
    Simple function to crop images using PIL
    """
    return img.crop(vertices)


def imagePreProcess(path):
    """
    Rotates images, crop them to ECG removing additional elements.
    Expect `path` folder to be divided into `matches` and `unmatched`
    """
    matches_files = os.scandir(f'{path}/matches')
    unmatched_files = os.scandir(f'{path}/unmatched')

    print('-' * _string_mult)
    print('Pre-processing PNG images...')

    # Left/Upper Right/Lower coordinates of the cropping box
    crop_vertices = (38, 203, 767, 560)

    for file in matches_files:
        with Image.open(file.path) as img:
            cropImage(
                rotateImage(img, 270, expand=True),
                crop_vertices
                ).save(file.path)
    for file in unmatched_files:
        with Image.open(file.path) as img:
            cropImage(
                rotateImage(img, 270, expand=True),
                crop_vertices
                ).save(file.path)

    print('Completed.')
    print('-' * _string_mult)



if __name__ == '__main__':
    _, data_path = argv

    # Copy data into local content folder
    os.system(f'cp -r {data_path} /content/')

    # Rename XML files
    renameXMLFiles('/content/data/xml')

    # Rename PDF files
    renamePDFFiles('/content/data/pdf')
    
    # Convert PDF files to PNG
    # convertPdfToPng(f'/content/data/pdf', remove_pdf_folder=True)

    # Rotates PNG and crop to PNG
    # imagePreProcess('/content/data/png')

    # Find matches between xml / png and organize files
    # matchesFinder('content/data/png', 'content/data/xml')