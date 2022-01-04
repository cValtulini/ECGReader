"""
This module is useful to handle our files to make sure that pdf and xml name corresponds
"""
from sys import argv
import os
import random
import pdfplumber

# Multiplies the '-' character that separates sections of outputs
# for some functions
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


def matchesFinder(path_to_jpeg, path_to_xml):
    """
    Find matches between files in the two folder, excluding file extensions.
    Reorganize files into `matches` and `unmatched` folders.
    """
    # List files in the two directories keeping only the filename without
    # extension
    jpeg_list = [file.path.split('.')[0].split('/')[-1]
                for file in os.scandir(path_to_jpeg)]
    xml_list = [file.path.split('.')[0].split('/')[-1]
                for file in os.scandir(path_to_xml)]

    print('-' * _string_mult)
    print('Finding matches:')
    # Finds the elements in both lists
    matches = set(jpeg_list).intersection(xml_list)
    print(f'There are {len(matches)} matches in data.')
    print(f'There are {len(jpeg_list)} jpeg files.')
    print(f'There are {len(xml_list)} xml files.')

    # Creates folders to put matches and unmatched files into
    os.mkdir(f'{path_to_jpeg}/matches')
    os.mkdir(f'{path_to_jpeg}/unmatched')
    os.mkdir(f'{path_to_xml}/matches')
    os.mkdir(f'{path_to_xml}/unmatched')

    # Moves matches into the proper folder
    for filename in matches:
        jpeg_src = f'{path_to_jpeg}/{filename}.jpeg'
        jpeg_dst = f'{path_to_jpeg}/matches/{filename}.jpeg'
        os.rename(jpeg_src, jpeg_dst)

        xml_src = f'{path_to_xml}/{filename}.xml'
        xml_dst = f'{path_to_xml}/matches/{filename}.xml'
        os.rename(xml_src, xml_dst)

    # Moves unmatched files, ignores subdirectories
    for file in os.scandir(path_to_jpeg):
        if file.is_file():
            dst = f'{path_to_jpeg}/unmatched/{file.name}'
            os.rename(file.path, dst)
    for file in os.scandir(path_to_xml):
        if file.is_file():
            dst = f'{path_to_xml}/unmatched/{file.name}'
            os.rename(file.path, dst)

    print('Matches found and files moved')
    print(f'{len([_ for _ in os.scandir(path_to_jpeg) if _.is_file()])} jpeg files remaining')
    print(f'{len([_ for _ in os.scandir(path_to_xml) if _.is_file()])} xml files remaining')
    print('-' * _string_mult)


if __name__ == '__main__':
    _, data_path = argv

    # Copy data into local content folder
    os.system(f'cp -r {data_path} /content/')

    # Rename XML files
    renameXMLFiles('/content/data/xml')

    # Rename PDF files
    renamePDFFiles('/content/data/pdf')
    
    # Convert PDF files to JPEG

    # Find matches between xml / jpeg and organize files
    # matchesFinder('content/data/jpeg', 'content/data/xml')
