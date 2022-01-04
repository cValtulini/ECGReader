"""
This module is useful to handle our files to make sure that pdf and xml name corresponds
"""
from sys import argv
import os
import random
import pdfplumber
from PyPDF2 import PdfFileWriter, PdfFileReader

def renameXMLFiles(path):
    """
    Renames XML files into the new name format
    initials_patientID_ddmmaa.xml
    """

    #assert isinstance(path, str), 'Warning: path is not a string'
    
    # Creates iterator over directory content
    directory_list = os.scandir(path)
    
    print('-' * 32)
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
    print('-' * 32)


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

def MultipagesPdfPatientCodeNameExtractor(path,code_bottom=793.0860326260371,break_bottom=366.121624373963):
    """
    MultipagesPdfPatientCodeNameExtractor extracts patient codes from '.pdf' files of ECGs composed by many pages.

    :param path: should be a full file path comprehensive of '.pdf'
    :param code_bottom: is the pdf coordinate at which the function finds the patient code
    :param break_bottom: is the pdf coordinate at which the function stops to read the patient name
    :return: the patient codes and names as two different lists
    """
    patient_codes=[]
    names=[]
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            list_pdf=page.extract_words(vertical_ttb=False)
            code=False
            name=str()
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
    return patient_codes,names

def renamePDFFiles(path):
    """
    Renames PDF files into the new name format
    initials_patientID_ddmmaa.pdf
    """
    #assert isinstance(path, str), 'Warning: path is not a string'
    
    # Creates iterator over directory content
    
    directory_list = os.scandir(path)

    print('-' * 32)
    print(f'Iterating over {path} to rename PDF files...')
    for sub_directory in directory_list:

        # We know that files are organized into subfolder identified by dates
        if sub_directory.is_dir():
            if sub_directory.name == 'ECG 301221':
                sub_dir_list = os.scandir(sub_directory.path)
                for file in sub_dir_list:
                    patient_ids, names= MultipagesPdfPatientCodeNameExtractor(file.path)
                    
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


if __name__ == '__main__':
    _, data_path = argv

    # Copy data into local content folder
    os.system(f'cp -r {data_path} /content/')

    # Rename XML files
    renameXMLFiles('/content/data/xml')

    # Rename PDF files
    renamePDFFiles('/content/data/pdf')
    
    # Find association xml + pdf
