"""
This module is useful to handle our files to make sure that pdf and xml name corresponds
"""
from sys import argv
import os
from tqdm import *

"""
Renames XML files into the new name format
FOR NOW: check comments inside the function to see the name format
"""
def renameXMLFiles(path):
    #assert isinstance(path, str), 'Warning: path is not a string'
    
    # Creates iterator over directory content
    directory_list = os.scandir(path)
    
    print('-' * 32)
    print(f'Iterating over {path} to rename XML files...')
    for file in tqdm(directory_list):
        if file.is_file:
            file_path = file.path

            # Produces a list containing
            # ['patient ID', 'surname', 'name', 'aaaa-mm-dd_hh-mm-ss']
            # surname and name may have the structure 'sur1_sur2' 'name1_name2'
            parse_filename = file.name.split(',')

            # Creates a list containing the elements for the new filename:
            # initials_patientID_ddmmaa.xml
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

            os.rename(file_path, path+'/'+'_'.join(new_filename)+'.xml')  

        elif file.is_dir:
            print(f'Found subdirectory: {file.path}')
    
    print('All XML files renamed')
    print('-' * 32)


# Main Code Execution
_, data_path = argv

# Copy data into local content folder
os.system(f'cp -r {data_path} /content/')

# Rename XML files
renameXMLFiles('/content/data/xml')


# Get PDF files
# Rename PDF files

# Find association name + data
