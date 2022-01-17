"""
Install the XMLReader in the `spxml` folder
Adds the installation directory to the system path
"""
import os

#os.system('mv ./ECGReader/spxml .')

print("Installing spxml: ")
os.chdir('spxml')
os.system('python3 setup.py install')
os.chdir('..')