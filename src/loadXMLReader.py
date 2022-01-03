"""
Install the XMLReader in the `spxml` folder
Adds the installation directory to the system path
"""
import os

print("Looking for dependencies:")
os.system('apt-get install libxml2')
os.system('apt-get install libxml2-dev')

os.system('mv ./ECGReader/spxml .')

print("Installing spxml: ")
os.chdir('spxml')
os.system('python3  setup.py install --user')
os.chdir('..')

print("Adding installation folder to path: ")
import sys
sys.path.append(r'/root/.local/lib/python3.7/site-packages')