import os
os.system('apt-get install libxml2')
os.system('apt-get install libxml2-dev')

os.mkdir('/content/spxml/')
os.system('cp -r /content/gdrive/MyDrive/Colab\ Notebooks/IDAProject/ProjectCode/XMLReader/* /content/spxml/')

os.chdir('spxml')
os.system('python3  setup.py install --user')
os.chdir('..')

import sys
sys.path.append(r'/root/.local/lib/python3.7/site-packages')
