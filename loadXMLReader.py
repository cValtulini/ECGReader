import os
os.system('apt-get install libxml2')
os.system('apt-get install libxml2-dev')

os.system('mv /content/ECGREader/spxml /content/')

os.chdir('spxml')
os.system('python3  setup.py install --user')
os.chdir('..')

import sys
sys.path.append(r'/root/.local/lib/python3.7/site-packages')
