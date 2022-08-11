atividades   = ['Andando', 'Sentado', 'Deitado']
intensidades = ['Leve', 'Moderado', 'Vigoroso']
ativ_intens  = [i+j for i in atividades for j in intensidades]

dataset_dir  = '/home/brnm/ic/dataset/data/'
p_dir        = ['Aluno'+str(i+1) for i in range(11)]
files        = [a+".txt" for a in ativ_intens]

files_new = [dataset_dir + a + '/' + f for a in p_dir for f in files if 'Deitado' in f]

import re
files_old = [re.sub('Deitado', 'Deitaodo', s) for s in files_new]

import os
for o,n in zip(files_old, files_new):
    os.rename(o, n)

"""
https://pynative.com/python-rename-file/#h-os-rename

 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("fix-names.py").read(), globals())

from pprint import pprint as pp
import inspect as insp

import os

pp(insp.getmembers(os)
pp(insp.getdoc(os.rename))
pp(insp.getsource(os.renames))
pp(insp.getsourcefile(os.renames))
# (find-fline "/usr/lib/python3.10/os.py")

insp.getsourcefile(os.rename)
pp(insp.getmembers(insp))

os.__builtins__
pp(os.__builtins__)


import re
pp(insp.getmembers(re))
pp(insp.getdoc(re.sub))
pp(insp.getsource(re.sub))
"""
