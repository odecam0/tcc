from os import walk             
import re

import Pandas as pd

root_dir = "./processed_data/"
scheme   = "Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.csv"

def load_data(self, root_dir:str, name_scheme:str):

    # Extraindo nomes dos campos de metadados (names) e regexps (rgs) para
    # encontrar seus valores nos nomes dos arquivos. E uma regexp para
    # encontrar todos os nomes de arquivos na pasta especificada (text/rg_scheme)
    text = name_scheme
    names, rgs = [], []
    while True:
	pbegin, pend = text.find('<'), text.find('>')
	if pbegin == -1: break
	sub_text     = text[pbegin+1:pend]
	st_sep       = sub_text.find(':')
	names       += [sub_text[          :st_sep]]
	rgs         += [sub_text[st_sep + 1:      ]]
	text         = text[:pbegin] + rgs[-1] + text[pend + 1:]

    # Contém regexp utilizada para encontrar arquivos que serão
    # carregados na memória
    rg_scheme = text

    # Pegando lista de arquivos para carregar na memoria
    files_to_load = []
    for root, dirs, files in walk(root_dir):
        for f in files:
            if re.search(rg_scheme, str(os.path.join(root, f))):
                files_to_load.append(os.path.join(root, f))

    for f in files_to_load:
        f_orig = f
        
	# Cria dicionário com campo de metadados e seu valor
        metad = {}
	for i in range(len(names)):
	    f = re.sub(root_dir, "", f)
	    m = re.search(rgs[i], f)
	    try:
                # Corta string pra remover oque já foi 'matched'
		f = f[m.span()[1]:]
		metad[names[i]] = m[0]
	    except AttributeError as e:
		print(e,'\nO regexp não deve ter dado match')

        df = pd.DataFrame()
	df.read_csv(f_orig, delim_whitespace=True,
		    names=['x', 'y', 'z', 'tempo', 'sensor'])\
          .assign(metad)
