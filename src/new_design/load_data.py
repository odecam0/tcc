from os import walk             
import os
import re

import pandas as pd

# https://docs.python.org/3/library/typing.html

def process_name_scheme(name_scheme:str):
    """
    Extraindo nomes dos campos de metadados (names) e regexps (rgs) para
    encontrar seus valores nos nomes dos arquivos. E uma regexp para
    encontrar todos os nomes de arquivos na pasta especificada (name_scheme)
    """
    names, rgs = [], []
    while True:
        pbegin, pend = name_scheme.find('<'), name_scheme.find('>')
        if pbegin == -1: break # Caso não ache a próxima palavra sai do loop
        sub_text    = name_scheme[pbegin + 1: pend]
        sep_pos     = sub_text.find(':')
        names      += [sub_text[           : sep_pos]]
        rgs        += [sub_text[sep_pos + 1:        ]]
        name_scheme = name_scheme[:pbegin] + rgs[-1] + name_scheme[pend + 1:]

    return name_scheme, names, rgs

def list_files(root_dir:str, files_regexp:str):
    """
    Retorna nome dos arquivos que estão em 'root_dir' e seus subdiretórios
    e são match da regexp 'files_regexp'
    """
    resulting_files = []
    for root, dirs, files in walk(root_dir):
        resulting_files += [os.path.join(root, f) for f in files
                            if re.search(files_regexp, str(os.path.join(root, f)))]

    return resulting_files

def load_data(root_dir:str, name_scheme:str):

    rg_scheme, names, rgs = process_name_scheme(name_scheme)
    # rg_scheme possui regexp dos nomes dos arquivos

    # Pegando lista de arquivos para carregar na memoria
    files_to_load = list_files(root_dir, rg_scheme)

    # ! ESSA PARTE ESTA MAL ESCRITA !
    # Utiliza os nomes dos campos, as regexps e a lista de arquivos
    # para carregar os dados em um pandas.DataFrame e atribuir os metadados
    # às colunas
    df = pd.DataFrame()
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

            df_aux = pd.read_csv(f_orig, delim_whitespace=True,
                                 names=['x', 'y', 'z', 'tempo', 'sensor'])\
                       .assign(**metad)

        df = pd.concat([df, df_aux])

    return df.reset_index(drop=True)

## Usage example
# root_dir = "./processed_data/"
# scheme   = "Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.csv"
# df       = load_data(root_dir, scheme)
