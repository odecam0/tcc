import pandas as pd
import numpy as np

from pdb import set_trace

def calc_feature(df, func, featured_columns=['x', 'y', 'z']):
    """
    aplica FUNC nas colunas especificadas por FEATURED_COLUMNS
    recebe um DF com N linhas e retorna um DF com 1 linha
    FUNC não pode ser uma função lambda, pois seu nome é utilizado
    para criar novas colunas no novo DataFrame
    """

    df_size = df.shape[0]
    new_dic = {}

    for c in featured_columns:
        s = df.loc[:,c] # Column series
        new_dic[func.__name__ + '_' + c] = func(s)

    not_featured_columns = [c for c in df.columns if c not in featured_columns]

    for c in not_featured_columns:
        new_dic[c] = df.loc[:,c].iloc[df_size//2]

    return_df = pd.DataFrame(new_dic, [0])

    return return_df.copy()
    
def merge_features(dfs):
    """
    Recebe vários dataframes gerados por calc_features (ou não, né) e os transforma
    em 1 dataframe só, sem duplicar as colunas de metadados
    """

    aux = pd.concat(dfs, axis=1)
    aux = aux.loc[:, ~aux.columns.duplicated()]

    return aux.copy()

