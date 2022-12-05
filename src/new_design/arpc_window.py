import pandas as pd
from arpc_utils import aip_gen

from pdb import set_trace

def get_rolling_windows(df, size=10, params={}):
    """
    Recebe um dataframe,
    retorna 1 <alguma_coisa> para cada janela criada com df.rolling(..)
    """

    # O problema disso aqui é o seguinte:
    #  Do jeito que tá implementado no pandas as janelas são
    #  obtidas de forma 'lazy'
    # Penso que talvez deve-se tirar proveito disso..
    # Mas já decidi que não

    windows = {}

    for d in aip_gen(df):
        # Assuming all metadata is string
        metadata_columns = ['participante', 'atividade', 'intensidade'] # modify this to get generic
        key = ''
        for c in metadata_columns:
            try:
                key += d[c].iloc[0]
            except:
                set_trace()

        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
        #                                                       dont get windows with wrong size
        windows[key] = [i for i in d.rolling(size, **params) if i.shape[0] == size]

    return windows
