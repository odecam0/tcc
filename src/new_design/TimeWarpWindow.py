import pandas as pd
import numpy  as np

from tsaug  import Resize
from random import randint

def warp_window(window : pd.DataFrame,
                W : int,
                features = ['x', 'y', 'z']):

    """
    Window deve ser um pandas.DataFrame representando uma janela de dados
    W é o parâmetro do algoritmo de time warping
      define uma margem para se pôr o novo centro.
    features representa quais colunas serão modificadas
    """

    # Se os indices não forem resetados, ocorrem erros
    # ao atribuir os tempos.
    window = window.reset_index(drop=True)

    # W não pode ser maior que metade do tamanho da janela (N)
    N = window.shape[0]
    if W >= N/2:
        raise Exception('Parâmetro W não pode ser maior que N/2')

    anchor_point = randint( W, N - W )

    # Se d == 0, então não haverá modificação nos dados
    while True :
        d = randint( -W + 1, W - 1 )
        if d != 0:
            break

    augmented_window = pd.DataFrame()

    for f in features:
        # TODO : Verificar indices
        left  = window[f].iloc[              : anchor_point ]
        right = window[f].iloc[ anchor_point :             ]

        left  = np.array( left  )
        right = np.array( right )

        left_size  =  left.shape[0]
        right_size = right.shape[0]

        # TODO : Verificar distorção
        # Se d < 0 : comprime left e estica   right
        # Se d > 0 : estica   left e comprime right
        left  = Resize( left_size  + d ).augment(left)
        right = Resize( right_size - d ).augment(right)

        augmented_window[f] = pd.Series( np.append(left, right) )

    # Adcionar colunas não modificadas na janela aumentada
    other_columns = [ i for i in window.columns if i not in features ]

    for o in other_columns:
        augmented_window[o] = window[o]
    
    return augmented_window, ( anchor_point, d )
