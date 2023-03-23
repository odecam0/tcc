import numpy as np
import pandas as pd

def get_window_freqs(df:pd.DataFrame, columns=['x', 'y', 'z']):

    # Aqui estarão a distribuição de frequências para cada coluna
    freq_columns = [np.abs(np.fft.fft(df[c])) for c in columns]

    freqs_df = pd.DataFrame(freq_columns, columns=columns)

    return freqs_df
