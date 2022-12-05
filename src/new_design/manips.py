import pandas as pd
import numpy  as np

from scipy.spatial.transform import Rotation as R

from arpc_utils import aip_gen

from sklearn.preprocessing import StandardScaler

def fix_dup(df, remFirst=False):
    """
    Função para resolver o caso onde em uma mesma classe se encontram
    duas séries temporais. Ou seja, em algum momento o 'timestamp' de
    uma amostra é inferior ao da amostra anterior.
    """
    df_ret = pd.DataFrame()

    for df_it in aip_gen(df):
        for id in df_it.index[1:]:
            # Encontra id onde tempo da série volta para início
            if df_it.loc[id, 'tempo'] < df_it.loc[id-1, 'tempo']:
                tempo_base = df_it.loc[id-1, 'tempo']
                id_base    = id - 1
                break
        else:
            # Caso não haja duas séries na mesma classe
            df_ret = pd.concat([df_ret, df_it])
            continue

        if remFirst:
            tempo = df_it.loc[id_base:, 'tempo']
        else:
            # Caso tenha encontrado o início da segunda série temporal
            # Cria um pandas.Series com os timestamps da segunda série
            # temporal somados ao último timestamp da pimeira série
            tempo  = pd.concat([df_it.loc[:id_base-1, 'tempo'],
                                df_it.loc[id_base:, 'tempo'] + tempo_base])

        if remFirst:
            df_aux = df_it.loc[:id_base]
        else:
            # Cria um novo pandas.DataFrame com o novo pandas.Series na
            # coluna 'tempo'
            df_aux = df_it.drop(columns=['tempo'])
            df_aux['tempo'] = tempo.values
            
        df_ret = pd.concat([df_ret, df_aux], ignore_index=True)

    return df_ret

def rotate_class(df, classes, vec):
    result = []

    d = list(aip_gen(df, select=classes))
    for i, n in zip(d, range(len(d))):
        a = [i.loc[:,'x'].mean(),
             i.loc[:,'y'].mean(),
             i.loc[:,'z'].mean()]
        b = vec

        perp_vec = np.cross(a, b)
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
        angle    = np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

        rotation = R.from_rotvec(angle * perp_vec)

        arr = np.array((i.loc[:,'x'], i.loc[:,'y'], i.loc[:,'z']))
        arr = arr.transpose()
        arr = rotation.apply(arr)
        # Aqui os dados já devem estar rotacionados

        result += [(*classes[n], arr)]

    result_df = pd.DataFrame()

    for d in  aip_gen(df):
        a, i, p = d.iloc[0][['atividade', 'intensidade', 'participante']]
        for r in result:
            if p==r[0] and a==r[1] and i==r[2]:
                arr = r[3]
                df_aux = d.drop(columns=['x', 'y', 'z'])
                df_aux['x'] = arr[:,0]
                df_aux['y'] = arr[:,1]
                df_aux['z'] = arr[:,2]

                result_df = pd.concat([result_df, df_aux])
                break
        else:
            result_df = pd.concat([result_df, d])

    return result_df
            
def _rem_outliers(df):
    df_new = df.copy()

    eixos = ['x', 'y', 'z']

    for e in eixos:

        Q1 = df_new[e].quantile(0.25)
        Q3 = df_new[e].quantile(0.75)
        iQ = Q3 - Q1

        inf = Q1 - 1.5 * iQ
        sup = Q3 + 1.5 * iQ

        df_new[e] = np.where(
            df_new[e] > sup,
            sup,
            np.where(
                df_new[e] < inf,
                inf,
                df_new[e]
            )
        )

    return df_new

def remove_outliers(df):
    df_noout = pd.DataFrame()

    for df_remout in aip_gen(df):
        df_remout = _rem_outliers(df_remout)
        df_noout  = pd.concat([df_noout, df_remout])

    return df_noout

def remove_beginning(df, s=10):
    # Removing first s seconds
    return df.loc[df['tempo'] >= (s * 1000)]\
             .reset_index(drop=True)

def scale_data(df):
    # Devo fazer para cada atividade-intensidade individualmente?
    # ou devo fazer para todos os dados em um mesmo conjunto?
    #        Professor falou para fazer com todos os dados

    non_data_columns = ['tempo', 'sensor', 'atividade',
                        'intensidade', 'participante']
    data = df.drop(columns=non_data_columns)

    scaler = StandardScaler().fit(data)

    scaled_data = scaler.transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
    # scaled_data = pd.concat([scaled_data, self.data.loc[:, non_data_columns]],
    #                         ignore_index=True,
    #                         axis=1)
    scaled_data[non_data_columns] = df.loc[:, non_data_columns]

    # ! Eis aqui algo que eu também deveria entender melhor matemáticamente.

    return scaled_data
