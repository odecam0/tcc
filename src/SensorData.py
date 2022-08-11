import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

from pdb import set_trace

# for loading data
atividades   = ['Andando', 'Sentado', 'Deitado']
intensidades = ['Leve', 'Moderado', 'Vigoroso']

dataset_dir  = '~/ic/dataset/data/'
p_dir        = ['Aluno'+str(i+1) for i in range(11)]

full_paths = {}
for p in p_dir:
    full_paths[p] = {}
    for a in atividades:
        full_paths[p][a] = {}
        for i in intensidades:
            full_paths[p][a][i] = dataset_dir + p + '/' + a + i + '.txt'

# for iterating over data
def aip_gen(df, select=[]):
    """Generator for activity, intensity and participant pandas DataFrame
       Specific classes can be selected passing array of tuples with Participant,
       Activity and Intensity as strings in this order
       """
    try:
        atividades = df['Atividade'].unique()
        intensidades = df['Intensidade'].unique()
        participantes = df['Participante'].unique()
    except KeyError as err:
        print('Uma das colunas do dataframe está com nome errado?')
        print('Na função aip_gen')
        print('Coluna que deu errado: ', err.args[0])

    if len(select) > 0:
        for s in select:
            yield df.loc[(df['Atividade'] == s[1]) &
                         (df['Intensidade'] == s[2]) &
                         (df['Participante'] == s[0])]
        return

    for a in atividades:
        for i in intensidades:
            for p in participantes:
                yield df.loc[(df['Atividade'] == a) &
                             (df['Intensidade'] == i) &
                             (df['Participante'] == p)]

def rem_outliers(df):
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

from pprint import pprint
from inspect import getmembers

class SensorData:
    def __init__(self):
        df = pd.DataFrame(columns=['x', 'y', 'z', 'tempo', 'sensor'])

        participantes = list(range(len(p_dir)))

        # Loading data
        for p, pn in zip(p_dir, participantes):
            for a in atividades:
                for i in intensidades:
                    df_r = pd.read_csv(full_paths[p][a][i], delim_whitespace=True,
                                    names=['x', 'y', 'z', 'tempo', 'sensor'])\
                            .assign(Atividade = a,
                                    Intensidade = i,
                                    Participante = pn)

                    df_r = df_r.loc[df_r['sensor'] == 'a']

                    df = pd.concat([df, df_r], ignore_index=True)

        self.data = df
        self.participantes = participantes

    def fix_dup(self, remFirst=False):
        """Função para resolver o caso onde em uma mesma classe se encontram
           duas séries temporais. Ou seja, em algum momento o 'timestamp' de
           uma amostra é inferior ao da amostra anterior."""
        df = self.data

        df_ret = pd.DataFrame()

        for df_it in aip_gen(df):
            for id in df_it.index[1:]:
                # Encontra id onde tempo da série volta para início
                if df_it.loc[id, 'tempo'] < df_it.loc[id-1, 'tempo']:
                    tempo_base = df_it.loc[id-1, 'tempo']
                    id_base    = id
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
            # Cria um novo pandas.DataFrame com o novo pandas.Series na
            # coluna 'tempo'
            df_aux = df_it.drop(columns=['tempo']).loc[:id_base-1]
            df_aux['tempo'] = tempo.values

            if df_aux['tempo'].isnull().values.any():
                set_trace()

            df_ret = pd.concat([df_ret, df_aux], ignore_index=True)

        self.data = df_ret

    def rotate_class(self, classes, vec):
        result = []

        d = list(aip_gen(self.data, select=classes))
        for i, n in zip(d, range(len(list(d)))):
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

        for d in  aip_gen(self.data):
            a, i, p = d.iloc[0][['Atividade', 'Intensidade', 'Participante']]
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

        self.data = result_df
            
                    
            
    def remove_outliers(self):
        df = self.data

        df_noout = pd.DataFrame()

        for df_remout in aip_gen(df):
            df_remout = rem_outliers(df_remout)
            df_noout  = pd.concat([df_noout, df_remout])

        self.data = df_noout

    def remove_beginning(self):
        # Removing first 10 seconds
        df = self.data
        df = df.loc[df['tempo'] >= 10000]
        df = df.reset_index(drop=True)
        self.data = df

    def segment_extract_features(self, window_size=10):
        df = self.data
        df1_out = pd.DataFrame()

        for df_local in aip_gen(df):
            try:
                df_local  = df_local.astype({'tempo': int})
            except:
                set_trace()
            rolling_w = df_local[['x', 'y', 'z', 'tempo']].rolling(window=window_size, center=True,  on='tempo')

            df_aux = pd.DataFrame()

            df_aux['mean_x'] = rolling_w.mean().dropna()['x']
            df_aux['mean_y'] = rolling_w.mean().dropna()['y']
            df_aux['mean_z'] = rolling_w.mean().dropna()['z']

            df_aux['median_x'] = rolling_w.median().dropna()['x']
            df_aux['median_y'] = rolling_w.median().dropna()['y']
            df_aux['median_z'] = rolling_w.median().dropna()['z']

            df_aux['var_x'] = rolling_w.var().dropna()['x']
            df_aux['var_y'] = rolling_w.var().dropna()['y']
            df_aux['var_z'] = rolling_w.var().dropna()['z']

            df_aux['std_x'] = rolling_w.std().dropna()['x']
            df_aux['std_y'] = rolling_w.std().dropna()['y']
            df_aux['std_z'] = rolling_w.std().dropna()['z']

            df_aux['tempo'] = rolling_w.mean().dropna()['tempo']

            df_aux['Atividade']    = df_local['Atividade'].iloc[0]
            df_aux['Intensidade']  = df_local['Intensidade'].iloc[0]
            df_aux['Participante'] = df_local['Participante'].iloc[0]

            df1_out = pd.concat([df1_out, df_aux])

        self.data = df1_out

    def kfold_crossval(self, k):
        df = self.data

        # Building and testing classifier
        X = df.drop(columns=['tempo', 'Atividade', 'Intensidade', 'Participante'])
        y = df['Atividade'] + df['Intensidade']

        rf  = RandomForestClassifier(n_estimators=100, random_state=42)
        cvs = cross_val_score(rf, X, y, cv=k)

        return cvs

    def confusion_matrix(self, save=False):
        df = self.data
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Primeira classificação: Utilizando os próprios dados do participante para treinar o modelo
        X = df.drop(columns=['Atividade', 'Intensidade', 'Participante', 'tempo'])
        y = df['Atividade'] + df['Intensidade']

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        rf.fit(X_train, y_train)

        prediction = rf.predict(X_test)

        cmat = metrics.confusion_matrix(y_test, prediction, labels=rf.classes_)
        ativ_intens = [a+i for a in atividades for i in intensidades]

        cmat_disp = sklearn.metrics.ConfusionMatrixDisplay(cmat, display_labels=ativ_intens).plot(cmap="Blues")
        cmat_disp.ax_.set(ylabel="Rótulo verdadeiro", xlabel="Rótulo previsto")
        plt.xticks(rotation=45, ha='right')
        plt.title("Matriz de confusão")

        if save:
            plt.savefig('mat_mesmo_conjunto.png', bbox_inches='tight')
        else:
            plt.show()

    def plot_all(self, save=False, participantes=[]):
        # atividades e intensidades foram declarados no início do arquivo
        df = self.data

        if len(participantes) == 0:
            participantes = self.participantes

        for participante in participantes:
            fig, axs = plt.subplots(len(atividades), len(intensidades), sharey=True)
            fig.suptitle('Aluno'+str(participante))

            for a, n1 in zip(atividades, range(len(atividades))):
                for i, n2 in zip(intensidades, range(len(intensidades))):

                    if n1 != len(atividades) - 1:
                        axs[n1][n2].set_xticks([])

                    sub_d_raw = df.loc[(df['Atividade'] == a) &
                                       (df['Intensidade'] == i) &
                                       (df['Participante'] == participante)]

                    axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['x'], linewidth=0.5, color='blue')
                    axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['y'], linewidth=0.5, color='red')
                    axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['z'], linewidth=0.5, color='green')

            for a, n1 in zip(atividades, range(len(atividades))):
                axs[n1][0].set_ylabel(a)

            for i, n2 in zip(intensidades, range(len(intensidades))):
                axs[len(atividades)-1][n2].set_xlabel(i)

            if save:
                plt.savefig('Aluno' + str(participante) +'.png')
            else:
                plt.show()
