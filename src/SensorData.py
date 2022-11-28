import pandas as pd
import numpy as np

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import sklearn

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import os

from pdb import set_trace

# for loading data
atividades   = ['Andando', 'Sentado', 'Deitado']
intensidades = ['Leve', 'Moderado', 'Vigoroso']

p_dir        = ['Aluno'+str(i+1) for i in range(11)]

# for iterating over data
def aip_gen(df, select=[]):
    """
    Generator for activity, intensity and participant pandas DataFrame
    Specific classes can be selected passing array of tuples with Participant,
    Activity and Intensity as strings in this order
    """
    try:
        atividades    = df['Atividade'   ].unique()
        intensidades  = df['Intensidade' ].unique()
        participantes = df['Participante'].unique()
    except KeyError as err:
        print('Uma das colunas do dataframe está com nome errado?')
        print('Na função aip_gen'                                 )
        print('Coluna que deu errado: ', err.args[0]              )

    if len(select) > 0:
        for s in select:
            yield df.loc[(df['Atividade']    == s[1]) &
                         (df['Intensidade']  == s[2]) &
                         (df['Participante'] == s[0])]
        return

    for a in atividades:
        for i in intensidades:
            for p in participantes:
                yield df.loc[(df['Atividade']    == a) &
                             (df['Intensidade']  == i) &
                             (df['Participante'] == p)]

def p_gen(df, select=[]):
    if len(select)==0:
        try:
            participantes = df['Participante'].unique()
        except KeyError as err:
            print('Uma das colunas do dataframe está com nome errado?')
            print('Na função aip_gen')
            print('Coluna que deu errado: ', err.args[0])

        for p in participantes:
            yield df.loc[df['Participante'] == p]
    else:
        for p in select:
            yield df.loc[df['Participante'] == p]

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

def get_accuracy(cmat):
    result = []

    n_classes = cmat.shape[0]

    for i in range(n_classes):
        n_class       = cmat[i].sum()
        n_right_pred  = cmat.diagonal()[i]
        result       += [n_right_pred / n_class]

    return result

class SensorData:
    def __init__(self, dataset_dir  = '~/ic/dataset/data/', extension='.txt'):
        df = pd.DataFrame(columns=['x', 'y', 'z', 'tempo', 'sensor'])

        full_paths = {}
        for p in p_dir:
            full_paths[p] = {}
            for a in atividades:
                full_paths[p][a] = {}
                for i in intensidades:
                    full_paths[p][a][i] = dataset_dir + p + '/' + a + i + extension

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

    def scale_data(self):
        # Devo fazer para cada atividade-intensidade individualmente?
        # ou devo fazer para todos os dados em um mesmo conjunto?
        #        Professor falou para fazer com todos os dados
        self.data = self.data.reset_index(drop=True)

        non_data_columns = ['tempo', 'sensor', 'Atividade',
                            'Intensidade', 'Participante']
        data = self.data.drop(columns=non_data_columns)

        scaler = StandardScaler().fit(data)

        scaled_data = scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
        # scaled_data = pd.concat([scaled_data, self.data.loc[:, non_data_columns]],
        #                         ignore_index=True,
        #                         axis=1)
        scaled_data[non_data_columns] = self.data.loc[:, non_data_columns]

        self.data = scaled_data


    def segment_extract_features(self, window_size=10):
        df = self.data
        df1_out = pd.DataFrame()

        for df_local in aip_gen(df):
            # Porque transformar a coluna tempo em int?
            df_local  = df_local.astype({'tempo': int})

            rolling_w = df_local[['x', 'y', 'z', 'tempo']].rolling(window=window_size, center=True)

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

        df1_out['sensor'] = 'a'

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

    def train_model_split1(self, select=[0], noIntens=False, cv=10):
        results = []

        for df in p_gen(self.data, select=select):
            rf = RandomForestClassifier(n_estimators=100, random_state=42)

            this_participant = df.loc[:, 'Participante'].unique()

            print(f'Performing train_model_split1 for pivot: {this_participant[0]}')

            X = df.drop(columns=['Atividade', 'Intensidade', 'tempo', 'sensor'])

            if noIntens:
                y = df['Atividade']
            else:
                y = [i + j for i, j in zip(df['Atividade'], df['Intensidade'])]

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
            rf.fit(X_train, y_train)
            prediction = rf.predict(X_test)

            cmat = metrics.confusion_matrix(y_test, prediction, labels=rf.classes_)

            cvs = cross_val_score(rf, X, y, cv=cv)

            results += [(cmat, prediction, rf, cvs)]

        return results

    def train_model_split2(self, select=[0], noIntens=False):
        results = []

        for df in p_gen(self.data, select=select):
            df_rest = pd.DataFrame()

            # Getting DataFrame with all other participants in it
            this_participant = df.loc[:, 'Participante'].unique()

            print(f'Performing train_model_split2 for pivot: {this_participant[0]}')

            all_participants = self.data.loc[:, 'Participante'].unique()
            others = [i for i in all_participants if float(i) not in this_participant]
            for df_aux in p_gen(self.data, select=others):
                df_rest = pd.concat([df_rest, df_aux])

            rf = RandomForestClassifier(n_estimators=100, random_state=42)

            X_test = df.drop(columns=['Atividade', 'Intensidade', 'tempo', 'Participante', 'sensor'])

            if noIntens:
                y_test = df['Atividade']
                y      = df_rest['Atividade']
            else:
                y_test = [i + j for i, j in zip(df['Atividade'], df['Intensidade'])]
                y = [i + j for i, j in zip(df_rest['Atividade'], df_rest['Intensidade'])]

            X = df_rest.drop(columns=['Atividade', 'Intensidade', 'Participante', 'tempo', 'sensor'])

            rf.fit(X, y)

            prediction = rf.predict(X_test)

            cmat = metrics.confusion_matrix(y_test, prediction)

            results += [(cmat, prediction, rf)]

        return results

    def train_model_split3(self, select=[0], noIntens=False, pct=.2):
        results = []

        for df in p_gen(self.data, select=select):
            df_rest = pd.DataFrame()

            # Getting DataFrame with all other participants in it
            this_participant = df.loc[:, 'Participante'].unique()

            print(f'Performing train_model_spli3 for pivot: {this_participant[0]}')

            all_participants = self.data.loc[:, 'Participante'].unique()
            others = [i for i in all_participants if float(i) not in this_participant]
            for df_aux in p_gen(self.data, select=others):
                df_rest = pd.concat([df_rest, df_aux])

            new_df = pd.DataFrame()
            new_df_rest = df_rest.copy()

            # Adding fraction of samples from df into df_rest
            for a in atividades:
                for i in intensidades:
                    df_aux = df.loc[(df['Atividade'] == a) & (df['Intensidade'] == i)]

                    df_removed = df_aux.sample(frac=pct)
                    df_remained = df_aux.drop(df_removed.index)

                    new_df = pd.concat([new_df, df_remained])
                    new_df_rest = pd.concat([new_df_rest, df_removed])

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            X = new_df_rest.drop(columns=['Atividade', 'Intensidade', 'Participante', 'tempo', 'sensor'])
            X_test = new_df.drop(columns=['Atividade', 'Intensidade', 'tempo', 'Participante', 'sensor'])

            if noIntens:
                y      = new_df_rest['Atividade']
                y_test = new_df['Atividade']
            else:
                y      = [i + j for i, j in zip(new_df_rest['Atividade'], new_df_rest['Intensidade'])]
                y_test = [i + j for i, j in zip(new_df['Atividade'], new_df['Intensidade'])]

            rf.fit(X, y)
            prediction = rf.predict(X_test)

            cmat = metrics.confusion_matrix(y_test, prediction)

            results += [(cmat, rf, prediction)]

        return results

    def initial_fix(self):
        self.remove_outliers()
        self.remove_beginning()
        self.fix_dup(remFirst=True)
        classes = [(1., 'Deitado', 'Moderado')]
        classes += [(4., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
        classes += [(7., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
        self.rotate_class(classes, [0,0,1])
        print("Data fixed")

    def get_all_accuracies(self,
                           noIntens=False,
                           save=False,
                           file_name='all_accuracies_data'):
        result = {}

        ativ_intens = [a+i for a in atividades for i in intensidades]

        if noIntens:
            for a in atividades:
                result[a] = [] 
        else:
            for a in ativ_intens:
                result[a] = [] 

        participantes = self.data.loc[:, 'Participante'].unique()
        results_exp1  = self.train_model_split1(select=participantes)
        results_exp2  = self.train_model_split2(select=participantes)
        results_exp3  = self.train_model_split3(select=participantes)
        
        cmats1 = [i for i,_,_,_ in results_exp1]
        cmats2 = [i for i,_,_ in results_exp2]
        cmats3 = [i for i,_,_ in results_exp3]

        for c1, c2, c3 in zip(cmats1, cmats2, cmats3):
            acc1 = get_accuracy(c1)
            acc2 = get_accuracy(c2)
            acc3 = get_accuracy(c3)

            if noIntens:
                for a, i in zip(atividades, range(len(atividades))):
                    result[a] += [(acc1[i], acc2[i], acc3[i])]
            else:
                for a, i in zip(ativ_intens, range(len(ativ_intens))):
                    result[a] += [(acc1[i], acc2[i], acc3[i])]

        if save:
            with open(file_name, 'wb') as file:
                pickle.dump(result, file)

        self.all_accuracies = result
        return result

    def load_all_accuracies_from_file(self, file_name='all_accuracies_data'):
        with open(file_name, 'rb') as file:
            self.all_accuracies = pickle.load(file)

    def plot_error_bar(self, dic=None, t_val=1.796, noIntens=False, save=False):
        if not dic:
            dic = self.all_accuracies

        ativ_intens = [a+i for a in atividades for i in intensidades]

        y     = []
        y_err = []
        x     = []

        for a in dic:
            arr = np.array(dic[a])
            exp1 = arr[:, 0]
            exp2 = arr[:, 1]
            exp3 = arr[:, 2]

            for i in [exp1, exp2, exp3]:
                mean = np.mean(i)
                stderr = np.std(i) / np.sqrt(len(i))

                simetric = stderr * t_val

                inf = simetric
                sup = simetric

                y += [mean]

                # deve ser no mínimo 0
                if mean - inf <= 0:
                    inf = mean

                # deve ser no máximo 1
                if mean + sup >= 1:
                    sup = 1 - mean

                y_err += [[inf, sup]]

        for i in range(1, len(dic)+1):
            x += [i*6+0, i*6+1, i*6+2]

        fig, ax = plt.subplots()

        if noIntens:
            plt.suptitle('Acurácia de classificador RandomForest por Atividade')
        else: 
            plt.suptitle('Acurácia de classificador RandomForest por Atividade-Intensidade')

        y_err = np.array(y_err)
        y_err = y_err.transpose()

        p1 = ax.errorbar(x[::3], y[::3], y_err[:, ::3], fmt='ob')
        p2 = ax.errorbar(x[1::3], y[1::3], y_err[:, 1::3], fmt='og')
        p3 = ax.errorbar(x[2::3], y[2::3], y_err[:, 2::3], fmt='or')

        p1.set_label('Modelo treinado e testado com dados do mesmo participante')
        p2.set_label('Modelo treinado com dados de todos os participantes exceto pivot')
        p3.set_label('Modelo treinado com dados de todos os participantes acrescidos de 20% dos dados do pivot')
        fig.legend(handles=[p1, p2, p3], loc='lower left', bbox_to_anchor=(0.12, 0.88))

        x_ticks_pos = []
        for i in range(1, len(dic)+1):
            x_ticks_pos += [i*6+1]

        if noIntens:
            ax.set_xticks(x_ticks_pos, atividades, rotation=45)
        else:
            ax.set_xticks(x_ticks_pos, ativ_intens, rotation=45)

        if save:
            plt.savefig('plot_error_bar.png', bbox_inches='tight')
        else:
            plt.show()

    def save_all_data(self, root_dir='./new_data/'):
        participantes = self.data.loc[:, 'Participante'].unique()

        for p in participantes:
            dir_name = 'Aluno' + str(int(p+1))
            os.makedirs(root_dir+dir_name, exist_ok=True)

        for d in aip_gen(self.data):
            participante = d.loc[:, 'Participante'].iloc[0]
            data_dir = root_dir + 'Aluno' + str(int(participante+1)) + '/'
            atividade_intensidade = d.loc[:, 'Atividade'].iloc[0] + d.loc[:, 'Intensidade'].iloc[0]
            d.loc[:, ['x', 'y', 'z', 'tempo']].to_csv(data_dir + atividade_intensidade + '.csv', index=False)

    def time_warp_window(self, W):
        pass


