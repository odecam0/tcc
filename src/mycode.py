class DataStore:
    def load_data(self, paths, tags):
        """
        paths must be a dictionary, each dictionary key is going to be
        used as the key for the data in this.data\n

        the values of the dictionary must be full paths to the csv,
        paths under the same key will be in the same dataset\n

        if there are nested dictionaries, the keys will be added to the
        resulting DataFrame as another column, with names specified in tags\n

        tags must be an array of strings\n
        """

        def parse_dic(dic, tags_key, tags_val, depth):
            if depth > 0:
                df = pd.DataFrame()

                for key in dic.keys():
                    dic_arg   = dic[key]
                    tags_arg  = tags_val + [key]
                    depth_arg = depth - 1
                    df = pd.concat([df, parse_dic(dic_arg, tags_key, tags_arg, depth_arg)])

                return df

            else:
                assert type(dic) is dict, "Depth of paths and tags are not the same" 
                assert len(tags_val) == len(tags_key), "Depth of tags keys and actual tags don't match"

                df = pd.dataFrame()

                for val in dic.values():
                    assert type(val) is not dict, "Depth of paths and tags are not the same" 
                    df = df.concat([df, pd.read_csv(val)])

                    for tag_v, tag_k in zip(tags_val, tags_key):
                        df[tag_k] = tag_v

                return df

        self.data = {}
        for key in paths.keys():
            self.data[key] = parse_dic(paths[key], tags, [], len(tags)-1)

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())

ds = DataStore()
ds.load_data(full_paths, ['Atividade', 'Intensidade'])

from pprint import pprint
pprint(full_paths)
"""

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
    
"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("code.py").read(), globals())
ativ_intens
p_dir
files
from pprint import pprint as pp
pp(full_paths)
"""

import pandas as pd

def load_dfs(name=p_dir[0], only_acc=False):

    df1   = pd.DataFrame(columns=['x', 'y', 'z', 'tempo', 'sensor'])
    dfAb1 = pd.DataFrame(columns=['x', 'y', 'z', 'tempo', 'sensor'])

    # Para poder fazer a remoção de outliers corretamente
    # Pois é necessário diferenciar um participante de outro
    participant = 0

    for p in p_dir:
        for a in atividades:
            for i in intensidades:
                df_r = pd.read_csv(full_paths[p][a][i], delim_whitespace=True,
                                   names=['x', 'y', 'z', 'tempo', 'sensor'])\
                         .assign(Atividade = a,
                                 Intensidade = i,
                                 participante = participant)

                if only_acc:
                    df_r = df_r.loc[df_r['sensor'] == 'a']

                if p == name:
                    df1 = pd.concat([df1, df_r], ignore_index=True)
                else:
                    dfAb1 = pd.concat([dfAb1, df_r], ignore_index=True)
        
        participant += 1

    return (df1, dfAb1)


"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())

d1, d2 = load_dfs()
d1
d2

d1, d2 = load_dfs(only_acc=True)
d1
d2

d1, d2 = load_dfs('Aluno2')
d1
d2

load_dfs.__defaults__

"""
import numpy as np

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


"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())

d1, d2 = load_dfs(only_acc=True)
d1
d2

import seaborn as           sns
import matplotlib.pyplot as plt

df_testes = d1.loc[(d1['Atividade'] == 'Andando') & (d1['Intensidade'] == 'Vigoroso')] 
ax = sns.boxplot(x=df_testes['x'])
plt.show()
ax = sns.boxplot(x=df_testes['y'])
plt.show()
ax = sns.boxplot(x=df_testes['z'])
plt.show()

df_testes = rem_outliers(df_teste)
ax = sns.boxplot(x=df_testes['x'])
plt.show()
ax = sns.boxplot(x=df_testes['y'])
plt.show()
ax = sns.boxplot(x=df_testes['z'])
plt.show()

"""

def pre_process(dfs, remOut=True, remIni=True):
    d1, d2 = dfs

    if remOut:
        d1out, d2out = pd.DataFrame(), pd.DataFrame()

        participantes = d2['participante'].unique()

        for a in atividades:
            for i in intensidades:
                df_remout = d1.loc[(d1['Atividade'] == a) &
                                   (d1['Intensidade'] == i)]
                df_remout = rem_outliers(df_remout)
                d1out     = pd.concat([d1out, df_remout])

                for p in participantes:
                    df_remout = d2.loc[(d2['Atividade'] == a) &
                                       (d2['Intensidade'] == i) &
                                       (d2['participante'] == p)]
                    df_remout = rem_outliers(df_remout)
                    d2out     = pd.concat([d2out, df_remout])

    else:
        d1out, d2out = d1, d2

    if remIni:
        d1out = d1out.loc[d1out['tempo'] >= 10000]
        d2out = d2out.loc[d2out['tempo'] >= 10000]

    return d1out, d2out

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())

d1, d2 = pre_process(load_dfs(only_acc=True))
d1.head()
d2.head()
d2

d1, d2 = pre_process(load_dfs(only_acc=True), remIni=False)
d1.head()
d2.head()

import seaborn as sns
import matplotlib.pyplot as plt

def plot_boxsns(x):
    ax = sns.boxplot(x=x)
    plt.show()

d1, d2 = pre_process(load_dfs(only_acc=True), remOut=False)
d_teste = d1.loc[(d1['Atividade'] == 'Andando') & (d1['Intensidade'] == 'Moderado')]
plot_boxsns(d_teste['x'])
plot_boxsns(d_teste['y'])
plot_boxsns(d_teste['z'])

d_teste = d2.loc[(d2['Atividade'] == 'Andando') & (d2['Intensidade'] == 'Moderado') & (d2['participante'] == 3.)]
plot_boxsns(d_teste['x'])
plot_boxsns(d_teste['y'])
plot_boxsns(d_teste['z'])

d1, d2 = pre_process(load_dfs(only_acc=True))
d_teste = d1.loc[(d1['Atividade'] == 'Andando') & (d1['Intensidade'] == 'Moderado')]
plot_boxsns(d_teste['x'])
plot_boxsns(d_teste['y'])
plot_boxsns(d_teste['z'])

d_teste = d2.loc[(d2['Atividade'] == 'Andando') & (d2['Intensidade'] == 'Moderado') & (d2['participante'] == 3.)]
plot_boxsns(d_teste['x'])
plot_boxsns(d_teste['y'])
plot_boxsns(d_teste['z'])
"""

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
# https://pandas.pydata.org/docs/reference/window.html

def segment_and_features(dfs, window_size=10 ,getMean=True, getMedian=True, getVar=True, getStd=True):
    df1, df2 = dfs 

    df1_out = pd.DataFrame()
    df2_out = pd.DataFrame()

    for a in atividades:
        for i in intensidades:
            df = df1.loc[(df1['Atividade'] == a) & (df1['Intensidade'] == i)]
            rolling_w = df[['x', 'y', 'z', 'tempo']].rolling(window=window_size, center=True,  on='tempo')

            df_aux = pd.DataFrame()

            if getMean:
                df_aux['mean_x'] = rolling_w.mean().dropna()['x']
                df_aux['mean_y'] = rolling_w.mean().dropna()['y']
                df_aux['mean_z'] = rolling_w.mean().dropna()['z']

            if getMedian:
                df_aux['median_x'] = rolling_w.median().dropna()['x']
                df_aux['median_y'] = rolling_w.median().dropna()['y']
                df_aux['median_z'] = rolling_w.median().dropna()['z']

            if getVar:
                df_aux['var_x'] = rolling_w.var().dropna()['x']
                df_aux['var_y'] = rolling_w.var().dropna()['y']
                df_aux['var_z'] = rolling_w.var().dropna()['z']

            if getStd:
                df_aux['std_x'] = rolling_w.std().dropna()['x']
                df_aux['std_y'] = rolling_w.std().dropna()['y']
                df_aux['std_z'] = rolling_w.std().dropna()['z']

            df_aux['tempo'] = rolling_w.mean().dropna()['tempo']

            df_aux['Atividade'] = a
            df_aux['Intensidade'] = i
                
            df1_out = pd.concat([df1_out, df_aux])

    
    participantes = df2['participante'].unique()

    for p in participantes:
        for a in atividades:
            for i in intensidades:

                df = df2.loc[(df2['Atividade'] == a) &
                             (df2['Intensidade'] == i) &
                             (df2['participante'] == p)]

                rolling_w = df[['x', 'y', 'z', 'tempo']].rolling(window=window_size, center=True, on='tempo')

                df_aux = pd.DataFrame()

                if getMean:
                    df_aux['mean_x'] = rolling_w.mean().dropna()['x']
                    df_aux['mean_y'] = rolling_w.mean().dropna()['y']
                    df_aux['mean_z'] = rolling_w.mean().dropna()['z']

                if getMedian:
                    df_aux['median_x'] = rolling_w.median().dropna()['x']
                    df_aux['median_y'] = rolling_w.median().dropna()['y']
                    df_aux['median_z'] = rolling_w.median().dropna()['z']

                if getVar:
                    df_aux['var_x'] = rolling_w.var().dropna()['x']
                    df_aux['var_y'] = rolling_w.var().dropna()['y']
                    df_aux['var_z'] = rolling_w.var().dropna()['z']

                if getStd:
                    df_aux['std_x'] = rolling_w.std().dropna()['x']
                    df_aux['std_y'] = rolling_w.std().dropna()['y']
                    df_aux['std_z'] = rolling_w.std().dropna()['z']

                df_aux['Atividade'] = a
                df_aux['Intensidade'] = i
                df_aux['Participante'] = p

                df2_out = pd.concat([df2_out, df_aux])

    return df1_out, df2_out


"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
segment_and_features(pre_process(load_dfs(only_acc=True)))
segment_and_features(pre_process(load_dfs(only_acc=True)), getMean=False)

getMedian=True
getVar=True
getStd=True
"""

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, d2 = pre_process(load_dfs(only_acc=True))
d1
rollw = d1[['x', 'y', 'z', 'tempo']].rolling(window=10, on="tempo")
rollw.mean()
rollw.median()
rollw.var()
rollw.std()
rollw.min()
rollw.max()

rollw.corr()
rollw.cov()
rollw.skew()
rollw.kurt()

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pdb

def train_model_split1(df, noIntens=False):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Primeira classificação: Utilizando os próprios dados do participante para treinar o modelo
    X = df.loc[:, ~df.columns.isin(['Atividade', 'Intensidade', 'tempo'])]

    if noIntens:
        y = df['Atividade']
    else:
        y = [i + j for i, j in zip(df['Atividade'], df['Intensidade'])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    rf.fit(X_train, y_train)

    prediction = rf.predict(X_test)

    cmat = metrics.confusion_matrix(y_test, prediction, labels=rf.classes_)

    cvs = cross_val_score(rf, X, y, cv=10)

    return cmat, prediction, rf, cvs

def train_model_split2(df1, df2, noIntens=False):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Segunda classificação: Utilizandoo os dados de todos os outros participantes para treinar o modelo
    X_test = df1.loc[:, ~df1.columns.isin(['Atividade', 'Intensidade', 'tempo'])]

    if noIntens:
        y_test = df1['Atividade']
        y      = df2['Atividade']
    else:
        y_test = [i + j for i, j in zip(df1['Atividade'], df1['Intensidade'])]
        y = [i + j for i, j in zip(df2['Atividade'], df2['Intensidade'])]

    X = df2.loc[:, ~df2.columns.isin(['Atividade', 'Intensidade', 'Participante'])]

    rf.fit(X, y)

    prediction = rf.predict(X_test)

    cmat = metrics.confusion_matrix(y_test, prediction)

    return cmat, prediction, rf

def treina_modelos(dfs, pct=0.2, noIntens=False):
    df1, df2 = dfs

    new_df1 = pd.DataFrame()
    new_df2 = df2.copy()

    for a in atividades:
        for i in intensidades:
            df = df1.loc[(df1['Atividade'] == a) & (df1['Intensidade'] == i)]

            df_removed = df.sample(frac=pct)
            df_remained = df.drop(df_removed.index)
            
            new_df1 = pd.concat([new_df1, df_remained])
            new_df2 = pd.concat([new_df2, df_removed])

    cmat1, prediction1, _, _ = train_model_split1(df1, noIntens=True)

    cmat2, prediction2, _ = train_model_split2(df1, df2, noIntens=True)

    # Terceira classificação: Utilizando os dados de outros participantes com uma parcela dos dados do partipante onde o modelo será testado
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    X = new_df2.loc[:, ~new_df2.columns.isin(['Atividade', 'Intensidade', 'Participante', 'tempo'])]

    X_test = new_df1.loc[:, ~new_df1.columns.isin(['Atividade', 'Intensidade', 'tempo'])]

    if noIntens:
        y      = new_df2['Atividade']
        y_test = new_df1['Atividade']
    else:
        y      = [i + j for i, j in zip(new_df2['Atividade'], new_df2['Intensidade'])]
        y_test = [i + j for i, j in zip(new_df1['Atividade'], new_df1['Intensidade'])]

    rf.fit(X, y)

    prediction_3 = rf.predict(X_test)
    
    cmat3 = metrics.confusion_matrix(y_test, prediction_3)

    return cmat1, cmat2, cmat3

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, d2 = segment_and_features(pre_process(load_dfs(only_acc=True)))
cmat1, cmat2, cmat3 = treina_modelos((d1, d2))

d1, d2 = segment_and_features(pre_process(load_dfs(name='Aluno2', only_acc=True)))
d1, d2 = segment_and_features(pre_process(load_dfs(name='Aluno3', only_acc=True)))
d1, d2 = segment_and_features(pre_process(load_dfs(name='Aluno4', only_acc=True)))
d1, d2 = segment_and_features(pre_process(load_dfs(name='Aluno5', only_acc=True)))
d1, d2 = segment_and_features(pre_process(load_dfs(name='Aluno10', only_acc=True)))


cmat, _, _, cvs = train_model_split1(d1)
cmat
cvs
cvs.mean()

import sklearn
import matplotlib.pyplot as plt
cmat_disp = sklearn.metrics.ConfusionMatrixDisplay(cmat, display_labels=ativ_intens).plot(cmap="Blues")
cmat_disp.ax_.set(ylabel="Rótulo verdadeiro", xlabel="Rótulo previsto")
plt.xticks(rotation=45, ha='right')
plt.title("Matriz de confusão")
plt.savefig('mat_1.png', bbox_inches='tight')
plt.show()

cmat_disp = sklearn.metrics.ConfusionMatrixDisplay(cmat2).plot(); plt.show()
cmat_disp = sklearn.metrics.ConfusionMatrixDisplay(cmat3).plot(); plt.show()

"""
from pprint import pprint

def something():
    alunos = ['Aluno' + str(i) for i in range(1, 12)]
    results = []
    for a in alunos:
        df, _ = segment_and_features(pre_process(load_dfs(name=a, only_acc=True)))
        cmat, _, _, cvs = train_model_split1(df)
        results += [(a, cmat, cvs, cvs.mean())]

    return results

        
"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
results = something()
pprint(results)

vec_mats = [i for _, i, _, _ in results]
arr = np.array(vec_mats)
arr.shape
arr.sum(axis=0)
cmat = arr.sum(axis=0)

import sklearn
import matplotlib.pyplot as plt
cmat_disp = sklearn.metrics.ConfusionMatrixDisplay(cmat, display_labels=ativ_intens).plot(cmap="Blues")
cmat_disp.ax_.set(ylabel="Rótulo verdadeiro", xlabel="Rótulo previsto")
plt.xticks(rotation=45, ha='right')
plt.title("Matriz de confusão")
plt.savefig('mat_all.png', bbox_inches='tight')
plt.show()

vec_cvs = [i for _, _, i, _ in results]
arr = np.array(vec_cvs)
arr.reshape(110,)
arr.sum()
arr.size
arr.sum()/arr.size

"""


def get_accuracy(cmat):
    result = []

    n_classes = cmat.shape[0]

    for i in range(n_classes):
        n_class       = cmat[i].sum()
        n_right_pred  = cmat.diagonal()[i]
        result = result + [n_right_pred / n_class]

    return result


"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, d2 = segment_and_features(pre_process(load_dfs(only_acc=True)))
cmat1, cmat2, cmat3 = treina_modelos((d1, d2))
get_accuracy(cmat1)
get_accuracy(cmat2)
get_accuracy(cmat3)

cmat2
"""

def get_all_accuracies(noIntens = False):

    result = {}

    if noIntens:
        for a in atividades:
            result[a] = [] 
    else:
        for a in ativ_intens:
            result[a] = [] 

    for p in p_dir:
        d1, d2 = load_dfs(name=p, only_acc=True)
        d1, d2 = pre_process((d1, d2))
        d1, d2 = segment_and_features((d1, d2))

        cmat1, cmat2, cmat3 = treina_modelos((d1, d2), noIntens=noIntens)

        acc1 = get_accuracy(cmat1)
        acc2 = get_accuracy(cmat2)
        acc3 = get_accuracy(cmat3)

        if noIntens:
            for a, i in zip(atividades, range(len(atividades))):
                result[a] += [(acc1[i], acc2[i], acc3[i])]
        else:
            for a, i in zip(ativ_intens, range(len(ativ_intens))):
                result[a] += [(acc1[i], acc2[i], acc3[i])]
        
    return result

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
dic_acc = get_all_accuracies()

import pickle

with open('pickled_data', 'wb') as file:
    pickle.dump(dic_acc, file)

with open('pickled_data', 'rb') as file:
    new_var = pickle.load(file)

new_var

import numpy as np
from pprint import pprint
pprint(new_var['AndandoLeve'])
np.array(new_var['AndandoLeve'])
another = np.array(new_var['AndandoLeve'])
another[:,0].shape
another[:,1]
another[:,2]
"""
import matplotlib.pyplot as plt

def plot_error_bar(dic, t_val=1.796, noIntens=False):

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

    plt.show()

# https://onlinestatbook.com/2/estimation/mean.html
# https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf
# 95% com 11 valores = 1.796
   
"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
dic_acc = get_all_accuracies(noIntens = True)
plot_error_bar(dic_acc, noIntens=True)
"""

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
plot_error_bar(dic)

import pickle
with open('pickled_data', 'rb') as file:
    dic = pickle.load(file)

plot_error_bar(dic)
"""
 
def plot_compare(df_raw, df_feature):
    
    fig, axs = plt.subplots(len(atividades), len(intensidades))

    for a, n1 in zip(atividades, range(len(atividades))):
        for i, n2 in zip(intensidades, range(len(intensidades))):

            sub_d_raw = df_raw.loc[(df_raw['Atividade'] == a) & (df_raw['Intensidade'] == i)]
            sub_d_feature = df_feature.loc[(df_feature['Atividade'] == a) & (df_feature['Intensidade'] == i)]
                     
            axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['x'], '--', color='grey')
            axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['y'], '-', color='grey')
            axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['z'], ':', color='grey')

            axs[n1][n2].plot(sub_d_feature['tempo'], sub_d_feature.iloc[:, 0], c='r')
            axs[n1][n2].plot(sub_d_feature['tempo'], sub_d_feature.iloc[:, 1], c='b')
            axs[n1][n2].plot(sub_d_feature['tempo'], sub_d_feature.iloc[:, 2], c='c')

    plt.show()

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, _ = pre_process(load_dfs(name='Aluno2', only_acc=True))
d2, _ = segment_and_features(pre_process(load_dfs(name='Aluno2', only_acc=True)))

d1
d2

d2.columns
d2.drop(columns=d2.columns[[0, 1, 2]])

df_mean   = d2[['mean_x', 'mean_y', 'mean_z', 'tempo', 'Atividade', 'Intensidade']]
df_median = d2[['median_x', 'median_y', 'median_z', 'tempo', 'Atividade', 'Intensidade']]
df_var    = d2[['var_x', 'var_y', 'var_z', 'tempo', 'Atividade', 'Intensidade']]
df_std    = d2[['std_x', 'std_y', 'std_z', 'tempo', 'Atividade', 'Intensidade']]

plot_compare(d1, d1)
plot_compare(d1, df_mean)
plot_compare(d1, df_median)
plot_compare(d1, df_var)
plot_compare(d1, df_std)

"""

def plot_all(p):

    df = pd.DataFrame(columns=['x', 'y', 'z', 'tempo', 'sensor'])

    for a in atividades:
        for i in intensidades:
            df_r = pd.read_csv(full_paths[p][a][i], delim_whitespace=True,
                               names=['x', 'y', 'z', 'tempo', 'sensor'])\
                     .assign(Atividade = a,
                             Intensidade = i)

            df_r = df_r.loc[df_r['sensor'] == 'a']

            df = pd.concat([df, df_r], ignore_index=True)
    

    fig, axs = plt.subplots(len(atividades), len(intensidades), sharey=True)
    fig.suptitle(p)

    for a, n1 in zip(atividades, range(len(atividades))):
        for i, n2 in zip(intensidades, range(len(intensidades))):

            if n1 != len(atividades) - 1:
                axs[n1][n2].set_xticks([])

            sub_d_raw = df.loc[(df['Atividade'] == a) & (df['Intensidade'] == i)]
                     
            axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['x'], linewidth=0.5, color='blue')
            axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['y'], linewidth=0.5, color='red')
            axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['z'], linewidth=0.5, color='green')

    for a, n1 in zip(atividades, range(len(atividades))):
        axs[n1][0].set_ylabel(a)

    for i, n2 in zip(intensidades, range(len(intensidades))):
        axs[len(atividades)-1][n2].set_xlabel(i)

    plt.savefig(p+'.png')
    #plt.show()


"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, _ = pre_process(load_dfs(name='Aluno2', only_acc=True))
plot_all(p_dir[0])
plot_all(p_dir[1])
plot_all(p_dir[2])
plot_all(p_dir[3])
plot_all(p_dir[4])
plot_all(p_dir[5])
plot_all(p_dir[6])
plot_all(p_dir[7])
plot_all(p_dir[8])
plot_all(p_dir[9])
plot_all(p_dir[10])
"""
import sklearn
import matplotlib.pyplot as plt

def aip_gen(df):
    'Generator for activity, intensity and participant pandas DataFrame'
    try:
        atividades = df['Atividade'].unique()
        intensidades = df['Intensidade'].unique()
        participantes = df['Participante'].unique()
    except KeyError as err:
        print('Uma das colunas do dataframe está com nome errado?')
        print('Na função aip_gen')
        print('Coluna que deu errado: ', err.args[0])

    for a in atividades:
        for i in intensidades:
            for p in participantes:
                yield df.loc[(df['Atividade'] == a) &
                             (df['Intensidade'] == i) &
                             (df['Participante'] == p)]

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, _ = pre_process(load_dfs(name='Aluno2', only_acc=True))
d1
aip_gen(d1)
d1 = d1.rename(columns={'participante':'Participante'})
aip_gen(d1)
list(aip_gen(d1))
"""

""" Descobrindo os atributos da exceção 'KeyError'
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
d1, _ = pre_process(load_dfs(name='Aluno2', only_acc=True))

from pprint import pprint

from inspect import getmembers

try:
    d1['X']
except KeyError as err:
    pprint(getmembers(err))

"""

from pdb import set_trace

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

    def fix_dup(self):
        df = self.data

        for df_aux in aip_gen(df):
            for id in df_aux.index[1:]:
                if df_aux.iloc[id]['tempo'] < df_aux.iloc[id-1]['tempo']:
                    tempo_base = df_aux.iloc[id-1]['tempo']
                    id_base    = id
                    break
            else:
                index_tempo = list(df_aux.columns).index('tempo')
                df_aux.iloc[id_base:, index_tempo] += tempo_base

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
        self.data = df
    
    def segment_extract_features(self, window_size=10):
        df = self.data
        df1_out = pd.DataFrame()

        for df_local in aip_gen(df):
            df_local  = df_local.astype({'tempo': int})
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

        # set_trace()
        cmat_disp = sklearn.metrics.ConfusionMatrixDisplay(cmat, display_labels=ativ_intens).plot(cmap="Blues")
        cmat_disp.ax_.set(ylabel="Rótulo verdadeiro", xlabel="Rótulo previsto")
        plt.xticks(rotation=45, ha='right')
        plt.title("Matriz de confusão")

        if save:
            plt.savefig('mat_mesmo_conjunto.png', bbox_inches='tight')
        else:
            plt.show()

    def plot_all(self, save=False):
        for df in aip_gen(self.data):
            # atividades e intensidades foram declarados no início do arquivo
            participante = df['Participante'][0]

            fig, axs = plt.subplots(len(atividades), len(intensidades), sharey=True)
            fig.suptitle(participante)

            for a, n1 in zip(atividades, range(len(atividades))):
                for i, n2 in zip(intensidades, range(len(intensidades))):

                    if n1 != len(atividades) - 1:
                        axs[n1][n2].set_xticks([])

                    sub_d_raw = df.loc[(df['Atividade'] == a) & (df['Intensidade'] == i)]

                    axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['x'], linewidth=0.5, color='blue')
                    axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['y'], linewidth=0.5, color='red')
                    axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['z'], linewidth=0.5, color='green')

            for a, n1 in zip(atividades, range(len(atividades))):
                axs[n1][0].set_ylabel(a)

            for i, n2 in zip(intensidades, range(len(intensidades))):
                axs[len(atividades)-1][n2].set_xlabel(i)

            if save:
                plt.savefig(participante+'.png')
            else:
                plt.show()
        
"""
# (defun b () (interactive) (find-icfile "mycode.py" "sd = SensorData()"))
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
sd = SensorData()
sd.remove_outliers()
sd.remove_beginning()
sd.segment_extract_features()
sd.confusion_matrix()

sd = SensorData()
sd.plot_all()
"""


def kfold_all(k):
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

    df_noout = pd.DataFrame()
    
    # Removing outliers
    for a in atividades:
        for i in intensidades:
            for p in participantes:
                df_remout = df.loc[(df['Atividade'] == a) &
                                   (df['Intensidade'] == i) &
                                   (df['Participante'] == p)]
                df_remout = rem_outliers(df_remout)
                df_noout  = pd.concat([df_noout, df_remout])

    df = df_noout

    # Removing first 10 seconds
    df = df.loc[df['tempo'] >= 10000]

    # Segmenting and extracting features
    window_size = 10
    df1_out = pd.DataFrame()

    for a in atividades:
        for i in intensidades:
            for p in participantes:
                df_local  = df.loc[(df['Atividade'] == a) & (df['Intensidade'] == i) & (df['Participante'] == p)]
                df_local  = df_local.astype({'tempo': int})
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

                df_aux['Atividade'] = a
                df_aux['Intensidade'] = i
                df_aux['Participante'] = p

                df1_out = pd.concat([df1_out, df_aux])

    df = df1_out

    # Building and testing classifier
    X = df.drop(columns=['tempo', 'Atividade', 'Intensidade', 'Participante'])
    y = df['Atividade'] + df['Intensidade']

    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    cvs = cross_val_score(rf, X, y, cv=k)

    return cvs

"""
 (eepitch-python)
 (eepitch-kill)
 (eepitch-python)
exec(open("mycode.py").read(), globals())
kfold_all(10)
"""
