from arpc_utils import p_gen

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import pandas as pd

from pdb import set_trace

# Data Splits

def multi_split(x, func, select=['1', '2']):
    r1, r2 = [], []
    for i in select:
        aux = func(x, i)
        r1+= aux[0]
        r2+= aux[1]
    return r1, r2

def data_split1(dataset, select='1'):
    """
    Escolhe o participante com SELECT e retorna 75% dos dados deste
    participante como treino e 25% como avaliação
    """

    for df in p_gen(dataset, select=[select]):
        X = df.drop(columns=['atividade', 'intensidade', 'tempo', 'sensor', 'participante'])
        y = [i + j for i, j in zip(df['atividade'], df['intensidade'])]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return [(X_train, y_train)], [(X_test, y_test)]

def loso_split(dataset, select='1'):
    for df in p_gen(dataset, select=[select]):
        eval_data = df.drop(columns=['atividade', 'intensidade', 'tempo', 'sensor', 'participante'])
        eval_label = [i + j for i, j in zip(df['atividade'], df['intensidade'])]

    other_ones = [i for i in dataset.participante.unique() if i != select]

    train_data = pd.DataFrame()
    train_label = []
    for df in p_gen(dataset, select=other_ones):
        train_data = pd.concat([train_data,
                                df.drop(columns=['atividade', 'intensidade', 'tempo',
                                                 'sensor', 'participante'])])
        train_label += [i + j for i, j in zip(df['atividade'], df['intensidade'])]

    return [(train_data, train_label)], [(eval_data, eval_label)]

def semi_loso_split(dataset, select='1', pct=.2):
    train_split, eval_split = loso_split(dataset, select)
    train_split, eval_split = train_split[0], eval_split[0]

    train_df = train_split[0].assign(label=train_split[1])
    eval_df  =  eval_split[0].assign(label= eval_split[1])

    # remove os PCT% de dados de eval_df de forma estratificada
    # e os adciona em train_df
    for label in eval_df.label.unique():
        df = eval_df.loc[eval_df.label == label]
        df_removed  = df.sample(frac=pct)
        train_df = pd.concat([train_df, df_removed])
        eval_df = eval_df.drop(df_removed.index)

    train_data = train_df.drop(columns=['label'])
    train_label = train_df.label
    eval_data = eval_df.drop(columns=['label'])
    eval_label = eval_df.label

    return [(train_data, train_label)], [(eval_data, eval_label)]

# Train models

def train_randomforest(train_data, train_label):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_data, train_label)
    return rf

# Eval models

def eval_randomforest(trained_model, eval_data, eval_label):
    prediction = trained_model.predict(eval_data)
    cmat = metrics.confusion_matrix(eval_label, prediction, labels=trained_model.classes_)
    return cmat
