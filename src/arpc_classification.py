from arpc_utils import p_gen

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

import pandas as pd
import numpy as np

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

import keras

from sklearn.metrics import confusion_matrix

from overload import *

from pdb import set_trace

import re

# UTILS

# DATA é recebido como um DataFrame apenas com dados
# relevantes para o treinamento, uma coluna por feature.
# LABEL é recebido como uma lista de strings.
def fix_data_lstm(data, label):
  """
  Will make one-hot-enconding.
  """
  possible_labels = {}
  for i, x in enumerate(np.unique(label)):
    possible_labels[x] = i
  label = [possible_labels[x] for x in label]
  new_label = tf.keras.utils.to_categorical(label)

  # It is important to keep the original labels, to inform the rest of
  # the process, and to know wich metric is about wich label.
  return data, new_label, list(possible_labels.keys())

# Data Splits
# multi_split and multi_split_window are responsible for executing the data split procedure
# on diferent selections of participants and storing the data in a conventional way. being:
#         ( [<Array of training data], [<Corresponding array of testing data] )
def multi_split(x, func, select=['1', '2']):
    r1, r2 = [], []
    for i in select:
        aux = func(x, i)
        r1+= aux[0]
        r2+= aux[1]
    return r1, r2
def multi_split_window(x, y, func, select=['1', '2']):
    r1, r2 = [], []
    for i in select:
        aux = func(x, y, i)
        r1+= aux[0]
        r2+= aux[1]
    return r1, r2

def prepare_window(segmented_data : dict, columns):
    """
    ...
    """
    labels = []
    data = []
    for key in segmented_data.keys():
        # Adciona várias janelas em data e adciona a mesma quantidade
        # de labels em labels
        data += [df.loc[:,columns].to_numpy() for df in segmented_data[key]]
        labels += [key] * len(segmented_data[key])

    return np.array(data), labels

@overload
def data_split1(dataset: pd.DataFrame, select:str='1'):
    """
    Escolhe o participante com SELECT e retorna 75% dos dados deste
    participante como treino e 25% como avaliação
    """

    for df in p_gen(dataset, select=[select]):
        X = df.drop(columns=['atividade', 'intensidade', 'tempo', 'sensor', 'participante'])
        y = [i + j for i, j in zip(df['atividade'], df['intensidade'])]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return [(X_train, y_train)], [(X_test, y_test)]
@data_split1.add
def data_split1(data, label, select='1'):
    # selecionar o participante
    selected_samples = [re.search(r'\d+', l).group() == select for l in label]
    data = data[selected_samples]
    label = [re.sub(r'\d+', '', l) for l in label]
    label = np.array(label)
    label = label[selected_samples]

    X_train, X_test, y_train, y_test = train_test_split(data, label, random_state=0)

    return [(X_train, y_train)], [(X_test, y_test)]

@overload
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
@loso_split.add
def loso_split(data, label, select='1'):
    selected_samples = [re.search(r'\d+', l).group() == select for l in label]
    X_test = data[selected_samples]
    label = [re.sub(r'\d+', '', l) for l in label]
    label = np.array(label)
    y_test = label[selected_samples]

    X_train = data[[not i for i in selected_samples]]
    y_train = label[[not i for i in selected_samples]]

    return [(X_train, y_train)], [(X_test, y_test)]

@overload
def semi_loso_split(dataset: pd.DataFrame, select='1', pct=.2):
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
@semi_loso_split.add
def semi_loso_split(data, label, select='1', pct=.2):
    # O algoritmo é:
    #  Separa os dados como em loso_split,
    #  Nos dados de teste, para cada classe, serão selecionados pct%, de
    #  amostras aleatórias que serão adcionados nos dados de teste.
    #  Esta seleção será feita em forma de uma lista com True correspondendo
    #  aos dados selecionados. Fancy indexing
    selected_samples = [re.search('\d+', l).group() == select for l in label]
    X_test = data[selected_samples]
    label = [re.sub(r'\d+', '', l) for l in label]
    label = np.array(label)
    y_test = label[selected_samples]

    X_train = data[[not i for i in selected_samples]]
    y_train = label[[not i for i in selected_samples]]

    sample_index = np.array([])
    rng = np.random.default_rng()
    for label in np.unique(y_test):
        labels_fancy_index = [l == label for l in y_test] # List com True onde é de determinado label.
        # Agora é preciso coletar uma amostra destes dados que são True, deixando apenas pct$
        # desses dados ainda como True, e transformando em False todos os outros.
        labels_index_4_this_class = [i for i, l in enumerate(labels_fancy_index) if l]
        # /\ Lista com os indices onde l_f_i é True, onda os dados são da determinada classe
        n_labels = len(labels_index_4_this_class)
        sample_index = np.concatenate((np.array([]), rng.choice(labels_index_4_this_class, int(pct * n_labels), replace=False)),
                                      dtype=np.integer, casting='unsafe')

    X_train = np.concatenate((X_train, X_test[sample_index]))
    y_train = np.concatenate((y_train, y_test[sample_index]))

    inverse_sample_index = [i for i in range(len(y_test)) if i not in sample_index]
    X_test = X_test[inverse_sample_index]
    y_test = y_test[inverse_sample_index]

    return [(X_train, y_train)], [(X_test, y_test)]

# Train models

# It is important that the model returned by the function has an attribute called
# classes_, with all the possible classes in the right order. If it is not an SKLearn
# model, then it must be created explicitly.

def train_randomforest(train_data, train_label):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_data, train_label)
    return rf

def train_lstm(train_data, train_label):
  # Aqui será necessário tratar os dados recebidos para que
  # estejam em conformes com como o modelo os espera
  train_data, train_label, original_label  = fix_data_lstm(train_data, train_label)

  # Aqui é preciso instanciar o modelo com seus parâmetros e
  # treina-lo com os dados, e retornalo.
  n_timesteps = train_data.shape[1]
  n_features = train_data.shape[2]

  n_outputs = train_label.shape[1]

  verbose = 1
  epochs = 15
  batch_size = 32
  Dropout_value = 0.5
  
  model = Sequential()
  model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
  # model.add(Dropout(Dropout_value))
  model.add(Dense(100, activation='sigmoid'))
  model.add(Dense(n_outputs, activation='tanh'))
  model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['categorical_accuracy'])

  model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, verbose=verbose)

  # To keep compatible with sklearn models
  model.classes_ = original_label

  return model

# Eval models

def eval_lstm(trained_model, eval_data, eval_label):
  eval_data, eval_label, original_label = fix_data_lstm(eval_data, eval_label)
  # Aqui é necessário testar o modelo de forma que ele retorne uma matriz de confusão.

  predicted_label = trained_model.predict(eval_data)

  eval_label = np.argmax(eval_label, axis=-1)
  predicted_label = np.argmax(predicted_label, axis=-1)

  eval_label = [original_label[i] for i in eval_label]
  predicted_label = [original_label[i] for i in predicted_label]

  return confusion_matrix(eval_label, predicted_label)

def eval_randomforest(trained_model, eval_data, eval_label):
    prediction = trained_model.predict(eval_data)
    cmat = metrics.confusion_matrix(eval_label, prediction, labels=trained_model.classes_)
    return cmat
