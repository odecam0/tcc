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

from sklearn.metrics import confusion_matrix

from pdb import set_trace

# UTILS

# DATA é recebido como um DataFrame apenas com dados
# relevantes para o treinamento, uma coluna por feature.
# LABEL é recebido como uma lista de strings.
def fix_data_lstm(data, label):
  # One hot encode

  possible_labels = {}
  for i, x in enumerate(np.unique(label)):
    possible_labels[x] = i
  label = [possible_labels[x] for x in label]
  new_label = tf.keras.utils.to_categorical(label)

  # LSTM recebe como argumento um array de 3 dimensões
  # de shape -> [batch, timesteps, feature]
  new_data = data.to_numpy()
  new_data = new_data.reshape(new_data.shape[0], 1, new_data.shape[1])

  # It is important to keep the original labels, to inform the rest of
  # the process, and to know wich metric is about wich label.
  return new_data, new_label, list(possible_labels.keys())

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

  verbose = 0
  epochs = 15
  batch_size = 64
  Dropout_value = 0.5
  
  model = Sequential()
  model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
  model.add(Dropout(Dropout_value))
  model.add(Dense(100, activation='sigmoid'))
  model.add(Dense(n_outputs, activation='tanh'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
