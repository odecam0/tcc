# Adcionando pasta com código ao path para
# poder importar os módulos desenvolvidos
import sys
from pathlib import Path
p = Path("../")
sys.path.append(str(p.resolve()))

import ARPC
import arpc_plot as ap
import arpc_utils as au

import ie_data as ie

from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
import keras

import tensorflow as tf

import arpc_classification as ac

import arpc_metrics as am


test = ARPC.Arpc(name='Regular Split with LSTM')
test.load_data('../../dataset/data',
               'Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.txt')

import manips
import arpc_utils as au
test.add_manip(au.get_acc_data)
test.add_manip(lambda x: manips.fix_dup(x, remFirst=True))
classes = [(1., 'Deitado', 'Moderado')]
classes += [(4., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
classes += [(7., 'Deitado', i) for i in ['Leve', 'Moderado', 'Vigoroso']]
test.add_manip(lambda x: manips.rotate_class(x, classes, [0, 0, 1]))
test.add_manip(manips.remove_outliers)
test.add_manip(manips.remove_beginning)
test.add_manip(manips.scale_data)
test.add_manip(manips.set_vec_module)
test.do_manip()

test.set_windows(size=30) # 3 segundos de dados

from pdb import set_trace
import numpy as np
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
  # model.add(Dropout(Dropout_value))
  model.add(Dense(100, activation='sigmoid'))
  model.add(Dense(n_outputs, activation='tanh'))
  model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['categorical_accuracy'])

  model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, verbose=verbose)

  # To keep compatible with sklearn models
  model.classes_ = original_label

  return model

def eval_lstm(trained_model, eval_data, eval_label):
  eval_data, eval_label, original_label = fix_data_lstm(eval_data, eval_label)
  # Aqui é necessário testar o modelo de forma que ele retorne uma matriz de confusão.

  predicted_label = trained_model.predict(eval_data)

  eval_label = np.argmax(eval_label, axis=-1)
  predicted_label = np.argmax(predicted_label, axis=-1)

  eval_label = [original_label[i] for i in eval_label]
  predicted_label = [original_label[i] for i in predicted_label]

  return confusion_matrix(eval_label, predicted_label)


# Classificação
participantes = [str(i) for i in range(1, 12)]
test.classify(train_lstm, eval_lstm,
              lambda x, y: ac.multi_split_window(x, y, ac.data_split1, participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

test = test.start_new_exp(reuse='featured', name='Split LOSO com LSTM')
test.classify(train_lstm, eval_lstm,
              lambda x, y: ac.multi_split_window(x, y, ac.loso_split, participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

test = test.start_new_exp(reuse='featured', name='Split semi LOSO com LSTM')
test.classify(train_lstm, eval_lstm,
              lambda x, y: ac.multi_split_window(x, y,
                                                 lambda x,y,i: ac.semi_loso_split(x, y, i, .2), #driblando um bug
                                                 participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

ie.save_group_of_experiments(test, "regular_classified_arpos")

def my_get_plot_data(x):
    return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1])

ap.plot_compare_err_bar(test, my_get_plot_data, ap.group_labels_by_first_word)


# # Aumentando os dados
# # Now augmenting segmented data
# test = test.start_new_exp(reuse='segmented', name='Split comum com aumento de dados')
# from TimeWarpWindow import warp_window
# def timewarped(df):
#     result, _ = warp_window(df, 5) # A margem para definir um centro pro algoritmo
#     return result

# test.apply_each_window(funcs=[timewarped])

# test.classify(train_lstm, eval_lstm,
#               lambda x, y: ac.multi_split_window(x, y, ac.data_split1, participantes),
#               featured_data=False,
#               prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

# test = test.start_new_exp(reuse='featured', name='Split LOSO com LSTM')
# test.classify(train_lstm, eval_lstm,
#               lambda x, y: ac.multi_split_window(x, y, ac.loso_split, participantes),
#               featured_data=False,
#               prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

# test = test.start_new_exp(reuse='featured', name='Split semi LOSO com LSTM')
# test.classify(train_lstm, eval_lstm,
#               lambda x, y: ac.multi_split_window(x, y,
#                                                  lambda x,y,i: ac.semi_loso_split(x, y, i, .2), #driblando um bug
#                                                  participantes),
#               featured_data=False,
#               prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

# ap.plot_compare_2_set_of_exps(test, 3, am.get_label_accuracy_mean,
#                               lambda x: x.confusion_matrixes[0][1],
#                               file_name=file_name, gray_experiment_summary_name="experimento sem aumento de dados")
