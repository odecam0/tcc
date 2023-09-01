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

test = ARPC.Arpc()
test = ie.load_group_of_experiments("regular_classified_arpos")

# Aumentando os dados
# Now augmenting segmented data
test = test.start_new_exp(reuse='segmented', name='Split comum com aumento de dados')

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

from TimeWarpWindow import warp_window
def timewarped(df):
    result, _ = warp_window(df, 5) # A margem para definir um centro pro algoritmo
    return result

participantes = [str(i) for i in range(1, 12)]

test.apply_each_window(funcs=[timewarped], merge=True)

test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x, y: ac.multi_split_window(x, y, ac.data_split1, participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

test = test.start_new_exp(reuse='featured', name='Split LOSO com aumento de dados')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x, y: ac.multi_split_window(x, y, ac.loso_split, participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

test = test.start_new_exp(reuse='featured', name='Split semi LOSO com aumento de dados')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x, y: ac.multi_split_window(x, y,
                                                 lambda x,y,i: ac.semi_loso_split(x, y, i, .2), #driblando um bug
                                                 participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

ie.save_group_of_experiments(test, "dataaug_classified_arpos")

file_name="lstm-dataaug-comparisson.png"

ap.plot_compare_2_set_of_exps(test, 3, am.get_label_accuracy_mean,
                              lambda x: x.confusion_matrixes[0][1],
                              file_name=file_name, gray_experiment_summary_name="experimento sem aumento de dados")

