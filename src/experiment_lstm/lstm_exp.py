import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

import pickle

import ARPC
import arpc_plot as ap
import arpc_utils as au
test = ARPC.Arpc(name='Regular Split with LSTM')
test.load_data('../../dataset/data',
               'Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.txt')
test.raw_data

# Preprocessing data
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

# Segment  data
test.set_windows(size=30) # 3 segundos de dados

# Extracting features
import numpy as np
test.set_features([np.mean, np.std], columns=['x', 'y', 'z', 'module'])

# Código para salvar o objeto com as features extraídas.

# Não dá pra picklar os manips......
def load_pickled_arpc_data(arpc_obj, file_name):
    with open(file_name, 'rb') as file:
        pickled_data = pickle.load(file)
    # Preciso relembrar, como é a passagem de parâmetros em
    # python, imagino que seja por referência, dessa forma não
    # preciso retornar um novo objeto
    arpc_obj.name = pickled_data["name"]
    arpc_obj.raw_data = pickled_data["raw_data"]
    arpc_obj.preprocessed_data = pickled_data["preprocessed_data"]
    arpc_obj.segmented_data = pickled_data["segmented_data"]
    arpc_obj.featured_data = pickled_data["featured_data"]
    # arpc_obj.manips = pickled_data.manips

def save_data_into_pickle(arpc_obj, file_name):
    data_to_pickle = {}
    
    data_to_pickle["name"] = arpc_obj.name
    data_to_pickle["raw_data"] = arpc_obj.raw_data
    data_to_pickle["preprocessed_data"] = arpc_obj.preprocessed_data
    data_to_pickle["segmented_data"] = arpc_obj.segmented_data
    data_to_pickle["featured_data"] = arpc_obj.featured_data
    # data_to_pickle["manips"] = arpc_obj.manips
    
    with open(file_name, 'wb') as file:
        pickle.dump(data_to_pickle, file)

save_data_into_pickle(test, 'arpo_featured')
test = ARPC.Arpc()
load_pickled_arpc_data(test, 'arpo_featured')

import arpc_classification as ac
participantes = [str(i) for i in range(1, 12)]
#test = test.start_new_exp(reuse='featured', name='Split comum com aumento de dados e LSTM')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x: ac.multi_split(x, ac.data_split1, participantes))
test = test.start_new_exp(reuse='featured', name='Split LOSO com LSTM')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x: ac.multi_split(x, ac.loso_split, participantes))
test = test.start_new_exp(reuse='featured', name='Split semi LOSO com LSTM')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x: ac.multi_split(x, ac.semi_loso_split, participantes))

def save_group_of_experiments(arpo, file_name):
    # Irei armazenar as matrizes de confuzão e cada objeto
    # e o nome de cada experimento em uma lista com (<name>, [<matrixes>])
    # para cada experimento
    data_to_pickle = []
    for exp in arpo.exp_gen():
        data_to_pickle+=[(exp.name, exp.confusion_matrixes)]
    
    with open(file_name, 'wb') as file:
        pickle.dump(data_to_pickle, file)

save_group_of_experiments(test, 'group_of_lstm_experiments')

# Seguindo a convenção da função anterior que salva os
# dados do ARPO sem salvar os manips, o arpo aqui será passado
# como parâmetro e não será criado dentro da função, nem
# retornado no fim da função.
#
# Na verdade é necessário retornar um novo elemento sim..
def load_group_of_experiments(arpo, file_name):
    with open(file_name, 'rb') as file:
        pickled_data = pickle.load(file)
    #   pickled_data --------------> [(<name0>, [<matrixes0>]), ... , (<nameN>, [<matrixesN>])]
    arpo.name = pickled_data[0][0] #     /\           /\
    arpo.conusion_matrixes = pickled_data[0][1] #     |
    set_trace()
    
    for exp in pickled_data[1:]:
        arpo = arpo.start_new_exp(name=exp[0])
        arpo.confusion_matrixes = exp[1]
    
    return arpo

test = ARPC.Arpc()
test = load_group_of_experiments(test, 'group_of_lstm_experiments')
test.confusion_matrixes

# Falta  tentar gerar o gráfico simples
# talvez tentando entender melhor como o gráfico está sendo gerado, e
# usar o pdb para debugar a geração de código. Oque provavelmente está
# dando erro é o mecanismo de recuperar os labels das paradas

import arpc_metrics as am

def my_get_plot_data(x):
    return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1])

ap.plot_compare_side_err_bar(test, my_get_plot_data) 

# ===============================================================================
# Trecho que carrega objeto já classificado

import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

import pickle
from pdb import set_trace

import ARPC
import arpc_plot as ap
import arpc_utils as au

def load_group_of_experiments(arpo, file_name):
    with open(file_name, 'rb') as file:
        pickled_data = pickle.load(file)
    #   pickled_data --------------> [(<name0>, [<matrixes0>]), ... , (<nameN>, [<matrixesN>])]
    arpo.name = pickled_data[0][0] #     /\           /\
    arpo.confusion_matrixes = pickled_data[0][1] #     |
    
    for exp in pickled_data[1:]:
        arpo = arpo.start_new_exp(name=exp[0])
        arpo.confusion_matrixes = exp[1]
    
    set_trace()
    
    return arpo

test = ARPC.Arpc()
test = load_group_of_experiments(test, 'group_of_lstm_experiments')
test.confusion_matrixes

import arpc_metrics as am

def my_get_plot_data(x):
    return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1])

ap.plot_compare_side_err_bar(test, my_get_plot_data) 
