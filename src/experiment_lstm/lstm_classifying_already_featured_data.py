import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

import pickle

import ARPC
import arpc_plot as ap
import arpc_utils as au

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

# (find-icfile "src/arpc_classification.py" "def eval_lstm")
