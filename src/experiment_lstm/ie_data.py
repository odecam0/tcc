import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))
import ARPC

import pickle

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

def save_group_of_experiments(arpo, file_name):
    # Irei armazenar as matrizes de confuzão e cada objeto
    # e o nome de cada experimento em uma lista com (<name>, [<matrixes>])
    # para cada experimento
    data_to_pickle = []
    for exp in arpo.exp_gen():
        data_to_pickle+=[(exp.name, exp.confusion_matrixes)]

    with open(file_name, 'wb') as file:
        pickle.dump(data_to_pickle, file)

def load_group_of_experiments(file_name):
    arpo = ARPC.Arpc()

    with open(file_name, 'rb') as file:
        pickled_data = pickle.load(file)
    #   pickled_data --------------> [(<name0>, [<matrixes0>]), ... , (<nameN>, [<matrixesN>])]

    pickled_data = pickled_data[::-1]

    arpo.name = pickled_data[0][0] #     /\           /\
    arpo.confusion_matrixes = pickled_data[0][1] #     |

    for exp in pickled_data[1:]:
        arpo = arpo.start_new_exp(name=exp[0])
        arpo.confusion_matrixes = exp[1]

    return arpo
