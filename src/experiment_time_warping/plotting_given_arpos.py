import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

from pdb import set_trace

import pickle

import ARPC
import arpc_plot as ap
import arpc_utils as au

def load_group_of_experiments(file_name):
    arpo = ARPC.Arpc()
    
    with open(file_name, 'rb') as file:
        pickled_data = pickle.load(file)
    #   pickled_data --------------> [(<name0>, [<matrixes0>]), ... , (<nameN>, [<matrixesN>])]

    pickled_data = pickled_data[::-1]
    
    arpo.name = pickled_data[0][0]
    arpo.confusion_matrixes = pickled_data[0][1]
    
    for exp in pickled_data[1:]:
        arpo = arpo.start_new_exp(name=exp[0])
        arpo.confusion_matrixes = exp[1]
    
    return arpo

test = load_group_of_experiments('all_exps_arpos')

# plot com um dos plots em cinza por traz
import arpc_plot as ap
import arpc_metrics as am
# from importlib import reload
# reload(ap)

if len(sys.argv) > 1:
    file_name=sys.argv[1]
else:
    file_name='default_name_2plots.png'

ap.plot_compare_2_set_of_exps(test, 3, am.get_label_accuracy_mean,
                              lambda x: x.confusion_matrixes[0][1],
                              file_name=file_name, gray_experiment_summary_name="experimento sem aumento de dados",
                              caption=False)

 # def my_get_plot_data(x):
#     return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean,
#                                              lambda x: x.confusion_matrixes[0][1])

# ap.plot_compare_err_bar(test, my_get_plot_data, ap.group_labels_by_first_word,
#                         save_file_name=file_name, show=False, metric_used='acurácia') 
