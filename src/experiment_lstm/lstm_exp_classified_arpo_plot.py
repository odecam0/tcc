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
    
    return arpo

test = ARPC.Arpc()
test = load_group_of_experiments(test, 'group_of_lstm_experiments')
test.confusion_matrixes

import arpc_metrics as am

def my_get_plot_data(x):
    return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1])

ap.plot_compare_err_bar(test, my_get_plot_data, ap.group_labels_by_first_word, save_file_name="lstm_plot_compare_err_bar1-gray-eb_width-no_legend.png", show=False, gray=True, eb_width=2, legend=False) 

"""
(defun doit ()
    (interactive)
    (async-shell-command "cd ~/ic/src/experiment_lstm/ && python lstm_exp_classified_arpo_plot.py"))
"""
