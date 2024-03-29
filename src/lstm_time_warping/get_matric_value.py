import sys
from pathlib import Path
p = Path("../")
sys.path.append(str(p.resolve()))

from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import ARPC
import ie_data as ie
import arpc_plot as ap
import arpc_metrics as am

test = ARPC.Arpc()
test = ie.load_group_of_experiments("dataaug_classified_arpos")

# from pprint import pprint

# print('[accuracy, experiment_name]')
# pprint([(am.get_accuracy_mean_from_all_exps(e), e.name) for e in test.exp_gen()])

#####################################################

def get_all_experiments_names(arpo):
    return [e.name for e in arpo.exp_gen()]

gaen = get_all_experiments_names
gaen(test)

def get_experiment_by_name(arpo, name):
    exp = ""
    for e in arpo.exp_gen():
        if e.name == name:
            exp = e
    return exp

gebn = get_experiment_by_name
split_loso_normal = gebn(test, 'Split LOSO com LSTM')
sln = split_loso_normal
sln.confusion_matrixes

def save_sum_of_confusion_matrixes(arpo, filename, normalize=False):
    cm = reduce(lambda x, y: x + y, [i for i,_ in arpo.confusion_matrixes])
    if normalize:
        cm = cm / np.sum(cm)
        
    ConfusionMatrixDisplay(cm, display_labels=arpo.confusion_matrixes[0][1]).plot()
    plt.savefig(filename)
    
    return np.sum(cm, axis=0)

ssocm = save_sum_of_confusion_matrixes
ssocm(sln, "sln_cms.png")
print("sln: " + str(ssocm(sln, "sln_cms.png", normalize=True)))
# (find-tccsfile "/lstm_time_warping/sln_cms.png")

# Conferindo valores das matrizes de confusão
split_loso_dataug = test.past_exp
sld = split_loso_dataug

print("sld: " + str(ssocm(sld, "sld_cms.png", normalize=True)))

summaries = ssocm(sld, "sld_cms.png", normalize=True)

def get_accuracies(cm):

labels = sld.confusion_matrixes[0][1]
data = {}
[data.update({l:s}) for l,s in zip(labels, summaries)]

from pprint import pprint
pprint(data)

from arpc_metrics import label_accuracy

data = [[label_accuracy(sld.confusion_matrixes[j], i) for i in range(len(labels))] for j in range(11)]
np.sum(data,axis=0)
