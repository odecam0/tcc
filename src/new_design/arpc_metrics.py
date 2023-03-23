# To extract metrics from a confusion matrix
import numpy as np

from overload import *

def call_label_func(cm, label:str, func):
    for i in range(len(cm[1])):
        if cm[1][i] == label:
            index = i
            break
    else:
        print('Não tem esse label aí não')
        return

    return func(cm, index)

@overload
def label_accuracy(cm, label:int):
    cm = cm[0]
    correct_predictions = cm.diagonal()[label]
    total_predictions   = cm[label, :].sum()

    return correct_predictions/total_predictions

@label_accuracy.add
def label_accuracy(cm, label:str):
    return call_label_func(cm, label, label_accuracy)

def accuracy(cm):
    cm = cm[0]
    correct_predictions = np.diagonal(cm).sum()
    total_predictions   = cm.sum()
    return correct_predictions/total_predictions

def get_label_accuracy_mean(cms, label):
    # faltou calcular o intervalo de confiança :) A parte mais cabreira
    t_val = 1.796
    e = np.array([label_accuracy(c, label) for c in cms])
    mean = np.mean(e)
    stderr = np.std(e) / np.sqrt(len(e))

    simetric = stderr * t_val

    inf = simetric
    sup = simetric

    # deve ser no mínimo 0
    if mean - inf <= 0:
        inf = mean

    # deve ser no máximo 1
    if mean + sup >= 1:
        sup = 1 - mean

    y_err = [inf, sup]

    # Agora oque que essa porra vai retornar?
    return mean, inf, sup


@overload
def label_precision(cm, label:int):
    cm = cm[0]
    true_predictions = cm.diagonal()[label]
    false_predictions = cm[:, label].sum()

    return true_predictions / (true_predictions + false_predictions)

@label_precision.add
def label_precision(cm, label:str):
    return call_label_func(cm, label, label_precision)

def precision(cm):
    labels = cm[1]
    cmt = cm[0]

    return sum([label_precision(cm, l) for l in labels])/len(labels)
