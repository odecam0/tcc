import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

from ie_data import load_group_of_experiments

import ARPC
import arpc_plot as ap
import arpc_metrics as am

test = load_group_of_experiments('arpos_lstm_segmented_data')

def my_get_plot_data(x):
    return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1])

ap.plot_compare_err_bar(test, my_get_plot_data, ap.group_labels_by_first_word)
