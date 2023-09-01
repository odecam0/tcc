import sys
from pathlib import Path
p = Path("../")
sys.path.append(str(p.resolve()))

import ARPC
import ie_data as ie
import arpc_plot as ap
import arpc_metrics as am

test = ARPC.Arpc()
test = ie.load_group_of_experiments("dataaug_classified_arpos")

file_name="lstm-dataaug-comparisson.png"
past_experiment_name="experimento sem aumento de dados"
ap.plot_compare_2_set_of_exps(
    test, 3, am.get_label_accuracy_mean,
    lambda x: x.confusion_matrixes[0][1],
    file_name=file_name,
    gray_experiment_summary_name=past_experiment_name,
    caption=False)
