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

from pprint import pprint

print('[accuracy, experiment_name]')
pprint([(am.get_accuracy_mean_from_all_exps(e), e.name) for e in test.exp_gen()])
