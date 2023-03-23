import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

import ARPC
import arpc_plot as ap
import arpc_utils as au
test = ARPC.Arpc(name='Regular Split no augmentation')
test.load_data('../../dataset/data',
               'Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.txt')
test.raw_data

plot_all(test.raw_data, participantes=['1'])
ap.plot_all(au.get_acc_data(test.raw_data), participantes=['2'])

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

ap.plot_all(test.preprocessed_data, participantes=['2'])

# Segment  data
test.set_windows(size=30) # 3 segundos de dados

# Extracting features
import numpy as np
test.set_features([np.mean, np.std], columns=['x', 'y', 'z', 'module'])

# Actual classification
import arpc_classification as ac
participantes = [str(i) for i in range(1, 12)]
test.classify(ac.train_randomforest, ac.eval_randomforest,
              lambda x: ac.multi_split(x, ac.data_split1, participantes))
test = test.start_new_exp(reuse='featured', name='LOSO split no augmentation')
test.classify(ac.train_randomforest, ac.eval_randomforest,
              lambda x: ac.multi_split(x, ac.loso_split, participantes))
test = test.start_new_exp(reuse='featured', name='Semi LOSO split no augmentation')
test.classify(ac.train_randomforest, ac.eval_randomforest,
              lambda x: ac.multi_split(x, ac.semi_loso_split, participantes))

# Now augmenting segmented data
test = test.start_new_exp(reuse='segmented', name='Split comum com aumento de dados')
from TimeWarpWindow import warp_window
def timewarped(df):
    result, _ = warp_window(df, 5) # A margem para definir um centro pro algoritmo
    return result

test.apply_each_window(funcs=[timewarped])

# Extracting features from augmented data 
test.set_features([np.mean, np.std], columns=['x', 'y', 'z', 'module'])

# Classifiying augmented features
test.classify(ac.train_randomforest, ac.eval_randomforest,
              lambda x: ac.multi_split(x, ac.data_split1, participantes))
test = test.start_new_exp(reuse='featured', name='Split LOSO com aumento de dados')
test.classify(ac.train_randomforest, ac.eval_randomforest,
              lambda x: ac.multi_split(x, ac.loso_split, participantes))
test = test.start_new_exp(reuse='featured', name='Split semi LOSO com aumento de dados')
test.classify(ac.train_randomforest, ac.eval_randomforest,
              lambda x: ac.multi_split(x, ac.semi_loso_split, participantes))

name_list = [ 'Split semi LOSO com aumento de dados',
	      'Split LOSO com aumento de dados',
	      'Split comum com aumento de dados' ]
for e,l in zip(test.exp_gen(3), name_list):
    e.name = l

# plot com um dos plots em cinza por traz
import arpc_plot as ap
import arpc_metrics as am
from importlib import reload
reload(ap)
ap.plot_compare_2_set_of_exps(test, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1]) 

def my_get_plot_data(x):
    return ap.get_compare_side_err_barr_data(x, 3, am.get_label_accuracy_mean, lambda x: x.confusion_matrixes[0][1])

ap.plot_compare_side_err_bar(test, my_get_plot_data) 
