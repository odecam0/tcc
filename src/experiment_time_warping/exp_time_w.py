import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

import pickle

import ARPC
import arpc_plot as ap
import arpc_utils as au

from pdb import set_trace

def save_group_of_experiments(arpo, file_name):
    # Irei armazenar as matrizes de confuzão e cada objeto
    # e o nome de cada experimento em uma lista com (<name>, [<matrixes>])
    # para cada experimento
    data_to_pickle = []
    for exp in arpo.exp_gen():
        data_to_pickle+=[(exp.name, exp.confusion_matrixes)]
    
    with open(file_name, 'wb') as file:
        pickle.dump(data_to_pickle, file)

test = ARPC.Arpc(name='Regular Split no augmentation')
test.load_data('../../dataset/data',
               'Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.txt')

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

save_group_of_experiments(test, 'all_exps_arpos')
