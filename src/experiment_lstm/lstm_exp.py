import sys
from pathlib import Path
p = Path("..")
sys.path.append(str(p.resolve()))

import ARPC
import arpc_plot as ap
import arpc_utils as au
import arpc_metrics as am
import arpc_classification as ac
import manips

import numpy as np

from ie_data import save_group_of_experiments

from pdb import set_trace
from pdb import pm

def exception_hook(t, v, tb):
    local_vars = {}
    while tb:
        filename = tb.tb_frame.f_code.co_filename
        name = tb.tb_frame.f_code.co_name
        line_no = tb.tb_lineno
        print((f"File {filename} line{line_no}, in {name}"))

        local_vars = tb.tb_frame.f_locals
        tb = tb.tb_next

    # print(f"Local variables in top frame: {local_vars}")
    pm()
sys.excepthook = exception_hook

test = ARPC.Arpc(name='Regular Split with LSTM')
test.load_data('../../dataset/data',
               'Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.txt')
test.raw_data

# Preprocessing data
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
# test.set_features([np.mean, np.std], columns=['x', 'y', 'z', 'module'])

participantes = [str(i) for i in range(1, 12)]
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x, y: ac.multi_split_window(x, y, ac.data_split1, participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))
test = test.start_new_exp(reuse='featured', name='Split LOSO com LSTM')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x, y: ac.multi_split_window(x, y, ac.loso_split, participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))
test = test.start_new_exp(reuse='featured', name='Split semi LOSO com LSTM')
test.classify(ac.train_lstm, ac.eval_lstm,
              lambda x, y: ac.multi_split_window(x, y,
                                                 lambda x,y,i: ac.semi_loso_split(x, y, i, .2), #driblando um bug
                                                 participantes),
              featured_data=False,
              prepare_featured_data_proc=lambda x: ac.prepare_window(x, columns=['x', 'y', 'z', 'module']))

save_group_of_experiments(test, 'arpos_lstm_segmented_data')

