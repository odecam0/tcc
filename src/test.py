import ARPC
import manips
from arpc_plot import plot_all
from arpc_utils import get_acc_data, get_gyr_data
import arpc_features as af
import numpy as np

test = ARPC.Arpc()

# Funcionando
test.load_data('../../dataset/data',
               'Aluno<participante:\d+>/<atividade:[A-Z][a-z]*><intensidade:[A-Z][a-z]*>.txt')
test.raw_data

test.add_manip(get_acc_data)
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
test.preprocessed_data
test.set_windows()
