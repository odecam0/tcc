from load_data import load_data

import pandas as pd
import numpy  as np

from arpc_utils import sort_metadata
import arpc_window

from copy import copy

from arpc_features import merge_features, calc_feature

from pdb import set_trace

from overload import *

class Arpc:
    def __init__(self, name='default_name'):
        # Inform what this experiment is about
        self.name              = name

        self.raw_data          = None
        self.preprocessed_data = None
        self.segmented_data    = None
        self.featured_data     = None

        # For preprocessing
        self.manips = []

        # For classification
        self.trained_models     = None
        
        # This atribute should be structured like this:
        #
        #  [ [ <confusion_matrix0>, [<labels>] ],
        #      ...,
        #    [ <confusion_matrix1>, [<labels>] ] ]
        #
        # Where labels are in the order that they appear
        # on the confusion_matrix, to guarantee that a given
        # line or column is really talking about a given label
        self.confusion_matrixes = None

        self.past_exp           = None


    def load_data(self, root_dir:str, name_scheme:str):
        self.raw_data = sort_metadata(load_data(root_dir, name_scheme))


    # Preprocessing data
    def add_manip(self, func):
        """
        Add a function to be called in order of added functions
        when using do_manip() to set self.preprocessed_data
        Func must have only one parameter to be passed, the dataframe
        """
        self.manips += [func]

    def do_manip(self):
        """
        Apply every function in self.manips to self.raw_data in sequence
        to produce self.preprocessed data
        """
        new_data = self.raw_data

        for f in self.manips:
            new_data = f(new_data)

        self.preprocessed_data = sort_metadata(new_data)


    # Janelamento dos dados
    def set_windows(self, w_type='rolling', size=10):
        # https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string
        #
        # A ideia é que seja possível chamar outros tipos de janelas apenas trocando o 'w_type'
        if hasattr(arpc_window, 'get_'+w_type+'_windows'):
            func = getattr(arpc_window, 'get_'+w_type+'_windows')
            self.segmented_data = func(self.preprocessed_data, size)
        else:
            print("Acho que esse tipo de função aí n existe não hein doido.. arpc_window.py, olha lá..")


    # Data augmentation
    def apply_each_window(self, funcs:list, substitute=False):
        """
        Apply every function in funcs on every window,
        if substitute is True, the results from funcs are stored in segmented_data and the old values are
        thrown away. Otherwise, the results are stored alongside the old values, in keys like this:
        <func_applied_name>_<original_key>
        """

        if not self.segmented_data:
            print("This function must be called after 'set_windows'")
            return 

        if substitute:
            data = {}
        else:
            # https://docs.python.org/3/library/copy.html
            data = copy(self.segmented_data)
        
        for func in funcs:
            for key in self.segmented_data:
                print('applying', func.__name__, 'to', key, 'data')
                data[func.__name__ + '_' + key] = [func(i) for i in self.segmented_data[key]] 
                 
        self.segmented_data = data


    # Feature extraction
    def set_features(self, funcs, columns=['x', 'y', 'z']):

        # Conseguindo uma lista com todas as janelas
        # self.segmented_data é um dicionário com as janelas separadas por classe
        windows = []
        for w in self.segmented_data.values():
            windows += [i for i in w]

        # Misturando todas as features de cada janela
        dfs = []
        p = .0
        for i in range(len(windows)):
            pp = p # previous progress
            p = "{:.0f}%".format(i/len(windows) * 100) # progress
            if p != pp:
                print(p, end='\r')

            feature_windows = []
            for f in funcs:
                feature_windows += [calc_feature(windows[i], f, featured_columns=columns)]

            dfs += [merge_features(feature_windows)]
        
        self.featured_data = pd.concat(dfs).reset_index(drop=True)


    # Classification
    def classify(self, train_proc, evaluate_proc, datasplit_proc,
                 featured_data=True, prepare_featured_data_proc=None):
        """
        Saves 1 trained model, for each datasplit pair returnd from
        DATASPLIT_PROC
        Saves 1 confusion_matrix for each datasplit pair returned from
        DATASPLIT_PROC
        Use TRAIN_PROC to train each model with each train data split
        and return it
        Use EVALUATE_PROC to get confusion matrix for each trained model
        DATASET is used to get data splits using DATASPLIT_PROC
        """

        if featured_data:
            train_splits, eval_splits = datasplit_proc(self.featured_data)
            # datasplit_proc must return 2-tuple of lists
        elif prepare_featured_data_proc:
            train_splits, eval_splits = datasplit_proc(
                *prepare_featured_data_proc(self.segmented_data))
        else:
            # TODO! Transformar em um erro pra valer
            print("Erro na função classify de ARPC.py.\n",
                  "feature_data = False and",
                  "prepare_featured_data_proc is not a function")
            return

        trained_models = []
        for t in train_splits:
            trained_models += [train_proc(*t)]

        confusion_matrixes = []
        for t, e in zip(trained_models, eval_splits):
            confusion_matrixes += [(evaluate_proc(t, *e), t.classes_)]

        self.trained_models     = trained_models
        self.confusion_matrixes = confusion_matrixes


    # For comparing experiments
    def start_new_exp(self, reuse='none', name='default_name'):
        if reuse not in ['none', 'raw', 'preprocessed', 'segmented', 'featured']:
            print("Argument passed as 'reuse' is not valid: ")
            return

        new_obj = Arpc(name=name)
        new_obj.past_exp = self

        if reuse == 'none':
            return new_obj

        new_obj.raw_data = self.raw_data
        if reuse == 'raw':
            return new_obj

        new_obj.preprocessed_data = self.preprocessed_data
        if reuse == 'preprocessed':
            return new_obj

        new_obj.segmented_data = self.segmented_data
        if reuse == 'segmented':
            return new_obj

        new_obj.featured_data = self.featured_data
        if reuse == 'featured':
            return new_obj


    # Retorna um iterable com cada experimento
    # realizado.
    @overload
    def exp_gen(self):
        exp = self
        while exp:
            yield exp
            exp = exp.past_exp
    # Retorna um iterable do experimento atual
    # até a profundidade indicada
    @exp_gen.add
    def exp_gen(self, depth):
        for e in self.exp_gen():
            if depth <= 0:
                break
            depth -= 1
            yield e
    # Retorna um iterable indicando uma distancia
    # do experimento atual, e uma profundidade
    @exp_gen.add
    def exp_gen(self, offset, depth):
        i = 0
        for e in self.exp_gen():
            if i < offset:
                i += 1
                continue
            elif depth >= 0 and depth != 'full':
                depth -= 1
                yield e
        
