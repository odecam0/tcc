from load_data import load_data

import pandas as pd

from arpc_utils import sort_metadata
import arpc_window

from copy import copy

from arpc_features import merge_features, calc_feature

from pdb import set_trace

class Arpc:
    def __init__(self):
        self.raw_data          = None
        self.preprocessed_data = None
        self.segmented_data    = None
        self.featured_data     = None

        # For preprocessing
        self.manips = []

        # For classification
        self.trained_models     = None
        self.confusion_matrixes = None


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
                print(p)

            feature_windows = []
            for f in funcs:
                feature_windows += [calc_feature(windows[i], f, featured_columns=columns)]

            dfs += [merge_features(feature_windows)]
        
        self.featured_data = pd.concat(dfs).reset_index(drop=True)
    
    def classify(self, train_proc, evaluate_proc, datasplit_proc):
        """
        Saves 1 trained model, for each datasplit pair returnd from    DATASPLIT_PROC
        Saves 1 confusion_matrix for each datasplit pair returned from DATASPLIT_PROC
        Use TRAIN_PROC to train each model with each train data split
        Use EVALUATE_PROC to get confusion matrix for each trained model
        MODEL is used to be trained by TRAIN_PROC
        DATASET is used to get data splits using DATASPLIT_PROC
        """

        train_splits, eval_splits = datasplit_proc(self.featured_data)
        # datasplit_proc must return 2-tuple of lists

        trained_models = []
        for t in train_splits:
            trained_models += [train_proc(*t)]

        confusion_matrixes = []
        for t, e in zip(trained_models, eval_splits):
            confusion_matrixes += [evaluate_proc(t, *e)]

        self.trained_models     = trained_models
        self.confusion_matrixes = confusion_matrixes
