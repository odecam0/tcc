from load_data import load_data

import pandas as pd

from arpc_utils import sort_metadata
import arpc_window

from copy import copy

from pdb import set_trace

class Arpc:
    def __init__(self):
        self.raw_data          = None
        self.preprocessed_data = None
        self.segmented_data    = None
        self.featured_data     = None

        # For preprocessing
        self.manips = []

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

