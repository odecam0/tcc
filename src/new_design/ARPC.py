from load_data import load_data

class Arpc:
    def __init__(self):
        self.raw_data          = None
        self.preprocessed_data = None
        self.segmented_data    = None
        self.featured_data     = None

    def load_data(self, root_dir:str, name_scheme:str):
        self.raw_data = load_data(root_dir, name_scheme)
