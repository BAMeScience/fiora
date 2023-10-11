import pandas as pd

class LibraryLoader():
        def __init__(self, path=None):
            self.path = path
        
        def load_from_csv(self, path):
            return pd.read_csv(path, index_col=[0], low_memory=False)
        
        def load_from_msp(self):
            #TODO IMPLEMENT
            return
        
        def clean_library(self):
            #TODO IMPLEMENT + parameters for filtration
            return