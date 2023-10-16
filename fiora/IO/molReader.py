from rdkit import Chem



'''
Functions to read mol files
'''

def load_MOL(path):
    MOL_string=open(path,'r').read()
    m = Chem.MolFromMolBlock(MOL_string)
    return m