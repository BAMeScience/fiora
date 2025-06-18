from rdkit import Chem
from rdkit.Chem import Descriptors

h_minus = Chem.MolFromSmiles("[H-]") #hydrid
h_plus = Chem.MolFromSmiles("[H+]") #h proton
h_2 = Chem.MolFromSmiles("[HH]") #h2 

ADDUCT_WEIGHTS = {
    "[M+H]+": Descriptors.ExactMolWt(h_plus), #1.007276,
    "[M+H]-": Descriptors.ExactMolWt(h_plus), # TODO might not technically exist
    "[M+NH4]+": 18.033823,
    "[M+Na]+": 22.989218 ,
    "[M-H]-": -1*Descriptors.ExactMolWt(h_plus),
    
    #
    # positvie fragment rearrangements
    #
    "[M-H]+": -1*Descriptors.ExactMolWt(h_minus), # Double bond replacing 2 hydrogen atoms + H
    "[M]+": 0,
    "[M-2H]+": -1 * Descriptors.ExactMolWt(h_2), # Loosing proton and hydrid
    "[M-3H]+": -1 * Descriptors.ExactMolWt(h_2) - 1 * Descriptors.ExactMolWt(h_minus), # 2 Double bonds  + H
    # experimental cases
    #"[M-4H]+": -1.007276 * 4,
    #"[M-5H]+": -1.007276 * 5,
    
    
    #
    # negative fragment rearrangements
    # 
    
    # "[M-H]-": -1*Chem.Descriptors.ExactMolWt(h_plus), # see above
    "[M]-": 0, # could be one electron too many 
    "[M-2H]-": -1 * Descriptors.ExactMolWt(h_2),
    "[M-3H]-": -1 * Descriptors.ExactMolWt(h_2) - 1 * Chem.Descriptors.ExactMolWt(h_plus),    
    }



PPM = 1/1000000
DEFAULT_PPM = 100 * PPM
DEFAULT_DALTON = 0.05 # equiv: 100ppm of 500 m/z 
MIN_ABS_TOLERANCE = 0.01 # 0.02 # Tolerance aplied for small fragment when relative PPM gets too small
#DEFAULT_MODES = ["[M+H]+", "[M-H]+", "[M-3H]+"]
DEFAULT_MODES = ["[M+H]+", "[M]+", "[M-H]+", "[M-2H]+", "[M-3H]+"] #"[M-4H]+"] #, "[M-5H]+"]
DEFAULT_MODE_MAP = {mode: i for i, mode in enumerate(DEFAULT_MODES)}
#NEGATIVE_MODES = ["[M]-", "[M-H]-", "[M-2H]-", "[M-3H]-", "[M-4H]-"]

# source: https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator


ORDERED_ELEMENT_LIST = ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S'] # Warning: Changes may affect model and version control 
ORDERED_ELEMENT_LIST_WITH_HYDROGEN = ORDERED_ELEMENT_LIST + ['H']  # Hydrogen is added at the end for element composition encoding