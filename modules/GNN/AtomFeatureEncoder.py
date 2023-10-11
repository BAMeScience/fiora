import torch 
import numpy as np
from rdkit import Chem
from typing import Literal




class AtomFeatureEncoder:
    def __init__(self, feature_list=["symbol", "num_hydrogen", "ring_type"]):
        self.encoding_dim = 0
        self.sets = {
            "symbol": ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S'], #OTHERS: Au, Se, Si  #standard list {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
            "num_hydrogen": [0, 1, 2, 3], #OTHERS: 5, 6, 7, 8},
            "ring_type": ["no-ring", "small-ring", "5-cycle", "6-cycle", "large-ring"],
            "hybridization": ["SP", "SP2", "SP3", "SP3D2"],
            "valence_electrons": [1,2,3,4,5,6,7,8],
            "oxidation_number": [1,2,3,4,5,6,7,8,9],
        }
        self.feature_list = feature_list
        self.reduced_features = ["symbol", "num_hydrogen", "hybridization"] # Reduced features may have additional values that will be encoded with another bit (representing OTHERS)
        
        self.one_hot_mapper = {}
        self.number_mapper = {}
        self.feature_numbers = {}
        for feature in self.feature_list:
            variables = self.sets[feature]
            num_variables = len(variables)
            self.one_hot_mapper[feature] = dict(zip(variables, range(self.encoding_dim, num_variables + self.encoding_dim)))
            self.number_mapper[feature] = dict(zip(variables, range(0, num_variables)))
            self.encoding_dim += num_variables
            if feature in self.reduced_features:
                self.encoding_dim += 1
                num_variables += 1
            self.feature_numbers[feature] = num_variables
    
    def encode(self, G, encoder_type: Literal['one_hot', 'number']):
        
        if encoder_type == 'one_hot':
            feature_matrix = torch.zeros(G.number_of_nodes(), self.encoding_dim, dtype=torch.float32)

            for i in range(G.number_of_nodes()):
                atom = G.nodes()[i]['atom']
                
                if 'symbol' in self.feature_list:
                    if not atom.GetSymbol() in self.sets['symbol']:
                        feature_matrix[i][self.one_hot_mapper['symbol'][list(self.sets['symbol'])[-1]] + 1] = 1.0
                    else:
                        feature_matrix[i][self.one_hot_mapper['symbol'][atom.GetSymbol()]] = 1.0

                if 'num_hydrogen' in self.feature_list:
                    value = atom.GetTotalNumHs()
                    if value in self.sets["num_hydrogen"]:
                        feature_matrix[i][self.one_hot_mapper['num_hydrogen'][atom.GetTotalNumHs()]] = 1.0
                    else:
                        feature_matrix[i][self.one_hot_mapper['num_hydrogen'][list(self.sets['num_hydrogen'])[-1]] + 1] = 1.0
                if 'ring_type' in self.feature_list:
                    if not atom.IsInRing():
                        ring_type = "no-ring"
                    elif atom.IsInRingSize(7):
                        ring_type = "large-ring"
                    elif atom.IsInRingSize(6):
                        ring_type = "6-cycle"
                    elif atom.IsInRingSize(5):
                        ring_type = "5-cycle"
                    else:
                        ring_type = "small-ring"
                    feature_matrix[i][self.one_hot_mapper['ring_type'][ring_type]] = 1.0
                if 'hybridization' in self.feature_list:
                    orbi = atom.GetHybridization().name
                    if orbi in self.sets['hybridization']:
                        feature_matrix[i][self.one_hot_mapper['hybridization'][orbi]] = 1.0
                    else:
                        feature_matrix[i][self.one_hot_mapper['hybridization'][list(self.sets['hybridization'])[-1]] + 1] = 1.0
        
        else: # Case: Number mapping 
            feature_matrix = torch.zeros(G.number_of_nodes(), len(self.feature_list), dtype=torch.int)
            for i in range(G.number_of_nodes()):
                atom = G.nodes()[i]['atom']

                for j, feature in enumerate(self.feature_list):
                    if feature == "symbol":
                        if atom.GetSymbol() in self.sets['symbol']:
                            feature_matrix[i][j] = self.number_mapper[feature][atom.GetSymbol()]
                        else:
                            feature_matrix[i][j] = self.feature_numbers[feature] - 1
                    elif feature == 'num_hydrogen':
                        value = atom.GetTotalNumHs()
                        if value in self.sets["num_hydrogen"]:
                            feature_matrix[i][j] = self.number_mapper[feature][value]
                        else:
                            feature_matrix[i][j] = self.feature_numbers[feature] - 1
                    elif feature == 'valence_electrons':
                        value = atom.GetExplicitValence()
                        if value in self.sets["valence_electrons"]:
                            feature_matrix[i][j] = self.number_mapper[feature][value]
                        else:
                            feature_matrix[i][j] = self.feature_numbers[feature] - 1
                    elif feature == 'oxidation_number':
                        raise NotImplementedError()
                        value = Chem.rdMolDescriptors.CalcOxidationNumbers(atom)
                        if value in self.sets["oxidation_number"]:
                            feature_matrix[i][j] = self.number_mapper[feature][value]
                        else:
                            feature_matrix[i][j] = self.feature_numbers[feature] - 1

                    elif feature == 'ring_type':
                        if not atom.IsInRing():
                            ring_type = "no-ring"
                        elif atom.IsInRingSize(7):
                            ring_type = "large-ring"
                        elif atom.IsInRingSize(6):
                            ring_type = "6-cycle"
                        elif atom.IsInRingSize(5):
                            ring_type = "5-cycle"
                        else:
                            ring_type = "small-ring"
                        feature_matrix[i][j] = self.number_mapper[feature][ring_type]
                if feature == 'hybridization':
                    orbi = atom.GetHybridization().name
                    if orbi in self.sets['hybridization']:
                        feature_matrix[i][j] = self.number_mapper[feature][orbi]
                    else:
                        feature_matrix[i][j] = self.feature_numbers[feature] - 1

        return feature_matrix
