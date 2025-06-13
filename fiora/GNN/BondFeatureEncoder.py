
import torch 
import numpy as np
from typing import Literal


class BondFeatureEncoder:
    def __init__(self, feature_list=["bond_type", "ring_type"]):
        self.encoding_dim = 0
        self.feature_list = feature_list
        self.sets = {
            "bond_type": ["AROMATIC", "SINGLE", "DOUBLE", "TRIPLE"],
            "ring_type": ["no-ring", "small-ring", "5-cycle", "6-cycle", "large-ring"],
            "ring_type_binary": ["is_in_ring"],       
        }
        self.reduced_features = [] # Reduced features may have additional values that will be encoded with another bit (representing OTHERS)
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

    
    def encode(self, G, edges,encoder_type: Literal['one_hot', 'number']):
        
        if encoder_type == 'one_hot':
            feature_matrix = torch.zeros(len(edges), self.encoding_dim, dtype=torch.float32)

            for i, (u,v) in enumerate(edges):
                bond = G[u][v]["bond"]
                if "bond_type" in self.feature_list:
                    feature_matrix[i][self.one_hot_mapper["bond_type"][bond.GetBondType().name]] = 1.0
                if 'ring_type' in self.feature_list:
                    if not bond.IsInRing():
                        ring_type = "no-ring"
                    elif bond.IsInRingSize(7):
                        ring_type = "large-ring"
                    elif bond.IsInRingSize(6):
                        ring_type = "6-cycle"
                    elif bond.IsInRingSize(5):
                        ring_type = "5-cycle"
                    else:
                        ring_type = "small-ring"
                    feature_matrix[i][self.one_hot_mapper['ring_type'][ring_type]] = 1.0
                if 'ring_type_binary' in self.feature_list:
                    if bond.IsInRing():
                        feature_matrix[i][self.one_hot_mapper['ring_type_binary']["is_in_ring"]] = 1.0
                    # else case implicit = 0
        
        elif encoder_type == 'number': # Case: Number mapping 
            feature_matrix = torch.zeros(len(edges), len(self.feature_list), dtype=torch.int)
            for i, (u,v) in enumerate(edges):
                bond = G[u][v]["bond"]


                for j, feature in enumerate(self.feature_list):
                    if feature == "bond_type":
                        value = bond.GetBondType().name
                        if value in self.sets['bond_type']:
                            feature_matrix[i][j] = self.number_mapper[feature][value]
                        else:
                            raise NotImplementedError("Unknown bond type is not accounted for.")
                    elif feature == 'ring_type':
                        if not bond.IsInRing():
                            ring_type = "no-ring"
                        elif bond.IsInRingSize(7):
                            ring_type = "large-ring"
                        elif bond.IsInRingSize(6):
                            ring_type = "6-cycle"
                        elif bond.IsInRingSize(5):
                            ring_type = "5-cycle"
                        else:
                            ring_type = "small-ring"
                        feature_matrix[i][j] = self.number_mapper[feature][ring_type]
                if feature == 'ring_type_binary':
                    raise NotImplementedError("Binary feature not implemented with number embedding. Use default 'ring_type' instead.")          

        return feature_matrix