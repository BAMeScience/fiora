
import torch 
import numpy as np
from fiora.MOL.constants import ORDERED_ELEMENT_LIST_WITH_HYDROGEN

class CovariateFeatureEncoder:
    def __init__(self, feature_list=["collision_energy", "molecular_weight", "precursor_mode", "instrument", "element_composition"], sets_overwrite: dict|None=None):
        if "ce_steps" in feature_list:
            raise ValueError("'ce_steps' is not meant as a setup feature. Remove from feature_list")
        self.encoding_dim = 0
        self.feature_list = feature_list
        self.categorical_sets = {
            "instrument": ["HCD", "Q-TOF", "IT-FT/ion trap with FTMS", "IT/ion trap"], # "IT-FT/ion trap with FTMS", "IT/ion trap", "QqQ", "QqQ/triple quadrupole"
            "precursor_mode": ["[M+H]+", "[M-H]-"]
        }
        if sets_overwrite:
            for new_set, new_categories in sets_overwrite.items():
                self.categorical_sets[new_set] = new_categories
        
        self.continuous_set = {
            "collision_energy",
            "molecular_weight"
        }
        self.normalize_features = {
            "collision_energy": {"min": 0, "max": 100, "transform": "linear"},
            "molecular_weight": {"min": 0, "max": 1000, "transform": "linear"}
        }
        
        self.reduced_categorical_features = ["instrument"] # Reduced features may have additional values that will be encoded with another bit (representing OTHERS)
        self.one_hot_mapper = {}
        for feature in self.feature_list:
            if feature in self.categorical_sets.keys():
                variables = self.categorical_sets[feature]
                self.one_hot_mapper[feature] = dict(zip(variables, range(self.encoding_dim, len(variables) + self.encoding_dim)))
                self.encoding_dim += len(variables)
                if feature in self.reduced_categorical_features:
                    self.encoding_dim += 1
            if feature in self.continuous_set:
                self.one_hot_mapper[feature] = self.encoding_dim
                self.encoding_dim += 1
        
        if "element_composition" in self.feature_list:
            self.one_hot_mapper["element_composition"] = {
                element: idx for idx, element in enumerate(ORDERED_ELEMENT_LIST_WITH_HYDROGEN, start=self.encoding_dim)
            }  # Note that element composition is using int numbers and not one hot mapping. But the index is still correct.
            self.encoding_dim += len(ORDERED_ELEMENT_LIST_WITH_HYDROGEN)
    
    def encode(self, dim0, metadata, G=None):
        feature_matrix = torch.zeros(dim0, self.encoding_dim, dtype=torch.float32)
        for feature in self.feature_list:
            if feature in self.categorical_sets.keys():
                value = metadata[feature]
                if value in self.categorical_sets[feature]:
                    feature_matrix[:, self.one_hot_mapper[feature][value]] = 1.0
                else:
                    feature_matrix[:, self.one_hot_mapper[feature][list(self.categorical_sets[feature])[-1]] + 1] = 1.0
            
            elif feature in self.continuous_set:
                value = metadata[feature]
                if feature in self.normalize_features.keys():
                    value = (value - self.normalize_features[feature]["min"]) / (self.normalize_features[feature]["max"] - self.normalize_features[feature]["min"])
                feature_matrix[:, self.one_hot_mapper[feature]] = value
                feature_matrix = torch.clamp(feature_matrix, 0.0, 1.0)

            elif feature == "element_composition":
                if G is None:
                    raise ValueError("Graph G must be provided to encode 'element_composition'")
                element_composition = self.get_element_composition(G)
                for idx, element in enumerate(ORDERED_ELEMENT_LIST_WITH_HYDROGEN):
                    feature_matrix[:, self.one_hot_mapper["element_composition"][element]] = element_composition[idx]
        return feature_matrix
    
    def normalize_collision_steps(self, ce_steps):
        norm_ce = lambda x: (x - self.normalize_features["collision_energy"]["min"]) / (self.normalize_features["collision_energy"]["max"] - self.normalize_features["collision_energy"]["min"]) 
        ce_steps = [norm_ce(x) for x in ce_steps]
        return ce_steps
    

    def get_element_composition(self, G):
        # Initialize composition vector with zeros
        element_composition = torch.zeros(len(ORDERED_ELEMENT_LIST_WITH_HYDROGEN), dtype=torch.float32)

        # Iterate through nodes in the graph
        for node in G.nodes:
            atom = G.nodes[node]['atom']
            symbol = atom.GetSymbol()  # Get the atomic symbol
            if symbol in ORDERED_ELEMENT_LIST_WITH_HYDROGEN:
                index = ORDERED_ELEMENT_LIST_WITH_HYDROGEN.index(symbol)  # Find the index of the element
                element_composition[index] += 1  # Increment the count for the element

            # Add hydrogens explicitly
            hydrogens = atom.GetTotalNumHs()
            hydrogen_index = ORDERED_ELEMENT_LIST_WITH_HYDROGEN.index('H')  # Ensure 'H' is in ORDERED_ELEMENT_LIST
            element_composition[hydrogen_index] += hydrogens

        return element_composition