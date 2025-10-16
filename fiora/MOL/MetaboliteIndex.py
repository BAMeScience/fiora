from typing import List, Literal
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.FragmentationTree import FragmentationTree

class MetaboliteIndex:

    def __init__(self) -> None:
        self.metabolite_index = {}

    def index_metabolites(self, list_of_metabolites: List) -> None:
        for metabolite in list_of_metabolites:

            id = self.find_metabolite_id(metabolite)
            if id is not None:
                metabolite.set_id(id)
            else:
                new_id = len(self.metabolite_index)
                self.metabolite_index[new_id] = {"Metabolite": metabolite}
                metabolite.set_id(new_id)

    def create_fragmentation_trees(self, depth: int=1) -> None:
        for id, entry in self.metabolite_index.items():
            metabolite = entry["Metabolite"]
            entry["FragmentationTree"] =  FragmentationTree(metabolite.MOL)
            entry["FragmentationTree"].build_fragmentation_tree(metabolite.MOL, metabolite.edges_as_tuples, depth=depth)

    def add_fragmentation_trees_to_metabolite_list(self, list_of_metabolites: List[Metabolite], graph_mismatch_policy: Literal["ignore", "recompute"]="recompute") -> None:
        list_of_mismatched_ids = []
        
        for metabolite in list_of_metabolites:
            id = metabolite.get_id()
            if id is not None:
                # Check if metabolite edges align with the index
                if metabolite.edges_as_tuples == self.metabolite_index[id]["Metabolite"].edges_as_tuples:
                    metabolite.add_fragmentation_tree(self.metabolite_index[id]["FragmentationTree"])
                else:
                    if graph_mismatch_policy == "recompute":
                        metabolite.fragment_MOL()
                    elif graph_mismatch_policy == "ignore":
                        metabolite.add_fragmentation_tree(self.metabolite_index[id]["FragmentationTree"])    
                    else:
                        raise ValueError("Invalid graph_mismatch_policy. Use 'ignore' or 'recompute'.")
                    list_of_mismatched_ids.append((metabolite, id))

        return list_of_mismatched_ids
    
    def find_metabolite_id(self, metabolite: Metabolite) -> int:
        for id, entry in self.metabolite_index.items():
            if metabolite == entry["Metabolite"]:
                return id
        return None

    def get_metabolite(self, id: int) -> Metabolite:
        return self.metabolite_index[id]
    
    def get_fragmentation_tree(self, id: int) -> FragmentationTree:
        return self.metabolite_index[id]["FragmentationTree"]
    
    def get_number_of_metabolites(self) -> int:
        return len(self.metabolite_index)