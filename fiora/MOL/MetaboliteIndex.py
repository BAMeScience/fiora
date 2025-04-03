from typing import List
from fiora.MOL.Metabolite import Metabolite

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


    def find_metabolite_id(self, metabolite: Metabolite) -> int:
        for id, entry in self.metabolite_index.items():
            if metabolite == entry["Metabolite"]:
                return id
        return None


    def get_metabolite(self, id: int) -> Metabolite:
        return self.metabolite_index[id]
    
    def get_number_of_metabolites(self) -> int:
        return len(self.metabolite_index)