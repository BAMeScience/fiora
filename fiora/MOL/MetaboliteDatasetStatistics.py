import pandas as pd
from collections import Counter
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.constants import ORDERED_ELEMENT_LIST

class MetaboliteDatasetStatistics:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the class with a DataFrame containing dataset information.
        :param data: pd.DataFrame with columns like 'Metabolite', 'group_id', etc.
        """
        self.data = data
        self.statistics = {}

    def _compute_element_composition_stats(self):
        """
        Compute binary presence and count of each element from ORDERED_ELEMENT_LIST for each metabolite.
        Also compute the total number of elements for each metabolite.
        :return: pd.DataFrame with binary presence, count of elements, and metadata.
        """
        all_meta = []
        for _, row in self.data.iterrows():
            metabolite = row['Metabolite']
            if isinstance(metabolite, Metabolite):
                # Create nested dictionaries for presence and count
                metadata_dict = {
                    "element_presence": {element: int(element in metabolite.node_elements) for element in ORDERED_ELEMENT_LIST},
                    "element_count": {element: metabolite.node_elements.count(element) for element in ORDERED_ELEMENT_LIST},
                }
                # Add additional metadata
                metadata_dict["ExactMolWeight"] = metabolite.ExactMolWeight
                metadata_dict["Formula"] = metabolite.Formula
                metadata_dict["SMILES"] = metabolite.SMILES
                metadata_dict["InChIKey"] = metabolite.InChIKey
                metadata_dict["TotalElements"] = len(metabolite.node_elements)  # Total number of elements
                all_meta.append(metadata_dict)

        return pd.DataFrame(all_meta)
    
    def _compute_element_summary(self):
        """
        Compute total counts, presence probability for each element, and ANY_RARE probability across the entire dataset.
        :return: dict with total counts, presence probabilities for each element, and ANY_RARE probability.
        """
        individual_stats = self.statistics['Individual_molecular_stats']

        # Initialize counters
        total_counts = Counter()
        presence_counts = Counter()
        any_rare_count = 0  # Counter for molecules with at least one rare element

        # Define rare elements (everything except C, O, N, H)
        rare_elements = [element for element in ORDERED_ELEMENT_LIST if element not in ["C", "O", "N", "H"]]

        # Aggregate counts and presence probabilities
        for _, row in individual_stats.iterrows():
            element_counts = row['element_count']
            element_presence = row['element_presence']
            total_counts.update(element_counts)
            presence_counts.update(element_presence)

            # Check if at least one rare element is present
            if any(element_presence[element] for element in rare_elements):
                any_rare_count += 1

        # Compute presence probabilities
        total_molecules = len(individual_stats)
        presence_probabilities = {element: presence_counts[element] / total_molecules for element in ORDERED_ELEMENT_LIST}

        # Compute ANY_RARE probability and add it as another "element"
        presence_probabilities["ANY_RARE"] = any_rare_count / total_molecules

        return {
            "Total Counts": total_counts,
            "Presence Probabilities": presence_probabilities,
        }


    def generate_molecular_statistics(self, unique_compounds: bool = True):
        """
        Precompute molecular statistics using the Metabolite class and store them in the class.
        """
        # Retrieve detailed information for each metabolite
        if unique_compounds:
            self.data = self.data.drop_duplicates(subset='group_id')
        self.statistics['Individual_molecular_stats'] = self._compute_element_composition_stats()
        self.statistics['Molecular Summary'] = self._compute_element_summary()

        
        

    def _compute_duplicates(self):
        """
        Compute duplicate occurrences based on 'group_id'.
        :return: pd.DataFrame with group_id counts.
        """
        group_counts = self.data['group_id'].value_counts().reset_index()
        group_counts.columns = ['group_id', 'Count']
        return group_counts

    def get_statistics(self):
        """
        Retrieve precomputed statistics.
        :return: dict containing all statistics.
        """
        if not self.statistics:
            raise ValueError("Statistics have not been generated yet. Call generate_molecular_statistics() first.")
        return self.statistics