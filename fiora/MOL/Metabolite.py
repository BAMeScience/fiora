import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from torch_geometric.data import Data
from typing import Literal

from fiora.MOL.constants import DEFAULT_PPM, DEFAULT_MODES, ADDUCT_WEIGHTS
from fiora.MOL.mol_graph import mol_to_graph, get_adjacency_matrix, get_degree_matrix, get_edges, get_identity_matrix, draw_graph, compute_edge_related_helper_matrices, get_helper_matrices_from_edges
from fiora.MOL.FragmentationTree import FragmentationTree 
from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.SetupFeatureEncoder import SetupFeatureEncoder


class Metabolite:
    def __init__(self, SMILES: str|None, InChI: str|None=None, id: int|None=None) -> None:
        if SMILES:
            self.SMILES = SMILES
            self.MOL = Chem.MolFromSmiles(self.SMILES)
            if not self.MOL:
                raise AssertionError("Molecule invalid; could not be generated from SMILES") 
            self.InChI = Chem.MolToInchi(self.MOL)
            self.InChIKey = Chem.InchiToInchiKey(self.InChI)
        elif InChI:
            self.InChI = InChI
            self.MOL = Chem.MolFromInchi(self.InChI)
            if not self.MOL:
                raise AssertionError("Molecule invalid; could not be generated from InChI")
            self.InChIKey = Chem.InchiToInchiKey(self.InChI)
            self.SMILES = Chem.MolToSmiles(self.MOL) 
        else:
            raise ValueError("Neither SMILES nor InChI were specified.")
        
        self.ExactMolWeight = Descriptors.ExactMolWt(self.MOL)
        self.Formula = rdMolDescriptors.CalcMolFormula(self.MOL)
        self.morganFinger = AllChem.GetMorganFingerprintAsBitVect(self.MOL, 2, nBits=2048) #1024
        self.morganFinger3 = AllChem.GetMorganFingerprintAsBitVect(self.MOL, 3, nBits=2048) #1024
        self.morganFingerCountOnes = self.morganFinger.GetNumOnBits()
        self.id = id
        self.loss_weight = 1.0

    def __repr__(self):
        return f"<Metabolite: {self.SMILES}>"
    
    def __str__(self):
        return f"<Metabolite: {self.SMILES}>"

    def __eq__(self, __o: object) -> bool:
        if self.ExactMolWeight != __o.ExactMolWeight:
            return False
        # Compare the number of bits=1 to prefilter mismatching Metabolites, since it is a much faster comparison
        if self.morganFingerCountOnes != __o.morganFingerCountOnes: 
            return False
        return self.get_morganFinger() == __o.get_morganFinger()

    def __lt__(self, __o: object) -> bool: # TODO not tested!s
        warnings.warn("Warning: < operation for Metabolite class is not tested. Potentially flawed.")
        if self.ExactMolWeight < __o.ExactMolWeight:
            return True
        for bit_this, bit_other in zip(self.get_morganFinger(), __o.get_morganFinger()):
            if bit_this < bit_other:
                return True
            elif bit_other < bit_this:
                return False
        return False
    
    def get_id(self):
        return self.id
    
    def set_id(self, id):
        self.id = id

    def set_loss_weight(self, weight):
        self.loss_weight = weight

    def get_theoretical_precursor_mz(self, ion_type: str=None):
        if ion_type is None:
            if hasattr(self, 'metadata') and 'precursor_mode' in self.metadata:
                ion_type = self.metadata['precursor_mode']
            else:
                raise ValueError("Ion type is not specified and no precursor_mode found in metadata.")
        return self.ExactMolWeight + ADDUCT_WEIGHTS[ion_type]

    def get_morganFinger(self):
        return self.morganFinger
                
    def tanimoto_similarity(self, __o: object, finger: Literal["morgan2", "morgan3"]="morgan2"):
        if finger == "morgan2":
            return DataStructs.TanimotoSimilarity(self.get_morganFinger(), __o.get_morganFinger())
        if finger == "morgan3":
            return DataStructs.TanimotoSimilarity(self.morganFinger3, __o.morganFinger3)
        raise ValueError(f"Unknown type of fingerprint: {finger}. Cannot compare Metabolites.")
    
    # draw
    def draw(self, ax=plt, show: bool=False):
        img = Draw.MolToImage(self.MOL, ax=ax)
    
        ax.grid(False)
        ax.tick_params(axis='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        ax.imshow(img)
        ax.axis("off")
        if show:
            plt.show()
        return img


    # class-specific functions
    def create_molecular_structure_graph(self):
        self.Graph = mol_to_graph(self.MOL)
    
    
    def compute_graph_attributes(self, node_encoder: AtomFeatureEncoder|None = None, bond_encoder: BondFeatureEncoder|None = None, free_memory: bool = True) -> None:

        # Adjacency
        A =  get_adjacency_matrix(self.Graph)
        self.edges_as_tuples = get_edges(A)

        # Graph attributes (obsolete)
        # self.Atilde =  self.A + torch.eye(self.A.shape[0])
        # self.Id = get_identity_matrix(self.A)
        # self.deg = get_degree_matrix(self.A)
        # self.Anorm = self.A / self.deg
        # self.edges = self.A.nonzero()
        # self.AL, self.AR, self.edges  = compute_edge_related_helper_matrices(self.A, self.deg)
        # self.AL, self.AR = get_helper_matrices_from_edges(self.edges_as_tuples, self.A)
        
        # Labels
        self.is_node_aromatic = torch.tensor([[self.Graph.nodes[atom]['is_aromatic'] for atom in self.Graph.nodes()]], dtype=torch.float32).t()
        self.is_edge_aromatic = torch.tensor([[self.Graph[u][v]['bond_type'].name == "AROMATIC" for u,v in self.edges_as_tuples]], dtype=torch.float32).t()
        self.is_edge_in_ring = torch.tensor([[self.Graph[u][v]['bond'].IsInRing() for u,v in self.edges_as_tuples]], dtype=torch.float32).t()
        self.is_edge_not_in_ring = torch.tensor([[not self.Graph[u][v]['bond'].IsInRing() for u,v in self.edges_as_tuples]], dtype=torch.float32).t()
        self.ring_proportion = sum(self.is_edge_in_ring) / len(self.is_edge_in_ring)
        self.edge_forward_direction = torch.tensor([[bool(u < v) for u,v in self.edges_as_tuples]], dtype=torch.bool).t()
        self.edge_backward_direction = torch.tensor([[bool(u > v) for u,v in self.edges_as_tuples]], dtype=torch.bool).t()
        
        # Lists
        self.atoms_in_order = [self.Graph.nodes[atom]['atom'] for atom in self.Graph.nodes()]
        self.node_elements = [self.Graph.nodes[atom]['atom'].GetSymbol() for atom in self.Graph.nodes()]
        self.edge_bond_names = [self.Graph[u][v]['bond_type'].name for u,v in self.edges_as_tuples]
        if bond_encoder:
            self.edge_bond_types = torch.tensor([bond_encoder.number_mapper["bond_type"][bond_name] for bond_name in self.edge_bond_names], dtype=torch.long)
        #self.edge_bond_types = torch.tensor([[bond_encoder.number_mapper["bond_type"][bond_name] for bond_name in self.edge_bond_names]], dtype=torch.long).t()
        
        # Features
        if node_encoder:
            self.node_features = node_encoder.encode(self.Graph, encoder_type="number")
            self.node_features_one_hot = node_encoder.encode(self.Graph, encoder_type="one_hot")
        if bond_encoder:
            self.bond_features = bond_encoder.encode(self.Graph, self.edges_as_tuples, encoder_type="number")
            self.bond_features_one_hot = bond_encoder.encode(self.Graph, self.edges_as_tuples, encoder_type="one_hot")
        else:
            self.bond_features = torch.zeros(len(self.edges_as_tuples), 0, dtype=torch.float32)
            
        self.mode_mapper = {mode: i for i, mode in enumerate(DEFAULT_MODES)} # Set a default mode mapper, might be overwritten later


    def add_metadata(self, metadata, setup_encoder: SetupFeatureEncoder=None, rt_feature_encoder: SetupFeatureEncoder=None, process_metadata: bool = True, max_RT=30.0):
        self.metadata = metadata
        mol_metadata = {"molecular_weight": self.ExactMolWeight}
        metadata.update(mol_metadata)
        if not process_metadata:
            return
        
        if setup_encoder:
            self.setup_features = setup_encoder.encode(1, metadata)
            self.setup_features_per_edge = setup_encoder.encode(len(self.edges_as_tuples), metadata)
            if "ce_steps" in metadata:
                self.ce_steps = torch.tensor([setup_encoder.normalize_collision_steps(metadata["ce_steps"]) + [np.nan for _ in range(7 - len(metadata["ce_steps"]))]]) # nan padding
            else:
                self.ce_steps = torch.tensor([np.nan] * 7, dtype=torch.float).unsqueeze(0)
            self.ce_idx = torch.tensor(setup_encoder.one_hot_mapper["collision_energy"], dtype=int).unsqueeze(dim=-1)
        else:
            self.setup_features = torch.zeros(1, 0, dtype=torch.float32)
            self.setup_features_per_edge = torch.zeros(len(self.edges_as_tuples), 0, dtype=torch.float32)
        
        if rt_feature_encoder:
            self.rt_setup_features = rt_feature_encoder.encode(1, metadata)
        
        if "retention_time" in metadata.keys():
            if not metadata["retention_time"] or np.isnan(metadata["retention_time"]) or "GC" in str(metadata["instrument"]) or metadata["retention_time"] > max_RT:
                metadata["retention_time"] = np.nan
                self.rt = torch.tensor([np.nan]).unsqueeze(dim=-1)
                self.rt_mask = torch.tensor([0], dtype=torch.bool).unsqueeze(dim=-1)
            else:
                self.rt = torch.tensor([metadata["retention_time"]]).unsqueeze(dim=-1)
                self.rt_mask = torch.tensor([1], dtype=torch.bool).unsqueeze(dim=-1)
        else:
            self.rt = torch.tensor([torch.nan]).unsqueeze(dim=-1)
            self.rt_mask = torch.tensor([0], dtype=torch.bool).unsqueeze(dim=-1)
        
        if "ccs" in metadata.keys():
            if not metadata["ccs"] or np.isnan(metadata["ccs"]) or "GC" in str(metadata["instrument"]):
                metadata["ccs"] = np.nan
                self.ccs_mask = torch.tensor([0], dtype=torch.bool).unsqueeze(dim=-1)
                self.ccs = torch.tensor([np.nan]).unsqueeze(dim=-1)
            else:
                self.ccs = torch.tensor([metadata["ccs"]]).unsqueeze(dim=-1)
                self.ccs_mask = torch.tensor([1], dtype=torch.bool).unsqueeze(dim=-1)
        else:
            self.ccs = torch.tensor([torch.nan]).unsqueeze(dim=-1)
            self.ccs_mask = torch.tensor([0], dtype=torch.bool).unsqueeze(dim=-1)

    def is_single_connected_structure(self):
        fragments = Chem.GetMolFrags(self.MOL, asMols=True)
        if len(fragments) == 1:
            return True
        return False
        
    def fragment_MOL(self, depth=1):
        self.fragmentation_tree = FragmentationTree(self.MOL)
        self.fragmentation_tree.build_fragmentation_tree(self.MOL, self.edges_as_tuples, depth=depth)
    
    def match_fragments_to_peaks(self, mz_fragments, int_list=None, mode_mapper=None, tolerance=DEFAULT_PPM, match_stats_only: bool = False):
        self.peak_matches = self.fragmentation_tree.match_peak_list(mz_fragments, int_list, tolerance=tolerance)
        self.edge_breaks = [frag.edges for mz in self.peak_matches.keys() for frag in self.peak_matches[mz]['fragments']]
        self.edge_breaks = [e for edges in self.edge_breaks for e in edges] # Flatten the edge breaks
        edge_break_labels = torch.tensor([[1.0 if (u, v) in self.edge_breaks or (v, u) in self.edge_breaks else 0.0 for u,v in self.edges_as_tuples]], dtype=torch.float32).t()
        
        if mode_mapper:
            self.mode_mapper = mode_mapper
        else:
            self.mode_mapper = {mode: i for i, mode in enumerate(DEFAULT_MODES)} # e.g. "[M+H]+": 0 etc.

        # Flatten out all edges from fragments
        self.edge_intensities = []
        for mz in self.peak_matches.keys(): 
            intensity = self.peak_matches[mz]["intensity"] / sum(f.num_of_edges() for f in self.peak_matches[mz]["fragments"])
            self.peak_matches[mz]["edges"] = [e for f in self.peak_matches[mz]["fragments"] for e in f.edges] 
            for i, f in enumerate(self.peak_matches[mz]["fragments"]):
                for j, edge in enumerate(f.edges):
                    entry = (edge, {'intensity': intensity, 'fragment': f, 'break_side': f.break_sides[j], 'ion_mode': self.peak_matches[mz]["ion_modes"][i][0]})

                    self.edge_intensities.append(entry)


        self.edge_break_count = torch.zeros(size = edge_break_labels.size(), dtype=torch.float32)
        self.precursor_count, self.precursor_prob, self.precursor_sqrt_prob  = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                
        
        self.edge_count_matrix = torch.zeros(size = (edge_break_labels.shape[0], 2*len(self.mode_mapper)), dtype=torch.float32)
        
        # Determining edge break probabilites from peak intensities. Multiple edges for the same fragment -> divide by number of edges. Multiple fragments from edge -> add intensities.
        for edge, values in self.edge_intensities:
            if edge == None: # precursor
                self.precursor_count += values['intensity']
                continue
            edge_index = torch.logical_or(self.edges == torch.tensor(edge), self.edges == torch.tensor(edge[::-1])).all(dim=1).nonzero().squeeze()
            self.edge_break_count[edge_index] += values['intensity']
            
            forward_idx = ((torch.tensor(edge) == self.edges).sum(dim=1) == 2).nonzero().squeeze()
            backward_idx = ((torch.tensor(edge[::-1]) == self.edges).sum(dim=1) == 2).nonzero().squeeze()
            
            
            col = self.mode_mapper[values["ion_mode"]] if values["break_side"]=="left" else self.mode_mapper[values["ion_mode"]] + len(self.mode_mapper) 
            self.edge_count_matrix[forward_idx, col] = values['intensity']
            col = (col + len(self.mode_mapper)) % (2*len(self.mode_mapper)) #to the other side of the break
            self.edge_count_matrix[backward_idx, col] = values['intensity']
        
    
        "bond_features_one_hot",
        # Compile probability vectors  
        # self.compiled_counts = torch.cat([self.edge_break_count.flatten(), self.precursor_count.unsqueeze(dim=-1), self.precursor_count.unsqueeze(dim=-1)])
        # self.compiled_probs = 2 * self.compiled_counts / torch.sum(self.compiled_counts)
        
        # COMPILED VECTORS COUNTS & PROBABILITIES FOR END-TO-END PREDICTION! Default is compiled_probsALL
        self.compiled_countsALL = torch.cat([self.edge_count_matrix.flatten(), self.precursor_count.unsqueeze(dim=-1), self.precursor_count.unsqueeze(dim=-1)])
        self.compiled_probsALL = 2 * self.compiled_countsALL / torch.sum(self.compiled_countsALL)
        
        # SQRT transformation
        self.compiled_countsSQRT = torch.sqrt(self.compiled_countsALL)
        self.compiled_probsSQRT = 2 * self.compiled_countsSQRT / torch.sum(self.compiled_countsSQRT)
        

        
        # MASKS
        # self.compiled_validation_mask = torch.cat([self.is_edge_not_in_ring.bool().squeeze(), torch.tensor([True, True], dtype=bool)], dim=-1)
        self.compiled_validation_maskALL = torch.cat([torch.repeat_interleave(self.is_edge_not_in_ring.bool().squeeze(), len(self.mode_mapper)*2), torch.tensor([True, True], dtype=bool)], dim=-1)
        # self.compiled_forward_mask = torch.cat([self.edge_forward_direction.squeeze(), torch.tensor([True, False], dtype=bool)], dim=-1)
        
        # Track additional statistics
        max_intensity = max(int_list)
        intensity_filter_threshold = 0.01
        self.match_stats = {
            'counts': self.compiled_countsALL.sum().tolist() / 2.0, # self.compiled_counts.sum().tolist() / 2.0,
            'ms_all_counts': sum(int_list),
            'coverage': (self.compiled_countsALL.sum().tolist() / 2.0) / sum(int_list),
            'coverage_wo_prec': (self.edge_break_count.sum().tolist() / 2.0) / (sum(int_list) - self.precursor_count.tolist()),
            'precursor_prob': self.precursor_count.tolist() / (self.compiled_countsALL.sum().tolist() / 2.0) if (self.compiled_countsALL.sum().tolist() / 2.0) > 0 else 0.0,
            'precursor_raw_prob': self.precursor_count.tolist() / sum(int_list), 
            'num_peaks': len(mz_fragments),
            'num_peak_matches': len(self.peak_matches),
            'percent_peak_matches': len(self.peak_matches) / len(mz_fragments),
            'num_peaks_filtered': sum([(i / max_intensity) > intensity_filter_threshold for i in int_list]),
            'num_peak_matches_filtered': sum([match["relative_intensity"] > intensity_filter_threshold for mz, match in self.peak_matches.items()]),
            'percent_peak_matches_filtered': sum([match["relative_intensity"] > intensity_filter_threshold for mz, match in self.peak_matches.items()]) / len(mz_fragments),
            'num_non_precursor_matches': sum([(None not in match["edges"]) for mz, match in self.peak_matches.items()]),
            'num_peak_match_conflicts': sum([len(match["edges"]) > 1 for mz, match in self.peak_matches.items()]),
            'num_fragment_conflicts': sum([len(match["fragments"]) > 1 for mz, match in self.peak_matches.items()]),
            'rel_fragment_conflicts': sum([len(match["fragments"]) > 1 for mz, match in self.peak_matches.items()]) / sum([(None not in match["edges"]) for mz, match in self.peak_matches.items()]) if sum([(None not in match["edges"]) for mz, match in self.peak_matches.items()]) > 0 else 0,
            'ms_num_all_peaks': len(mz_fragments)  
        }

        if match_stats_only:
            self.free_memory()


    def free_memory(self):
        attributes_to_free = [
            "edge_break_count", "precursor_count", 
            "precursor_prob", "precursor_sqrt_prob", "edge_count_matrix", 
            "compiled_countsALL", "compiled_probsALL", "compiled_countsSQRT", 
            "compiled_probsSQRT", "compiled_validation_maskALL",
            "edge_breaks", "edge_intensities", "setup_features", 
            "setup_features_per_edge", "node_features", "node_features_one_hot", 
            "bond_features", "bond_features_one_hot"
        ] # Tensors from peak matching


        for attr in attributes_to_free:
            if hasattr(self, attr):
                delattr(self, attr)



    def as_geometric_data(self, with_labels=True):
        if with_labels:
            return Data(
                x=self.node_features,
                edge_index=self.edges.t().contiguous(),
                edge_type=self.edge_bond_types,
                edge_attr=self.bond_features,
                static_graph_features=self.setup_features,
                static_edge_features=self.setup_features_per_edge,
                static_rt_features = self.rt_setup_features,
                
                
                # labels
                y=self.edge_break_labels,
                compiled_probsALL=self.compiled_probsALL,
                compiled_probsSQRT=self.compiled_probsSQRT,
                # compiled_counts=self.compiled_counts,
                edge_break_count=self.edge_break_count,
                #edge_break_prob=self.edge_break_prob,
                #edge_break_prob_wo_precursor=self.edge_break_prob_wo_precursor,
                #edge_break_sqrt_prob=self.edge_break_sqrt_prob,
                #precursor_prob = self.precursor_prob,
                retention_time = self.rt,
                retention_mask = self.rt_mask,
                ccs = self.ccs,
                ccs_mask = self.ccs_mask,
                
                # masks and groups
                validation_mask=self.is_edge_not_in_ring.bool(),
                # compiled_validation_mask = self.compiled_validation_mask,
                compiled_validation_maskALL = self.compiled_validation_maskALL,
                
                # group identity and loss weights
                group_id=self.id,
                weight = torch.tensor([self.loss_weight]).unsqueeze(dim=-1),
                weight_tensor=torch.full(self.compiled_probsALL.shape, self.loss_weight),
                
                # Stepped collision energies
                ce_steps = self.ce_steps,
                ce_idx = self.ce_idx, # geom treats values with suffix _index differently -> avoid
                
                
                # additional information
                is_node_aromatic=self.is_node_aromatic,
                is_edge_aromatic=self.is_edge_aromatic
                )
        else:
            return Data(
                x=self.node_features,
                edge_index=self.edges.t().contiguous(),
                edge_type=self.edge_bond_types,
                edge_attr=self.bond_features,
                static_graph_features=self.setup_features,
                static_edge_features=self.setup_features_per_edge,
                static_rt_features = self.rt_setup_features,
                
                # masks and groups
                validation_mask=self.is_edge_not_in_ring.bool(),
                group_id=self.id,
                
                # additional information
                is_node_aromatic=self.is_node_aromatic,
                is_edge_aromatic=self.is_edge_aromatic
                )