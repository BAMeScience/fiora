from fiora.MOL.mol_graph import mol_to_graph, get_adjacency_matrix, get_edges
from fiora.MS.ms_utility import do_mz_values_match
import fiora.MOL.constants as constants

from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Chem.Descriptors as Descriptors

import numpy as np
from treelib import Node, Tree
from copy import copy

# TODO can a fragment be tied to more than one edge: Yes. TODO see todo case in build_frag_tree

class Fragment:

    def __init__(self, mol, edge=None, isotope_labels=None):
        
        # Track edge break and reset isotope changes
        self.edges = [edge]
        if edge:
            subgraph = [a.GetIsotope() for a in mol.GetAtoms()] #use isotope info as a proxy for node id
            break_side = "left" if edge[0] in subgraph else "right" if edge[1] in subgraph else "unidentified"
            if break_side == "unidentified": 
                print("ERROR", edge, subgraph, Chem.MolToSmiles(mol))
                raise ValueError("Unidentified edge in fragment")
            self.break_sides = [break_side]
            self.subgraphs = [subgraph]
        else:
            self.break_sides = [None]
            self.subgraphs = []
        if isotope_labels: # Reset isotope info
            for a in mol.GetAtoms():
                id = a.GetIsotope()
                a.SetIsotope(isotope_labels[id])
        
        # __init__
        self.MOL = mol
        self.smiles = Chem.MolToSmiles(mol)
        self.neutral_mass = Chem.Descriptors.ExactMolWt(mol)

        self.modes = constants.DEFAULT_MODES
        self.mz = {mode: self.neutral_mass + constants.ADDUCT_WEIGHTS[mode] for mode in self.modes}
        self.mz.update({mode.replace("]+", "]-"): self.neutral_mass + constants.ADDUCT_WEIGHTS[mode.replace("]+", "]-")] for mode in self.modes})
    
    def __eq__(self, __o: object) -> bool:
        if self.neutral_mass != __o.neutral_mass:
            return False
        return self.get_morganFinger() == __o.get_morganFinger()

    def __repr__(self):
        return "<Fragment Object> :: " + self.smiles #+ " " + str(self.mz)
    
    def __str__(self):
        return "<Fragment Object> :: " + self.smiles #+ " " + str(self.mz)

    def num_of_edges(self):
        return len(self.edges)

    def match_peak(self, mz, tolerance=None):
        for mode in self.modes:
            if do_mz_values_match(mz, self.mz[mode], tolerance=tolerance):
                return True, (mode, self.mz[mode])
        return False, None

    def set_modes(self, modes):
        self.modes = modes
        self.mz = {mode: self.neutral_mass + constants.ADDUCT_WEIGHTS[mode] for mode in self.modes}

    def set_ID(self, ID):
        self.ID = ID

    def get_tag(self):
        return str(self.mz)

    def get_morganFinger(self):
        return AllChem.GetMorganFingerprintAsBitVect(self.MOL, 2, nBits=1024)

class FragmentationTree:
    def __init__(self, root_mol):
        self.root_mol = root_mol
        self.edge_map = {None: Fragment(root_mol)}

        self.patt = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')

    def __repr__(self):
        self.fragmentation_tree.show(idhidden=False)
        return "<FragmentationTree Object>"
    
    def __str__(self):
        self.fragmentation_tree.show(idhidden=False)
        return "<FragmentationTree Object>"

    '''
    Getter
    '''

    def get_fragment(self, id):
        return self.fragmentation_tree.get_node(id).data

    def set_fragment_modes(self, modes):
        for frag in self.get_all_fragments():
            frag.set_modes(modes)
        
    def get_all_fragments_as_nodes(self):
        return self.fragmentation_tree.all_nodes()

    def get_all_fragments(self):
        return [x.data for x in self.fragmentation_tree.all_nodes()]

    '''
    Core methods
    '''

    def build_fragmentation_tree(self, mol, edge_indices, depth=2, parent_tree=None, parent_id=None):
        self.fragmentation_tree = Tree(tree=parent_tree)
        root_fragment = Fragment(mol)
        root_fragment.set_ID(self.fragmentation_tree.size())
        
        mol_isotopes = [a.GetIsotope() for a in mol.GetAtoms()] 
        for i, atom in enumerate(mol.GetAtoms()): atom.SetIsotope(i) # use isotope information as a proxy for atom_id (such that the information is not lost when carrying out the bond break)


        self.fragmentation_tree.create_node(tag=root_fragment.get_tag(), parent=parent_id, identifier=root_fragment.ID, data=root_fragment)


        listed_fragments = []
        for i,j in edge_indices:
            if i > j:
                continue
            _, fragments = self.create_Fragments(mol, i, j, original_mol_isotopes=mol_isotopes)
            self.edge_map[(i,j)] = {frag.break_sides[0]: frag for frag in fragments} # TODO Maybe choose to update edge_map later, this way the same fragment can exist multiple times due to multiple edges leading to the same fragments: Maybe fix. Maybe not.
            
            for f in fragments:

                if f is not None:
                    f_existing = self.get_Fragment_if_in_list(f, listed_fragments)
                    if f_existing:
                        f_existing.edges.append(f.edges[0])
                        f_existing.break_sides.append(f.break_sides[0])
                        continue
                    if depth == 1: # anchor
                        f.set_ID(self.fragmentation_tree.size())
                        self.fragmentation_tree.create_node(tag=f.get_tag(), identifier=f.ID, parent=root_fragment.ID, data=f)
                    else: 
                        # build graph, adjacency matrix and index edges
                        G = mol_to_graph(f.MOL)
                        A = get_adjacency_matrix(G)
                        edge_indices = get_edges(A)
                        f.set_ID(self.fragmentation_tree.size())
                        self.fragmentation_tree = self.build_fragmentation_tree(f.MOL, edge_indices, depth=depth-1, parent_tree=self.fragmentation_tree, parent_id=root_fragment.ID)

                    listed_fragments.append(f)
                    
        for i, atom in enumerate(mol.GetAtoms()): atom.SetIsotope(mol_isotopes[i]) # Reset isotope information
        return self.fragmentation_tree

    def create_Fragments(self, mol, i, j, original_mol_isotopes=None):
        
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        if bond.IsInRing():
            return None, []
        try:
            new_mol, fragment_mols = self.break_bond(mol, int(i), int(j))
        except (Chem.AtomKekulizeException, Chem.KekulizeException):
            new_mol = None
            fragment_mols = []
        else:
            if len(fragment_mols) < 1: #TODO resolve ring break
                #fragment_mols = [fragment_mols[0]]
                pass
        return new_mol, [Fragment(m, edge=(int(i), int(j)), isotope_labels=original_mol_isotopes) for m in fragment_mols]


    def is_Fragment_in_list(self, fragment, fragment_list):
        for f in fragment_list:
            if fragment == f:
                return True
        return False

    def get_Fragment_if_in_list(self, fragment, fragment_list):
        for f in fragment_list:
            if fragment == f:
                return f
        return None


    def match_peak_list(self, mz_list, int_list=None, tolerance=None):
        fragments = self.get_all_fragments()
        matches = {}
        if not int_list:
            int_list = [0] * len(mz_list) 
            
        # Compare mz list to all fragments
        for i, mz in enumerate(mz_list):
            was_peak_matched_already = False
            for frag in fragments:
                does_match, frag_ion = frag.match_peak(mz, tolerance=tolerance)
                if does_match:
                    if was_peak_matched_already:
                        matches[mz]['fragments'] += [frag] # Report fragment for each edge leading to it
                        matches[mz]['ion_modes'] += [frag_ion]
                    else: 
                        matches[mz] = {
                            'intensity': int_list[i] if int_list else None,
                            'fragments': [frag], # Report fragment for each edge leading to it
                            'ion_modes': [frag_ion]
                            }
                    was_peak_matched_already = True
        
        # Normalize intensity values
        # if sum_matched == 0:
        #     return matches
        
        sum_intensity = sum([m["intensity"] for mz, m in matches.items() if m["intensity"] is not None])
        if sum_intensity > 0:
            for mz in matches.keys():
                int_value = matches[mz]['intensity']
                matches[mz]['relative_intensity'] = int_value / sum_intensity # only considered matched peaks
        
               
        # for mz in matches.keys():
        #     int_value = matches[mz]['intensity']
        #     matches[mz]['relative_intensity'] = int_value / sum_matched # only considered matched peaks
        #     matches[mz]['relative_sqrt_intensity'] = np.sqrt(int_value)  / sum_of_sqrt
        #     matches[mz]['relative_intensity_wo_precursor'] = int_value / sum_match_no_precursor if sum_match_no_precursor > 0 else 0.0 # only considered matched peaks
        #     matches[mz]['total_relative_intensity'] = int_value / sum_total

        return matches





    '''
    Old methods
    '''


    def build_fragmentation_tree_from_fraggraph_df(self, df):
        return

    def build_fragmentation_tree_by_rotatable_bond_breaks(self, depth=1):
        bonds = self.root_mol.GetSubstructMatches(self.patt)
        mol = self.root_mol
        em = Chem.EditableMol(mol)
        nAts = mol.GetNumAtoms()
        for a,b in bonds:
            em.RemoveBond(a,b)
            em.AddAtom(Chem.Atom(0))
            em.AddBond(a,nAts,Chem.BondType.SINGLE)
            em.AddAtom(Chem.Atom(0))
            em.AddBond(b,nAts+1,Chem.BondType.SINGLE)
            nAts+=2
        p = em.GetMol()
        Chem.SanitizeMol(p)

        smis = [Chem.MolToSmiles(x,True) for x in Chem.GetMolFrags(p,asMols=True)]
        for smi in smis: print(smi)


        #from rdkit.Chem import BRICS
        #bonds = [((x,y),(0,0)) for x,y in bonds]
        #
        # 
        #p = BRICS.BreakBRICSBonds(mol,bonds=bonds)
        #
        #smis = [Chem.MolToSmiles(x,True) for x in Chem.GetMolFrags(p,asMols=True)]
        #for smi in smis: print(smi)
        #
        return smi


    def build_fragmentation_tree_by_single_edge_breaks(self, mol, edge_indices, depth=2, parent_tree=None, parent_id=None):

        self.fragmentation_tree = Tree(tree=parent_tree)
        ID = self.fragmentation_tree.size()
        self.fragmentation_tree.create_node(tag=Chem.Descriptors.ExactMolWt(mol), parent=parent_id, identifier=ID, data=mol)


        listed_fragments = []


        for i,j in edge_indices:
            _, fragments = self.create_fragments(mol, i, j)
            for f in fragments:
                if f is not None:
                    if self.is_fragment_in_list(f, listed_fragments):
                        continue
                    if depth == 1: # anchor
                        self.fragmentation_tree.create_node(tag=Chem.Descriptors.ExactMolWt(f), identifier=self.fragmentation_tree.size(), parent=ID, data=f)
                    else: # recursion TODO OPTIMIZE
                        # build graph, adjacency matrix and index edges
                        G = mol_to_graph(f)
                        A = get_adjacency_matrix(G)
                        edge_indices = get_edge_indices(A)
                        self.fragmentation_tree = self.build_fragmentation_tree_by_single_edge_breaks(f, edge_indices, depth=depth-1, parent_tree=self.fragmentation_tree, parent_id=ID)

                    listed_fragments.append(f)
        return self.fragmentation_tree

    def break_bond(self, mol, i,j, add_dummy_atoms=False):
        num_atoms = mol.GetNumAtoms()

        em = Chem.EditableMol(mol)
        em.RemoveBond(i, j)
        
        if add_dummy_atoms:
            em.AddAtom(Chem.Atom(0)) #
            em.AddBond(i,num_atoms,Chem.BondType.SINGLE) #
            em.AddAtom(Chem.Atom(0)) #
            em.AddBond(j,num_atoms+1,Chem.BondType.SINGLE) #
            
        new_mol = em.GetMol() 
        Chem.SanitizeMol(new_mol) #

        frags = Chem.GetMolFrags(new_mol, asMols=True)
        return new_mol, frags


    def create_fragments(self, mol, i, j):
        try:
            new_mol, fragments = self.break_bond(mol, int(i), int(j))

        except (Chem.AtomKekulizeException, Chem.KekulizeException):
                #print(i,j, "Error", Chem.AtomKekulizeException)
            new_mol = None
            fragments = [None, None]
        else:
            if len(fragments) < 1:
                #TODO resolve ring break
                fragments = [fragments[0], None]
        return new_mol, fragments 

    def morganFinger(self, x):
        return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)

    def equalMols(self, mol, other):

        funcs = [Chem.Descriptors.ExactMolWt, self.morganFinger, AllChem.GetMACCSKeysFingerprint]
        #func = Chem.Descriptors.ExactMolWt # TODO add more here !!!!! When are mols equal????
        for func in funcs:
            if func(mol) == func(other):
                continue
            else:
                return False
        return True

    def is_fragment_in_list(self, fragment, fragment_list):
        for f in fragment_list:
            if self.equalMols(fragment, f):
                return True
        return False
