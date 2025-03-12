import torch
import torch_geometric as geom
import pandas as pd
from typing import Literal, Dict
import matplotlib.pyplot as plt

from fiora.MOL.Metabolite import Metabolite
from fiora.MS.spectral_scores import *
import fiora.visualization.spectrum_visualizer as sv

class SimulationFramework:
    
    def __init__(self, base_model: torch.nn.Module|None=None, with_RT=False, with_CCS=False, dev: str="cpu"):
        self.base_model = base_model
        self.dev = dev
        self.mode_mapper = None
        self.with_RT = with_RT
        self.with_CCS = with_CCS

    def __repr__(self):
        return f"Simulation framework for MS/MS spectrum generation"
    
    def __str__(self):
        return f"Simulation framework for MS/MS spectrum generation"

    def set_mode_mapper(self, mode_mapper):
        self.mode_mapper = mode_mapper

    def predict_metabolite_property(self, metabolite, model: torch.nn.Module|None=None, as_batch: bool=False):
        if not model:
            model = self.base_model 
        data = metabolite.as_geometric_data(with_labels=False).to(self.dev)
        if as_batch:
            data = geom.data.Batch.from_data_list([data])
        
        logits = model(data, with_RT=self.with_RT, with_CCS=self.with_CCS)
        return logits
    
    def pred_all(self, df: pd.DataFrame, model: torch.nn.Module|None=None, attr_name: str="", as_batch: bool=True):
        with torch.no_grad():
            model.eval()

            for i,d in df.iterrows():
                metabolite = d["Metabolite"]
                prediction = self.predict_metabolite_property(metabolite, model=model, as_batch=as_batch)
                if self.with_RT:
                    setattr(metabolite, attr_name + "_pred", prediction["fragment_probs"])
                    setattr(metabolite, "RT_pred", prediction["rt"].squeeze())
                else:
                    setattr(metabolite, attr_name + "_pred", prediction["fragment_probs"])
        return

    
    def simulate_spectrum(self, metabolite: Metabolite, pred_label: str, precursor_mode: Literal["[M+H]+", "[M-H]-"]="[M+H]+", min_intensity: float=0.001, merge_fragment_duplicates: bool=True, transform_prob: str="None"):

        if not self.mode_mapper:
            mode_mapper = metabolite.mode_mapper
        else:
            mode_mapper = self.mode_mapper
    
        edge_map = metabolite.fragmentation_tree.edge_map
        
        sim_probs = getattr(metabolite, pred_label)
        sim_peaks = {'mz': [], 'intensity': [], 'annotation': []}
        
        precursor_prob = sim_probs[-1].tolist()
        precursor = edge_map[None]
        
        sim_peaks["mz"].append(precursor.mz[precursor_mode]) # TODO allow multiple ion modes of precursor
        sim_peaks["intensity"].append(precursor_prob)
        sim_peaks["annotation"].append(precursor.smiles + "//" + precursor_mode)

        edge_probs = sim_probs[:-2].unflatten(-1, sizes=(-1, len(mode_mapper)*2))

        for i, edge in enumerate(metabolite.edges_as_tuples):
            if edge[0] > edge[1]: continue # skip backward directions
            frags = edge_map.get(edge)
            if not frags: continue
            
            lf = frags.get('left')
            if lf:
                for mode, idx in mode_mapper.items():
                    intensity = edge_probs[i,idx].tolist()
                    if intensity > min_intensity:
                        mz =lf.mz[mode]
                        mode_str = mode if precursor_mode=="[M+H]+" else mode.replace("]+", "]-")
                        annotation = lf.smiles + "//" + mode_str
                        merged = False
                        if merge_fragment_duplicates and (mz in sim_peaks["mz"]): # if exact mz value exists already
                            for j, mzx in enumerate(sim_peaks["mz"]):
                                if mz == mzx and annotation == sim_peaks["annotation"][j]: # check mz and annotation
                                    sim_peaks["intensity"][j] += intensity # and intensity if exact same fragments
                                    merged = True
                                    break
                        if merged: continue
                        sim_peaks["mz"].append(mz)
                        sim_peaks["intensity"].append(intensity)
                        sim_peaks["annotation"].append(annotation)
            rf = frags.get('right')
            if rf:
                for mode, idx in mode_mapper.items():
                    idx = (idx + len(mode_mapper)) % (2*len(mode_mapper))
                    intensity = edge_probs[i,idx].tolist()

                    if intensity > min_intensity:
                        mz = rf.mz[mode]
                        mode_str = mode if precursor_mode=="[M+H]+" else mode.replace("]+", "]-")
                        annotation = rf.smiles + "//" + mode_str
                        merged = False
                        if merge_fragment_duplicates and (mz in sim_peaks["mz"]):
                            for j, mzx in enumerate(sim_peaks["mz"]):
                                if mz == mzx and annotation == sim_peaks["annotation"][j]:
                                    sim_peaks["intensity"][j] += intensity # and intensity if exact same fragments
                                    merged = True
                                    break 
                        if merged: continue
                        sim_peaks["mz"].append(mz)
                        sim_peaks["intensity"].append(intensity)
                        sim_peaks["annotation"].append(annotation)    
        
        if transform_prob == "square":
            max_prob = max(sim_peaks["intensity"])**2
            for i in range(len(sim_peaks["intensity"])):
                sim_peaks["intensity"][i] == sim_peaks["intensity"][i]**2 / max_prob
                
        return sim_peaks
    
 
    
    def simulate_and_score(self, metabolite: Metabolite, model: torch.nn.Module|None=None, base_attr_name: str="compiled_probsALL", query_peaks: Dict|None=None, as_batch: bool=True, min_intensity: float=0.001):
        prediction = self.predict_metabolite_property(metabolite, model=model, as_batch=as_batch)
        stats = {}

        if self.with_RT:
            stats["RT_pred"] = prediction["rt"].squeeze().tolist()
        if self.with_CCS:
            stats["CCS_pred"] = prediction["ccs"].squeeze().tolist()
            
        setattr(metabolite, base_attr_name + "_pred", prediction["fragment_probs"])
        transform_prob = "square" if ("training_label" in model.model_params and model.model_params["training_label"] == "compiled_probsSQRT") else "None"
        stats["sim_peaks"] = self.simulate_spectrum(metabolite, base_attr_name + "_pred", precursor_mode=metabolite.metadata["precursor_mode"], transform_prob=transform_prob, min_intensity=min_intensity)
        
        
        # Score performance if groundtruth is available
        if hasattr(metabolite, base_attr_name):
            groundtruth = getattr(metabolite, base_attr_name).to(self.dev)
        
            stats["cosine_similarity"] = torch.nn.functional.cosine_similarity(prediction["fragment_probs"], groundtruth, dim=0).tolist() # TODO
            stats["kl_div"] = torch.nn.functional.kl_div(torch.log(prediction["fragment_probs"]), groundtruth, reduction='sum').tolist()
            
        if self.with_RT and "retention_time" in metabolite.metadata.keys():
            stats["RT_dif"] = abs(stats["RT_pred"] - metabolite.metadata["retention_time"])
        
        if query_peaks:
            stats["spectral_cosine"], stats["spectral_bias"] = spectral_cosine(query_peaks, stats["sim_peaks"], with_bias=True)
            stats["spectral_sqrt_cosine"], stats["spectral_sqrt_bias"] = spectral_cosine(query_peaks, stats["sim_peaks"], transform=np.sqrt, with_bias=True)
            stats["spectral_sqrt_cosine_wo_prec"], stats["spectral_sqrt_bias_wo_prec"] = spectral_cosine(query_peaks, stats["sim_peaks"], transform=np.sqrt, remove_mz=metabolite.get_theoretical_precursor_mz(ion_type=metabolite.metadata["precursor_mode"]), with_bias=True)
            stats["spectral_sqrt_cosine_avg"], stats["spectral_sqrt_bias_avg"] = (stats["spectral_sqrt_cosine"] + stats["spectral_sqrt_cosine_wo_prec"]) / 2.0, (stats["spectral_sqrt_bias"] + stats["spectral_sqrt_bias_wo_prec"]) / 2.0
            stats["spectral_refl_cosine"], stats["spectral_refl_bias"] = spectral_reflection_cosine(query_peaks, stats["sim_peaks"], transform=np.sqrt, with_bias=True)
            stats["steins_cosine"], stats["steins_bias"] = reweighted_dot(query_peaks, stats["sim_peaks"], int_pow=0.5, mz_pow=0.5, with_bias=True)
        return stats
    
    def simulate_all(self, df: pd.DataFrame, model: torch.nn.Module|None=None, base_attr_name: str="compiled_probsALL", suffix: str="", groundtruth=True, min_intensity: float=0.001):
        
        with torch.no_grad():
            model.eval()

            for i,data in df.iterrows():
                metabolite = data["Metabolite"]
                stats = self.simulate_and_score(metabolite, model, base_attr_name, query_peaks=data["peaks"] if groundtruth else None, min_intensity=min_intensity)
                df = pd.concat([df, pd.DataFrame(columns=[x + suffix for x in stats.keys()])]) # Add new empty columns for all statistics
                
                for key, value in stats.items():
                    if key + suffix in df.columns:
                        df.at[i, key + suffix] = value
                        setattr(metabolite, key + suffix, value)
                    else:
                        raise Warning("User Warning: Attempting to add data to non-existing column simulate_all().\n\tSolve by adding column with pd.concat()")
                
        return df
    
    
    def plot_feature_prediction_vectors(self, metabolite, label, with_mol=True, transform=None):
    
        if with_mol:
            fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.2), gridspec_kw={'width_ratios': [1, 3]}, sharey=False)
            img = metabolite.draw(ax=axs[0])
            axs1 = axs[1]
        else:
            fig, axs1 = plt.subplots(1, 1, figsize=(12.8, 4.2))
        
        relevant_edge_index = torch.logical_and(metabolite.compiled_validation_mask, metabolite.compiled_forward_mask)
        probs = getattr(metabolite, label).to(dev)[relevant_edge_index]
        preds = getattr(metabolite, 'predicted_' + label).to(self.dev)[relevant_edge_index]
        
        names = [f"e{i}" for i in range(preds.shape[0] - 1)] + ["prec"]

        ax = sv.plot_vector_spectrum(probs.tolist(), preds.tolist(), ax=axs[1], names=names)
        plt.show()
        
    
