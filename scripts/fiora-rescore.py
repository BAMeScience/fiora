import numpy as np
import pandas as pd
import logging

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*")

import argparse
import os

import fiora.visualization.spectrum_visualizer as sv
import fiora.IO.mgfReader as mgfReader

from fiora.GNN.GNNModules import GNNCompiler
from fiora.MS.SimulationFramework import SimulationFramework
from fiora.MOL.Metabolite import Metabolite
from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.SetupFeatureEncoder import SetupFeatureEncoder

from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.collision_energy import NCE_to_eV
from fiora.MOL.constants import ADDUCT_WEIGHTS

from fiora.MS.spectral_scores import spectral_cosine
from fiora.MS.spectral_scores import spectral_reflection_cosine
from fiora.MS.spectral_scores import reweighted_dot


def simulate_fiora(smiles, model, meta):
    metabolite = Metabolite(smiles)

    CE_upper_limit = 80.0
    weight_upper_limit = 800.0

    node_encoder = AtomFeatureEncoder(feature_list=["symbol", "num_hydrogen", "ring_type"])
    bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
    setup_encoder = SetupFeatureEncoder(feature_list=["collision_energy", "molecular_weight", "precursor_mode", "instrument"])
    rt_encoder = SetupFeatureEncoder(feature_list=["molecular_weight", "precursor_mode", "instrument"])

    setup_encoder.normalize_features["collision_energy"]["max"] = CE_upper_limit 
    setup_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 
    rt_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit

    fiora_run = SimulationFramework(model, True, True, dev="cuda:3")

    metabolite.create_molecular_structure_graph()
    metabolite.compute_graph_attributes(node_encoder, bond_encoder)
    metabolite.fragment_MOL()
    
    metabolite.add_metadata(meta, setup_encoder, rt_encoder)

    prediction = fiora_run.predict_metabolite_property(metabolite, model, as_batch=True)
    setattr(metabolite, "fragment_probs", prediction["fragment_probs"]) # metabolite.fragment_probs = prediction["fragment_probs"]
    simulated_spectrum = fiora_run.simulate_spectrum(metabolite, pred_label="fragment_probs")

    #print(simulated_spectrum)

    # formatted_sim_spectrum = {'peaks' : simulated_spectrum}
    # sv.plot_spectrum (formatted_sim_spectrum)
    # plt.ylim(0, 1.1)

    
    return simulated_spectrum, metabolite

def import_positive_candidates(csv_dir):
    positive_candidate_dfs = []
    for i in range (82, 209):
        if i < 100:
            candidate_df = pd.read_csv(f"{csv_dir}/Challenge-0{i}.csv")
        else:
            candidate_df = pd.read_csv(f"{csv_dir}/Challenge-{i}.csv")
        positive_candidate_dfs.append(candidate_df)
    return positive_candidate_dfs

def import_negative_candidates(csv_dir):
    negative_candidate_dfs = []
    negative_candidate_dfs = []
    for i in range (1, 82):
        if i < 10:
            candidate_df = pd.read_csv(f"{csv_dir}/Challenge-00{i}.csv")
        else:
            candidate_df = pd.read_csv(f"{csv_dir}/Challenge-0{i}.csv")
        negative_candidate_dfs.append(candidate_df)
    return negative_candidate_dfs

def import_candidates(csv_dir):
    candidate_dfs = []
    for file_name in os.listdir(csv_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(csv_dir, file_name)
            df = pd.read_csv(file_path)
            candidate_dfs.append(df)

    return candidate_dfs

def read_sirius_output(tsv_dir):
    candidate_dfs = []
    selected_columns = ["smiles", "CSI:FingerIDScore", "name"]
    for root, dirs, files in os.walk(tsv_dir):
        for file_name in files:
            if file_name.endswith("structure_candidates.tsv"):
                file_path = os.path.join(root, file_name)
                sirius_df = pd.read_csv(file_path, sep='\t', usecols=selected_columns)
                candidate_dfs.append(sirius_df)
    
    return candidate_dfs


def add_metabolite_info_to_df(df, i, metabolite, meta):
    try:
        df.at[i, "ExactMolMass"] = metabolite.ExactMolWeight
        df.at[i, "CollisionEnergy"] = meta["collision_energy"]
        df.at[i, "PrecursorProbability"] = metabolite.match_stats["precursor_prob"]
        df.at[i, "Coverage"] = metabolite.match_stats["coverage"]
    except Exception as e:
        logging.error(e)
        
def add_spectrum_score_info_to_df(df, spectrum_df, i, metabolite, simulated_spectrum):
    try:
        metabolite.match_fragments_to_peaks(spectrum_df['peaks']['mz'], spectrum_df['peaks']['intensity'])
        df.at[i, "Spectrum"] = simulated_spectrum
        df.at[i, "Metabolite"] = metabolite
        score_cosine, cosine_bias = spectral_cosine(simulated_spectrum, spectrum_df['peaks'], transform=np.sqrt, with_bias=True)
        score_reflection = spectral_reflection_cosine(simulated_spectrum, spectrum_df['peaks'], transform=np.sqrt)
        score_reweighted, reweighted_bias = reweighted_dot(simulated_spectrum,  spectrum_df['peaks'], int_pow=0.5, mz_pow=0.5, with_bias=True)
        df.at[i, "CosineScore"] = score_cosine
        df.at[i, "CosineBias"] = cosine_bias
        df.at[i, "ReflectionScore"] = score_reflection
        df.at[i, "ReweightedScore"] = score_reweighted
        df.at[i, "ReweightedBias"] = reweighted_bias
    except Exception as e:
        print(df)
        
def format_candidate_df(df, solution_spectrum_df, model, pos_or_neg):
    precursor_mode = "[M+H]+"
    if pos_or_neg == "neg":
        precursor_mode = "[M-H]-"
        
    for j in range (0, len(df[:10])):
        df[j]["ExactMolMass"] = None
        df[j]["CollisionEnergy"] = None
        df[j]["PrecursorProbability"] = None
        df[j]["Coverage"] = None
        df[j]["Spectrum"] = None
        df[j]["Metabolite"] = None
        df[j]["CosineScore"] = None
        df[j]["CosineBias"] = None
        df[j]["ReflectionScore"] = None
        df[j]["ReweightedScore"] = None
        df[j]["ReweightedBias"] = None
        df[j]["Challenge"] = None
        for i, row in df[j].iterrows():
            smiles = row["SMILES"]
            meta = {
                "collision_energy":  NCE_to_eV(35, (row["MonoisotopicMass"] + 1)),
                "instrument": "HCD", "name": row["CompoundName"], "precursor_mode": precursor_mode
            }
            
            try:
                simulated_spectrum, metabolite = simulate_fiora(smiles, model, meta)
                print(df[j])
                add_spectrum_score_info_to_df(df[j], solution_spectrum_df[j], i, metabolite, simulated_spectrum)
                add_metabolite_info_to_df(df[j], i, metabolite, meta)
            except Exception as e:
                print(e)
                
        if pos_or_neg == "pos":
            if j < 18:
                df[j]["Challenge"] = f"Challenge-0{j + 82}"
            else:    
                df[j]["Challenge"] = f"Challenge-{j + 82}"
        elif pos_or_neg == "neg":    
            if j < 9:
                df[j]["Challenge"] = f"Challenge-00{j + 1}"
            elif j < 99:
                df[j]["Challenge"] = f"Challenge-0{j + 1}"
            else:
                df[j]["Challenge"] = f"Challenge-{j + 1}"
            
def format_sirius_df(df, solution_spectrum_df, model, pos_or_neg):
    precursor_mode = "[M+H]+"
    if pos_or_neg == "neg":
        precursor_mode = "[M-H]-"
    
    for j in range (0, len(df[:10])):
        df[j]["ExactMolMass"] = None
        df[j]["CollisionEnergy"] = None
        df[j]["PrecursorProbability"] = None
        df[j]["Coverage"] = None
        df[j]["Spectrum"] = None
        df[j]["CosineScore"] = None
        df[j]["CosineBias"] = None
        df[j]["ReflectionScore"] = None
        df[j]["ReweightedScore"] = None
        df[j]["ReweightedBias"] = None
        df[j]["Challenge"] = None
        for i, row in df[j].iterrows():
            smiles = row["smiles"]
            try:
                mol_weight = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
            except:
                continue
            #print(mol_weight)
            meta = {
                "collision_energy":  NCE_to_eV(35, (mol_weight + ADDUCT_WEIGHTS[precursor_mode])),
                "instrument": "HCD", "name": row["name"], "precursor_mode": precursor_mode
            }
            
            try:
                simulated_spectrum, metabolite = simulate_fiora(smiles, model, meta)
            except Exception as e:
                print(e)
            
            add_spectrum_score_info_to_df(df[j], solution_spectrum_df[j], i, metabolite, simulated_spectrum)
            add_metabolite_info_to_df(df[j], i, metabolite, meta)
            
        if pos_or_neg == "pos":
            if j < 18:
                df[j]["Challenge"] = f"Challenge-0{j + 82}"
            else:    
                df[j]["Challenge"] = f"Challenge-{j + 82}"
        elif pos_or_neg == "neg":    
            if j < 9:
                df[j]["Challenge"] = f"Challenge-00{j + 1}"
            elif j < 99:
                df[j]["Challenge"] = f"Challenge-0{j + 1}"
            else:
                df[j]["Challenge"] = f"Challenge-{j + 1}"
           
            

def combine_challenges(df_pos, df_neg):
    pos_concat = pd.concat(df_pos, axis=0)
    neg_concat = pd.concat(df_neg, axis=0)
    pos_concat["SpectrumType"] = "Positive"     # for visualization pos vs neg
    neg_concat["SpectrumType"] = "Negative"
    pos_neg_combined = pd.concat([pos_concat, neg_concat], ignore_index=True)
    return pos_neg_combined

def mark_solutions(df, solution_df):
    df["source"] = "Candidate"
    for i, row in solution_df.iterrows():
        challenge_name = row["ChallengeName"]
        inchikey = row["INCHIKEY"]
        condition = (df["Challenge"] == challenge_name) & (df["InChIKey"] == inchikey)
        df.loc[condition, "source"] = "Solution"
    
        
def parse_args():
    parser = argparse.ArgumentParser(prog='fiora-rescore',
                    description='Use this script to input a list generated by a scoring algorithm and receive a rescored list as output.',
                    epilog='Disclaimer:\nNo prediction software is perfect. This is an early prototype. Use with caution.')
    parser.add_argument("-m", "--mgfinput", help="input containing mgf files ", type=str, required=True)
    parser.add_argument("-c", "--csvinput", help="input containing tsv/csv files ", type=str, required=True)
    parser.add_argument("-o", "--output", help="output file path (.csv file)", type=str, required=True) # degistir
    parser.add_argument("--model", help="path to prediction model (.pt file)", type=str, default="default")
    parser.add_argument("--dev", help="Device to the model. For example cuda:0 for GPU number 0.", type=str, default="cpu")

    args = parser.parse_args()

    return args

# def main():    # test_casmi

#     args = parse_args()
#     print(f"Rescore running with following parameters: {args}\n")

#     model = args.model
#     # model_path = "/home/lbarbut/pretrained_models/v0.0.1_merged_depth2.pt"
#     try:
#         model = GNNCompiler.load(model)
#     except:
#         try:
#             model = GNNCompiler.load_from_state_dict(model)
#         except:
#             logging.error("Couldn't load model. Exiting...")
#             exit(1)

#     model.eval()
#     model = model.to(args.dev)

#     challenge_spectra = mgfReader.read(args.mgfinput, '\t')
#     candidate_df = import_candidates(args.csvinput)

#     format_candidate_df(candidate_df, challenge_spectra, model, "pos")

#     try:
#         output_df = pd.concat(candidate_df)
#         output_df.to_csv(args.output, index=False)
#         print("output csv created successfully")
#     except Exception as e:
#         logging.error(e)
#         exit(1)

    
def main():
    args = parse_args()
    print(f"Rescore running with following parameters: {args}\n")
    
    model = args.model
    # model_path = "/home/lbarbut/pretrained_models/v0.0.1_merged_depth2.pt"
    try:
        model = GNNCompiler.load(model)
    except:
        try:
            model = GNNCompiler.load_from_state_dict(model)
        except:
            logging.error("Couldn't load model. Exiting...")
            exit(1)
    
    model.eval()
    model = model.to("cuda:3")
    
    challenge_spectra = mgfReader.read(args.mgfinput, '\t')
    candidate_df = read_sirius_output(args.csvinput)
    
    format_sirius_df(candidate_df, challenge_spectra, model, "pos")
    
    try:
        output_df = pd.concat(candidate_df)
        output_df.to_csv(args.output, index=False)
        logging.info("output csv created successfully")
    except Exception as e:
        logging.error(e)
        exit(1)
        
if __name__ == "__main__":
    main()
