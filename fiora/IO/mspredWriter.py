import os
from fiora.MOL.constants import ADDUCT_WEIGHTS

label_header = ["dataset", "spec", "name", "ionization", "formula", "smiles", "inchikey"]

def write_labels(df, output_file, label_map, from_metabolite=True):
    if from_metabolite:
        df["formula"] = df["Metabolite"].apply(lambda x: x.Formula)
        df["smiles"] = df["Metabolite"].apply(lambda x: x.SMILES)
        df["inchikey"] = df["Metabolite"].apply(lambda x: x.InChIKey)
    
    if "dataset" not in label_map.keys():
        df = df.drop(columns="dataset")
    try:
        df.rename(columns=label_map)[label_header].to_csv(output_file, index=False, sep="\t")
    except:
        raise NameError(f"Failed to write labels file. Make sure file path is correct. Make sure all headers are present in DataFrame {label_header}. Use label_map to rename columns.")

# def write_spec_files_wo_header(df, directory, spec="spec"):
#     for i, row in df.iterrows():
#         output_file = os.path.join(directory, row[spec] + ".ms")
#         with open(output_file, "w") as f:
#             f.write("> This spectrum only containing ms 2 peaks\n")
#             f.write("#No metadata\n\n")
#             f.write(">ms2peaks")
#             for j, mz in row["peaks"]["mz"]:
#                 f.write(mz + " " + row["peaks"]["intensity"][j] + "\n")


def write_spec_files(df, directory, spec_tag="spec"):
    for i, row in df.iterrows():
        output_file = os.path.join(directory, str(row[spec_tag]) + ".ms")
        with open(output_file, "w") as f:
            metabolite = row["Metabolite"]
            
            # Write header
            f.write(">compound " + row["Name"] + " \n")
            f.write(">formula " + metabolite.Formula + " \n")
            
            
            f.write(">parentmass " + str(metabolite.ExactMolWeight + ADDUCT_WEIGHTS[row["Precursor_type"]]) + " \n")
            f.write(">ionization " + row["Precursor_type"] + " \n")
            f.write(">InChi " + metabolite.InChI + " \n")
            f.write(">InChIKey " + metabolite.InChIKey + " \n")
            f.write("#smiles " + metabolite.SMILES + " \n")
            f.write("#scans " + "1" + " \n")
            f.write("#_FILE " + str(row[spec_tag]) + " \n")
            f.write("#spectrumid " + str(row[spec_tag]) + " \n")
            f.write("#InChi " + metabolite.InChI + " \n")
            f.write("\n")
            
            # Write peaks
            f.write(">ms2peaks")
            for j, mz in enumerate(row["peaks"]["mz"]):
                f.write("\n")
                f.write(str(mz) + " " + str(row["peaks"]["intensity"][j]))


def write_dataset(df, directory, label_map = {"dataset": "dataset", "spec": "spec", "name": "name", "formula": "formula", "ionization": "ionization", "smiles": "smiles", "inchikey": "inchikey"}):
    write_labels(df, output_file=os.path.join(directory, "labels.tsv"), label_map=label_map, from_metabolite=True)
    write_labels(df.iloc[::-1], output_file=os.path.join(directory, "reverse_labels.tsv"), label_map=label_map, from_metabolite=True)
    
    spec_tag = {v: k for k, v in label_map.items()}["spec"]
    spec_path = os.path.join(directory, "spec_files")
    if not os.path.exists(spec_path):
        os.mkdir(spec_path)
    write_spec_files(df, spec_path, spec_tag=spec_tag)
    
    return