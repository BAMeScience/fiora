import os

label_header = ["dataset", "spec", "name", "formula", "ionization", "smiles", "inchikey"]

def write_labels(df, output_file, label_map):
    try:
        df.rename(columns=label_map)[label_header].to_csv(output_file, index=False, sep="\t")
    except:
        raise NameError(f"Failed to write labels file. Make sure file path is correct. Make sure all headers are present in DataFrame {label_header}. Use label_map to rename columns.")

def write_labels_and_ms_files(df, directory, label_map = {"dataset": "dataset", "spec": "spec", "name": "name", "formula": "formula", "ionization": "ionization", "smiles": "smiles", "inchikey": "inchikey"}):
    
    
    write_labels(df, output_file=os.path.join(directory, "labels.csv"), label_map=label_map)
    
    
    raise NotImplementedError("Missing spectral")
    return