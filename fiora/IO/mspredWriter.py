import os

label_header = ["dataset", "spec", "name", "formula", "ionization", "smiles", "inchikey"]

def write_labels(df, output_file, label_map):
    try:
        df.rename(columns=label_map)[label_header].to_csv(output_file, index=False, sep="\t")
    except:
        raise NameError(f"Failed to write labels file. Make sure file path is correct. Make sure all headers are present in DataFrame {label_header}. Use label_map to rename columns.")

def write_spec_files(df, directory, spec="spec"):
    for i, row in df.iterrows():
        output_file = os.path.join(directory, row[spec] + ".ms")
        with open(output_file, "w") as f:
            f.write("> This spectrum only containing ms 2 peaks\n")
            f.write("#No metadata\n\n")
            f.write(">ms2peaks")
            for j, mz in row["peaks"]["mz"]:
                f.write(mz + " " + row["peaks"]["intensity"][j] + "\n")

def write_dataset(df, directory, label_map = {"dataset": "dataset", "spec": "spec", "name": "name", "formula": "formula", "ionization": "ionization", "smiles": "smiles", "inchikey": "inchikey"}):
    
    
    write_labels(df, output_file=os.path.join(directory, "labels.csv"), label_map=label_map)
    
    spec_tag = {v: k for k, v in label_map.items()}["spec"]
    write_spec_files(df, directory, spec=spec_tag)
    
    return