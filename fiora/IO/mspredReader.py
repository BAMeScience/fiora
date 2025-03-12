import json
import pandas as pd
import os
from collections import defaultdict


# Function was adjusted using code from https://github.com/samgoldman97/ms-pred
def convert_dict_to_mz(values):
    
    peak_dict = defaultdict(lambda: {})
    for k, val in values["frags"].items():
        masses, intens = val["mz_charge"], val['intens']
        for m, i in zip(masses, intens):
            if i <= 0:
                continue
            current_peak_object = peak_dict[m]
            if current_peak_object.get("inten", 0) > 0:
                # update
                if current_peak_object.get("inten") < i:
                    current_peak_object["frag_hash"] = k
                current_peak_object["inten"] += i
            else:
                current_peak_object["inten"] = i
                current_peak_object["frag_hash"] = k

    max_inten = max(*[i["inten"] for i in peak_dict.values()], 1e-9)
    peak_dict = {
        k: dict(inten=v["inten"] / max_inten, frag_hash=v["frag_hash"])
        for k, v in peak_dict.items()
    }
    
    
    peaks = {"mz": [], "intensity": []}
    for k, v in peak_dict.items():
        peaks["mz"].append(k)
        peaks["intensity"].append(v["inten"])
    
    return peaks


def read(dir):
    spectra = []
    
    for file in os.listdir(dir):
        if file.endswith(".json"):
            temp_dict = {"file": file, "name": file.split(".")[0].replace("pred_", "")}
            try:
                with open(os.path.join(dir, file), 'r') as fp:
                    values = json.load(fp)
                    peaks = convert_dict_to_mz(values)
                temp_dict["peaks"] = peaks 
                
                spectra.append(temp_dict)
            except:
                print(f"Warning: unable to read {file}")
    
    return pd.DataFrame(spectra)