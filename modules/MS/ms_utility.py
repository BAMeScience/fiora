#from modules.MOL.FragmentationTree import FragmentationTree
from modules.MOL.constants import PPM, DEFAULT_PPM, MIN_ABS_TOLERANCE
from typing import Literal
import numpy as np
import copy

def do_mz_values_match(mz, mz_other, tolerance, in_ppm=True, require_minimum_tolerance=True):
    if in_ppm: tolerance = tolerance * mz 
    if tolerance<MIN_ABS_TOLERANCE and require_minimum_tolerance:
        tolerance = MIN_ABS_TOLERANCE
    return abs(mz - mz_other) <= tolerance

def find_matching_peaks(mz, mz_list, tolerance=None):
    matches = []
    use_default_tolerance = (tolerance == None)
    for p in mz_list:
        if use_default_tolerance:
            tolerance = p * DEFAULT_PPM
        if do_mz_values_match(mz, p, tolerance):
            matches.append(p)
    return matches, len(matches)        

def match_fragment_lists(mz_list, other_mz_list, tolerance=None):
    uniques = []
    multiples = []
    unidentified = []
    for mz in mz_list:
        ff,n = find_matching_peaks(mz, other_mz_list, tolerance)
        if n==1:
            uniques.append((mz, ff))
        elif n > 0:
            multiples.append((mz, ff))
        elif n == 0:
            unidentified.append(mz)
    
    return uniques, multiples, unidentified


def normalize_spectrum(spec, type: Literal["max_intensity", "norm"]="norm"):
    if type=="max_intensity":
        maximum = max(spec["intensity"])
        spec["intensity"] = [i / maximum for i in spec["intensity"]]
    elif type=="norm":
        spec["intensity"] = list(np.array(spec["intensity"]) / np.linalg.norm(spec["intensity"]) )
    else:
        raise ValueError("Unknown type of normalization")


def merge_annotated_spectrum(spec1, spec2):
    spec1 = copy.deepcopy(spec1)
    spec2_red = {"mz": [], 'intensity': [], 'annotation': []}
    for i, mz2 in enumerate(spec2["mz"]):
        merged_peak = False
        if (mz2 in spec1["mz"]):
            for j, mz1 in enumerate(spec1["mz"]):
                if mz1 == mz2 and spec1["annotation"][j] == spec2["annotation"][i]:
                    spec1["intensity"][j] += spec2["intensity"][i]
                    merged_peak = True
                    break
        if not merged_peak:
            spec2_red["mz"] += [spec2["mz"][i]]
            spec2_red["intensity"] += [spec2["intensity"][i]]
            spec2_red["annotation"] += [spec2["annotation"][i]]
                
    spec1["mz"] += spec2_red["mz"]
    spec1["intensity"] += spec2_red["intensity"]
    spec1["annotation"] += spec2_red["annotation"]
    
    return spec1




def merge_spectrum(spec1, spec2, merge_tolerance: float=0.0):
    spec1 = copy.deepcopy(spec1)
    if merge_tolerance > 0.01:
        raise Warning("Merging peaks recommended only for very small tolerances. Peak merging has mainly a visual impact is not needed for computation.")
    spec2_red = {"mz": [], 'intensity': []}
    for i, mz2 in enumerate(spec2["mz"]):
        merged_peak = False
        
        for j, mz1 in enumerate(spec1["mz"]):
            if abs(mz1 - mz2) <= merge_tolerance:
                spec1["intensity"][j] += spec2["intensity"][i]
                spec1["mz"][j] = (mz1 + mz2) / 2
                merged_peak = True
                break
        if not merged_peak:
            spec2_red["mz"] += [spec2["mz"][i]]
            spec2_red["intensity"] += [spec2["intensity"][i]]
                
    spec1["mz"] += spec2_red["mz"]
    spec1["intensity"] += spec2_red["intensity"]
    
    return spec1


'''


def match_first_order_peaks(df, offset=0):
    DEPTH = 1
    MIN_PEAK_INT = 1
    peaks = []
    unique = []
    percentage = []
    for i in range(len(df.index)):
        if i%100 == 0: print(f'{i/len(df.index):.02f}', end="\r")
        d = df.iloc[i]
        c_peaks = 0
        c_unique = 0
        c_percentage = 0
        FT = FragmentationTree(d["MOL"])
        FT.build_fragmentation_tree_by_single_edge_breaks(d["MOL"], d.edges_idx, depth=DEPTH)

        for j in range(len(d.peaks["mz"])):
            #if d.peaks["intensity"][j] > MIN_PEAK_INT:
            _,n = find_matching_peaks(d.peaks["mz"][j], [f.tag + offset  for f in FT.get_all_fragments()[1:]])
            if n > 0:
                c_peaks += 1
                if n == 1:
                    c_unique += 1
        c_percentage = c_peaks / len(d.peaks["mz"]) 
        peaks.append(c_peaks)
        unique.append(c_unique)
        percentage.append(c_percentage)

    return peaks, unique, percentage
'''