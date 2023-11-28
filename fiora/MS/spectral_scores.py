import numpy as np
from typing import Literal, Dict


from fiora.MOL.constants import DEFAULT_DALTON

def cosine(vec, vec_other):
    return np.dot(vec, vec_other) / (np.linalg.norm(vec) * np.linalg.norm(vec_other))

def cosine_bias_alt(vec, vec_other, cosine_precomputed=None):
    if not cosine_precomputed:
        cosine_precomputed = cosine(vec, vec_other)
    
    if cosine_precomputed <= 0.0:
        return 1.0
    bias = np.sqrt(np.square(np.dot(vec, vec_other)) / np.square(np.linalg.norm(vec) * np.linalg.norm(vec_other)))
    return bias / cosine_precomputed


def cosine_bias(vec, vec_other, cosine_precomputed=None):
    if not cosine_precomputed:
        cosine_precomputed = cosine(vec, vec_other)
    
    if cosine_precomputed <= 0.0:
        return 1.0
    vec = vec / np.linalg.norm(vec)
    vec_other = vec_other / np.linalg.norm(vec_other)

    bias = np.sqrt(np.dot(np.square(vec), np.square(vec_other))) # wrong: bias = np.sqrt(np.square(np.dot(vec, vec_other)) / (np.square(np.linalg.norm(vec)) * np.square(np.linalg.norm(vec_other))))
    return bias / cosine_precomputed

def create_mz_map(mz, mz_other, tolerance):
    mz_unique = np.sort(np.unique(mz + mz_other))
    bin_map = dict(zip(mz_unique, range(len(mz_unique))))
    
    for i, mz in enumerate(mz_unique[:-1]):
        if (mz_unique[i+1] - mz) < tolerance: #TODO this may lead to continuous mz matches that lie >tolerance apart on the edges
            bin_map[mz_unique[i+1]] = bin_map[mz] #TODO remove unmapped bins

    return bin_map

def spectral_cosine(spec, spec_ref, tolerance=DEFAULT_DALTON, transform=None, with_bias=False):
    mz_map = create_mz_map(spec["mz"], spec_ref["mz"], tolerance=tolerance)
    vec, vec_ref = np.zeros(len(mz_map)), np.zeros(len(mz_map)) 
    
    bins = list(map(mz_map.get, spec["mz"]))
    bins_ref = list(map(mz_map.get, spec_ref["mz"]))
    
    np.add.at(vec, bins, spec["intensity"]) #vec.put(bins, spec["intensity"])
    np.add.at(vec_ref, bins_ref, spec_ref["intensity"])   

    if transform:
        vec=transform(vec)
        vec_ref=transform(vec_ref)

    cos = cosine(vec, vec_ref)
    if with_bias:
        bias = cosine_bias(vec, vec_ref, cosine_precomputed=cos)
        return cos, bias
    return cos

def spectral_reflection_cosine(spec, spec_ref, tolerance=DEFAULT_DALTON, transform=None, with_bias=False):
    mz_map = create_mz_map(spec["mz"], spec_ref["mz"], tolerance=tolerance)
    vec, vec_ref = np.zeros(len(mz_map)), np.zeros(len(mz_map)) 
    
    bins = list(map(mz_map.get, spec["mz"]))
    bins_ref = list(map(mz_map.get, spec_ref["mz"]))
    
    np.add.at(vec, bins, spec["intensity"]) #vec.put(bins, spec["intensity"])
    np.add.at(vec_ref, bins_ref, spec_ref["intensity"])   

    #Reflection score: Remove values that are not matched with the reference values
    unmatched_bins = [b for b in bins if b not in bins_ref] 
    vec.put(unmatched_bins, 0.) 
    
    if transform:
        vec=transform(vec)
        vec_ref=transform(vec_ref)
    
    cos = cosine(vec, vec_ref)
    if with_bias:
        bias = cosine_bias(vec, vec_ref, cosine_precomputed=cos)
        return cos, bias
    return cos




def create_mz_map(mz, mz_other, tolerance):
    mz_unique = np.sort(np.unique(mz + mz_other))
    bin_map = dict(zip(mz_unique, range(len(mz_unique))))
    
    for i, mz in enumerate(mz_unique[:-1]):
        if (mz_unique[i+1] - mz) < tolerance: #TODO this may lead to continuous mz matches that lie >tolerance apart on the edges
            bin_map[mz_unique[i+1]] = bin_map[mz] #TODO remove unmapped bins

    return bin_map

#### TODO TEST THIS
def reweighted_dot(spec, spec_ref, int_pow=0.5, mz_pow=0.5, tolerance=DEFAULT_DALTON, with_bias=False):
    mz_map = create_mz_map(spec["mz"], spec_ref["mz"], tolerance=tolerance)
    vec, vec_ref = np.zeros(len(mz_map)), np.zeros(len(mz_map)) 
    
    bins = list(map(mz_map.get, spec["mz"]))
    bins_ref = list(map(mz_map.get, spec_ref["mz"]))
    
    spec["mz_int"] = [np.power(spec["intensity"][i], int_pow)*np.power(mz, mz_pow) for i, mz in enumerate(spec["mz"])]
    spec_ref["mz_int"] = [np.power(spec_ref["intensity"][i], int_pow)*np.power(mz, mz_pow) for i, mz in enumerate(spec_ref["mz"])]
    
    np.add.at(vec, bins, spec["mz_int"])
    np.add.at(vec_ref, bins_ref, spec_ref["mz_int"])


    cos = cosine(vec, vec_ref)
    if with_bias:
        bias = cosine_bias(vec, vec_ref, cosine_precomputed=cos)
        return cos, bias
    return cos