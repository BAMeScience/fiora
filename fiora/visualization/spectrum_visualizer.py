import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
import seaborn as sns
from pyteomics import pylab_aux as pa, usi
import pandas as pd
import spectrum_utils.fragment_annotation as fa
from fiora.visualization.define_colors import *
from typing import Dict, List


# From spectrum utils issue https://github.com/bittremieux/spectrum_utils/issues/56
# Overwrite get_theoretical_fragments
def get_theoretical_fragments(proteoform, ion_types=None, max_ion_charge=None, neutral_losses=None):
    fragments_masses = []
    for mod in proteoform.modifications:
        fragment = fa.FragmentAnnotation(ion_type="w", charge=1)
        mass = mod.source[0].mass
        fragments_masses.append((fragment, mass))
    return fragments_masses

def set_custom_annotation():
    # Use the custom function to annotate the fragments
    fa.get_theoretical_fragments = get_theoretical_fragments
    fa._supported_ions += "w"    
    # Set peak color for custom ion
    sup.colors["w"] = lightblue_hex

def set_default_peak_color(color):
    sup.colors[None] = color

def annotate_and_plot(spectrum, mz_fragments, with_grid: bool=False, ppm_tolerance: int=100, ax=None):
    
    set_custom_annotation()

    # Instantiate Spectrum and annotate with proforma string format (e.g. X[+9.99] )
    spectrum = sus.MsmsSpectrum("None", 0, 1, spectrum['peaks']['mz'], spectrum['peaks']['intensity'])
    x_string = "".join([f"X[+{mz}]" for mz in sorted(mz_fragments)])
    spectrum.annotate_proforma(x_string, ppm_tolerance, "ppm")
    
    
    # Find ax and plot
    if not ax:
        ax = plt.gca()
    sup.spectrum(spectrum, grid=with_grid, ax=ax)
    if with_grid:
        ax.set_ylim(0, 1.075)
    else:
        sns.despine(ax=ax)

    return ax

def plot_spectrum(spectrum: Dict, second_spectrum: Dict|None=None, highlight_matches: bool=False, mz_matches: List[int]=[], facet_plot=False, ppm_tolerance: int=100, charge=0, title=None, out=None, with_grid=False, ax=None, show=False, color=None):
    top_spectrum = sus.MsmsSpectrum("None", 0, charge, spectrum['peaks']['mz'], spectrum['peaks']['intensity'])
    if color:
        set_default_peak_color(color)
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    # spectrum.set_mz_range(min_mz=0, max_mz=2000)
    if second_spectrum is not None:
        bottom_spectrum = sus.MsmsSpectrum("None", 0, charge, second_spectrum['peaks']['mz'], second_spectrum['peaks']['intensity'])
        if highlight_matches:
            set_custom_annotation()

            x_string = "".join([f"X[+{mz}]" for mz in sorted(second_spectrum['peaks']['mz'])])
            top_spectrum.annotate_proforma(x_string, ppm_tolerance, "ppm")
            x_string = "".join([f"X[+{mz}]" for mz in sorted(spectrum['peaks']['mz'])])
            bottom_spectrum.annotate_proforma(x_string, ppm_tolerance, "ppm")
        
        if facet_plot:
            sup.facet(spec_top=top_spectrum, spec_mass_errors=top_spectrum, spec_bottom=bottom_spectrum, mass_errors_kws={"plot_unknown": False})
        else: # mirror plot
            sup.mirror(spec_top=top_spectrum, spec_bottom=bottom_spectrum, ax=ax, spectrum_kws={"grid": with_grid})
        
        if with_grid:
            ax.set_ylim(-1.075, 1.075)
        else:
            sns.despine(ax=ax)
            if second_spectrum is not None: 
                ax.spines['bottom'].set_position(('outward', 10))

    # Single spectrum
    else:
        
        if highlight_matches and mz_matches:
            set_custom_annotation()

            x_string = "".join([f"X[+{mz}]" for mz in sorted(mz_matches)])
            top_spectrum.annotate_proforma(x_string, ppm_tolerance, "ppm")
        
        sup.spectrum(top_spectrum, grid=with_grid, ax=ax)
        if with_grid:
            ax.set_ylim(0, 1.1)
        else:
            sns.despine(ax=ax)

    plt.title(title)
    if out is not None:
       plt.savefig(out)
    else:
        if show:
            plt.show()
        else:
            return ax

def plot_vector_spectrum(vec1, vec2, ax=None, title=None, y_label="probability", names= None):
    v1 = pd.DataFrame({"range": names if names else range(len(vec1)), "prob": vec1, "group": "prob"})
    v2 = pd.DataFrame({"range": names if names else range(len(vec2)), "prob": vec2, "group": "pred"})
    V = pd.concat([v1, v2])

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 6.4))
    sns.barplot(ax=ax, data=V, y="prob", x="range",  edgecolor="black", hue="group", linewidth=1.5)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return ax


