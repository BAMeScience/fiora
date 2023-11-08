import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
import seaborn as sns
from pyteomics import pylab_aux as pa, usi
import pandas as pd

def plot_spectrum_obsolete(spectrum, second_spectrum=None, title=None, out=None):
    spectrum = sus.MsmsSpectrum("None", 0, 0,
                                spectrum['peaks']['mz'],
                                spectrum['peaks']['intensity'])

    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum.set_mz_range(min_mz=0, max_mz=2000)
    if second_spectrum is not None:
        bottom_spectrum = sus.MsmsSpectrum(second_spectrum["Name"], float(second_spectrum['MW']), int(second_spectrum["Name"].split('/')[-1]),
                                           second_spectrum['peaks']['mz'],
                                           second_spectrum['peaks']['intensity'],
                                           peptide=second_spectrum["Name"].split('/')[0])
        bottom_spectrum.set_mz_range(min_mz=0, max_mz=2000)
        sup.mirror(spec_top=spectrum,
                   spec_bottom=bottom_spectrum, #.annotate_peptide_fragments(0.5, 'Da', ion_types='aby'),
                   ax=ax)
    else:
        sup.spectrum(spectrum, ax=ax)

    plt.title(title)
    if out is not None:
       plt.savefig(out)
    else:
        plt.show()

def annotate_spectrum(spectrum, peptide):
    spectrum = usi.proxi(
    'mzspec:PXD004732:01650b_BC2-TUM_first_pool_53_01_01-3xHCD-1h-R2:scan:41840',
    'massive')
    print(spectrum)
    peptide = 'WNQLQAFWGTGK'
    pa.annotate_spectrum(spectrum, peptide, precursor_charge=2, backend='spectrum_utils',
                         ion_types='aby', title=peptide)
    plt.show()

def plot_spectrum(spectrum, second_spectrum=None, annotate=False, peptide="None", charge=0, title=None, out=None, ax=None, show=False):
    #sus.static_modification('C', 57.02146)
    #modifications = {i: 57.02200 for i, AS in enumerate(peptide) if AS=="C"}
    # modifications = {6: 57.02200}

    spectrum = sus.MsmsSpectrum("None", 0, charge,
                                spectrum['peaks']['mz'],
                                spectrum['peaks']['intensity'])
    if annotate:
        spectrum = spectrum.annotate_peptide_fragments(0.2, 'Da', ion_types='by')
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 6))
    spectrum.set_mz_range(min_mz=0, max_mz=2000)
    if second_spectrum is not None:
        bottom_spectrum = sus.MsmsSpectrum("None", 0, charge, #int(second_spectrum["Name"].split('/')[-1]),
                                           second_spectrum['peaks']['mz'],
                                           second_spectrum['peaks']['intensity'],
                                           )
        if annotate:
            bottom_spectrum = bottom_spectrum.annotate_peptide_fragments(0.2, 'Da', ion_types='by')
        bottom_spectrum.set_mz_range(min_mz=0, max_mz=2000)
        sup.mirror(spec_top=spectrum,
                   spec_bottom=bottom_spectrum, #.annotate_peptide_fragments(0.5, 'Da', ion_types='aby'),
                   ax=ax)
        ax.set_ylim(-1.075, 1.075)

    else:
        sup.spectrum(spectrum, ax=ax)
        ax.set_ylim(0, 1.1)

    plt.title(title)
    if out is not None:
       plt.savefig(out)
    else:
        if show:
            plt.show()
        else:
            return ax


def annotate_and_plot(spectrum, mz_fragments, ppm_tolerance=None, tolerance=0.1, ax=None):
    if ppm_tolerance:
        tolerances = [mz * ppm_tolerance for mz in mz_fragments]
    else:
        tolerances = [tolerance] * len(mz_fragments)
        
    spectrum = sus.MsmsSpectrum("None",
                                0, 
                                1,
                                spectrum['peaks']['mz'],
                                spectrum['peaks']['intensity'])
    
    annotate_mz_fragments(spectrum=spectrum, mz_fragments=mz_fragments, tolerances=tolerances)
    
    if not ax:
        ax = plt.gca()
    sup.spectrum(spectrum, ax=ax)
    return ax

    
    

def annotate_mz_fragments(spectrum, mz_fragments, tolerances, tag: str = "x"):
    for mz, tol in zip(mz_fragments, tolerances):
        spectrum = spectrum.annotate_mz_fragment(fragment_mz=mz, fragment_charge=1, fragment_tol_mass=tol, fragment_tol_mode='Da', text=tag)
    return spectrum


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
