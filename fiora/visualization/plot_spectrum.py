import argparse
import numpy as np
from define_colors import *
import fiora.IO.mgfReader as mgfReader
import fiora.IO.mspReader as mspReader
import spectrum_visualizer as sv
from pyteomics import pylab_aux as pa, usi
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file1", help="file where spectrum is contained (.mgf or .msp)", type=str,
                    default="/home/ynowatzk/data/9MM/mgf/9MM_FASP.mgf")
parser.add_argument("-n", "--name1", help="exact name of spectrum", type=str,
                    required=True)

parser.add_argument("-f2", "--file2", help="file where lower spectrum is found",
                    type=str)

parser.add_argument("-n2", "--name2", help="exact name of lower spectrum",
                    type=str)
parser.add_argument("-o", "--out", help="output file",
                    type=str)
#parser.add_argument("-a", "--annotate", help="perform spectrum annotation", action="store_true", default=False)
#parser.add_argument("-p", "--peptide", help="peptide", type=str, default="None")
#parser.add_argument("-c", "--charge", help="charge", type=int, default=0)
parser.add_argument("--fontsize", help="font size of the text", type=int)
args = parser.parse_args()


# def annotate_spectrum(spectrum, peptide, charge=args.charge):
#     # spectrum = usi.proxi(
#     #    'mzspec:PXD004732:01650b_BC2-TUM_first_pool_53_01_01-3xHCD-1h-R2:scan:41840',
#     #    'massive')
#     # print(spectrum)
#     # peptide = 'WNQLQAFWGTGK'
#     pa.annotate_spectrum(spectrum, peptide, precursor_charge=charge, backend='spectrum_utils',
#                          ion_types='aby', title=peptide)
#     plt.show()


def read_spectrum_from_file(file, name):
    if file.endswith(".mgf"):
        return mgfReader.get_spectrum_by_name(file, name)
    elif file.endswith(".msp"):
        return mspReader.get_spectrum_by_name(file, name)
    else:
        print("UNKNOWN FILE EXTENSION:\n", file)
        exit(1)


#
# Plotting first spectrum
#

if args.fontsize:
    set_all_font_sizes(args.fontsize)

s1 = read_spectrum_from_file(args.file1, args.name1)

if args.file2 and args.name2:
    s2 = read_spectrum_from_file(args.file2, args.name2)
    sv.plot_spectrum(s1, s2, title=args.name1 + " matched by " + args.name2.split("/")[0], out=args.out) #,annotate=args.annotate, peptide=args.peptide, charge=args.charge, font_size=args.fontsize)
else:
    sv.plot_spectrum(s1, title=args.name1, out=args.out, show=True)#, font_size=args.fontsize)