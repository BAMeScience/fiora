import mgfReader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("-i", "--infile",
                    help="path/file.mgf to the search file, which will be inspected",
                    type=str, required=True)

args = parser.parse_args()


# color_palette = "Set3"
color_palette = sns.color_palette("magma_r", 15)

df = mgfReader.read(args.infile)

print("Total number of spectra: %s" % df.shape[0])


# CHARGE
df['CHARGE'] = df['CHARGE'].apply(str)
df['CHARGE'] = pd.Categorical(df['CHARGE'], sorted(df.CHARGE.unique()))

ax = sns.countplot(data=df, x="CHARGE", palette=color_palette, edgecolor="black") #order=df['CHARGE'].value_counts().iloc[:10].index)
plt.title("MS/MS charge distribution")
plt.show()


# Precursor m/z

df['precursor_mz'] = df['PEPMASS'].apply(lambda x: float(x.split(' ')[0]))

sns.boxplot(data=df, y="precursor_mz", x="CHARGE", palette=color_palette)
plt.title("MS/MS precursor mz range over charge")
plt.show()


# Num of peaks
df['num_peaks'] = df['peaks'].apply(lambda p: len(p['mz']))

sns.boxplot(data=df, y="num_peaks", x="CHARGE", palette=color_palette)
plt.title("MS/MS number of peaks per spectrum over charge")
plt.show()