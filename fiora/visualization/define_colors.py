import matplotlib
import matplotlib.colors
import seaborn as sns
from matplotlib import pyplot as plt


def mix_colors(c1, c2, ratio=1.0):
    z = zip(c1, c2)
    return [((x * ratio) + y) / (ratio + 1) for (x, y) in z]


col_mistle = sns.color_palette("Set3")[0]
col_mistle_dark = (0.5 * col_mistle[0], 0.9 * col_mistle[1], 0.9 * col_mistle[2])
col_mistle_bright = (0.5 * col_mistle[0], 0.9 * col_mistle[1], 0.9 * col_mistle[2])

col_decoy = sns.color_palette("Set3")[1]
col_spectrast = sns.color_palette(palette="Set3")[4]  # (1,1,1) #sns.color_palette(palette="Set3")[4]
# col_spectrast = (1,1,1) #sns.color_palette(palette="Set3")[4]
col_xtandem = sns.color_palette(palette="Set3")[3]
col_msf = sns.color_palette(palette="Set3")[3]
col_olivegrey = (0.65, 0.8, 0.6)
col_st_line = "slateblue"

col_i1 = "purple"
col_i1_dot = "violet"

palette = sns.color_palette("colorblind")
color_palette = palette

C = {"green": palette[2], "orange": palette[1], "blue": palette[0], "red": palette[3], "yellow": palette[8]}
C["g"] = C["green"]
C["o"] = C["orange"]
C["b"] = C["blue"]
C["r"] = C["red"]
C["lightgreen"] = [1.25 * x for x in C["green"]]
C["darkgreen"] = [0.75 * x for x in C["green"]]
C["ivorygreen"] = mix_colors(C["green"], matplotlib.colors.to_rgb("ivory"), ratio=0.5)
C["chocolategreen"] = mix_colors(C["green"], matplotlib.colors.to_rgb("chocolate"), ratio=1.5)


PRINT_COL = {
    "black": "\033[98m",
    "blue": "\033[94m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "end": "\033[00m" 
}

lightblue = (46, 64, 85)
lightblue_hex = "#75a3d9"
lightpink = (81, 55, 52)
lightpink_hex = "#cf8c85"
newpink = (255, 64, 85)  # (light blue + Red 255)
newpink_hex = "#ffa3d6" 

newnewpink = (242, 163, 214)  # (light blue + Red 255)
newnewpink_hex = "#F2A3D6" 

wippinkbutbestsofar = (221, 140, 150)
wippinkbutbestsofar_hex = "#DD8C96"

evenbetterpink = (222, 140, 171)
evenbetterpink_hex = "#DE8CAB"

maybeevenbetterpink = (222, 148, 172)
maybeevenbetterpink_hex = "#DE94AC"

abitsofterclassic_hex = (226, 154, 181)
abitsofterclassic_hex = "#E29AB5"

black_hex = "#000000"
lightgreen_hex = "#ACF39D"
wine_hex = "#773344"

bluepink = sns.color_palette([lightblue_hex, lightpink_hex, black_hex, lightgreen_hex, wine_hex], as_cmap=True)
bluepink_grad = sns.diverging_palette(17.7, 245.8, s=75, l=50, sep=1, n=6, center='light', as_cmap=True)
bluepink_grad8 = sns.diverging_palette(17.7, 245.8, s=75, l=50, sep=1, n=8, center='light', as_cmap=False)

tri_palette=["gray", bluepink[0], bluepink[1]]


#
# Definition for Seaborn plots
#


def define_figure_style(style: str, palette_steps=8):
    # Define figure styles
    if "magma-white":
        color_palette = sns.color_palette("magma_r", palette_steps)
        sns.set_theme(style="whitegrid",
                        rc={'axes.edgecolor': 'black', 'ytick.left': True, 'xtick.bottom': True, 'xtick.color': 'black',
                            "axes.spines.bottom": True, "axes.spines.right": True, "axes.spines.top": True,
                            "axes.spines.left": True})
    return color_palette

def set_theme():
    sns.set_theme(style="darkgrid",
                  rc={'axes.edgecolor': 'black', 'ytick.left': True, 'xtick.bottom': True, 'xtick.color': 'black',
                      "axes.spines.bottom": True, "axes.spines.right": False, "axes.spines.top": False,
                      "axes.spines.left": True})


def set_light_theme():
    sns.set_theme(style="whitegrid",
                  rc={'axes.edgecolor': 'black', 'ytick.left': True, 'xtick.bottom': True, 'xtick.color': 'black',
                      "axes.spines.bottom": True, "axes.spines.right": True, "axes.spines.top": True,
                      "axes.spines.left": True})


def reset_matplotlib():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def set_all_font_sizes(size):
    zs = ['font.size', 'axes.labelsize', 'axes.titlesize', 'legend.fontsize', "xtick.labelsize", "xtick.major.size",
          "xtick.minor.size", "ytick.labelsize", "ytick.major.size", "ytick.minor.size"]

    for z in zs:
        plt.rcParams[z] = size


def set_plt_params_to_default():
    plt.rcParams.update(plt.rcParamsDefault)
