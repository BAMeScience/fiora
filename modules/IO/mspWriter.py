import pandas as pd


def write_msp(df, path, write_header=True, headers=["Name", "Precursor_type", "Spectrum_type", "PRECURSORMZ", "RETENTIONTIME", "Charge", "Comments",
                                 "Num peaks"]):
    with open(path, "w") as outfile:
        for x in df.index:
            peaks = df.loc[x].peaks
            if write_header:
                for key in headers:
                    outfile.write(key + ": " + str(df.loc[x][key]) + "\n")
            d = df.loc[x]
            for i in range(len(peaks['mz'])):
                outfile.write(str(peaks['mz'][i]) + "\t" + str(peaks['intensity'][i]) + "\n")
