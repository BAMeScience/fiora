import pandas as pd



def write_mgf(df, path, peak_tag="peaks", write_header=True, headers=["TITLE", "RTINSECONDS", "PEPMASS", "CHARGE"], header_map={}, annotation=False):
    for h in headers:
        if h not in header_map.keys():
            header_map[h] = h
    with open(path, "w") as outfile:
        for x in df.index:
            outfile.write("BEGIN IONS\n")
            peaks = df.loc[x][peak_tag]
            if write_header:
                for key in headers:
                    outfile.write(key + "=" + str(df.loc[x][header_map[key]]) + "\n")
            for i in range(len(peaks['mz'])):
                line = str(peaks['mz'][i]) + " " + str(peaks['intensity'][i])
                if annotation:
                    line += " " + peaks['annotation'][i]
                outfile.write(line + "\n")
            outfile.write("END IONS\n")
