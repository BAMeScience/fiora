import pandas as pd

def read(source, sep: str=" ", as_df=False):
    file = open(source, 'r')

    data = []
    data_piece = {}
    precursor = ""
    mz, intensity, annotation = [], [], []

    for line in file:
        if line == '\n':
            if energy2:
                data_piece['peaks40'] = {'mz': mz, 'intensity': intensity, 'annotation': annotation}
                mz, intensity, annotation = [], [], []
                energy2 = False
                continue
            else:
                continue
        if line.startswith("#PREDICTED"): continue
        if line.startswith("#In-silico"):
            precursor = line.split("ESI-MS/MS ")[1].split(" Spectra")[0]
            continue
        if line.strip().startswith("#ID="):
            energy2 = False
            data.append(data_piece)
            data_piece, mz, intensity, annotation = {}, [], [], []
            data_piece["Precursor_type"] = precursor
        if '=' in line:
            key = line.split('=')[0]
            value = "=".join(line.strip().split('=', 1)[1:])
            data_piece[key] = value
        elif line.strip() == "energy0":
            continue
        elif line.strip() == "energy1":
            data_piece['peaks10'] = {'mz': mz, 'intensity': intensity, 'annotation': annotation}
            mz, intensity, annotation = [], [], []
            # new data piece
        elif line.strip() == "energy2":
            energy2 = True
            data_piece['peaks20'] = {'mz': mz, 'intensity': intensity, 'annotation': annotation}
            mz, intensity, annotation = [], [], []
        else:
            line_split = line.split(sep)
            mz.append(float(line_split[0].strip()))
            intensity.append(float(line_split[1].strip()))

    data.append(data_piece)
    file.close()

    if as_df:
        return pd.DataFrame(data[1:])
    else:
        return data[1:]
    
    

