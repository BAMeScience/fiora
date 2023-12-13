#TODO Check if first spectrum is read
import pandas as pd

def read(source, sep: str=" ", as_df=False, debug=False):
    file = open(source, 'r')
    in_begin_ions = False
    data = []
    data_piece = {}
    mz, intensity, ion = [], [], []

    for line in file:
        if debug: print(line.strip())
        if line == "MASS=Monoisotopic\n": continue #TODO edge case hacky solution
        if line == '\n': continue
        if line.startswith("#"): continue
        if line.startswith("NA#"): continue
        if line.strip() == "END IONS":
            in_begin_ions = False
            data_piece['peaks'] = {'mz': mz, 'intensity': intensity, 'annotation': ion}
            data.append(data_piece)
            continue
        
        if line.strip() == "BEGIN IONS" or line.strip() == "BEGIN IONS:":
            in_begin_ions = True
            data_piece, mz, intensity, ion = {}, [], [], []
            continue

        if '=' in line:
            key = line.split('=')[0]
            value = "=".join(line.strip().split('=', 1)[1:]) #line.split('=', 1)[1].strip()
            data_piece[key] = value
        else:
            line_split = line.split(sep)
            mz.append(float(line_split[0].strip()))
            intensity.append(float(line_split[1].strip()))

    if in_begin_ions:
        data.append(data_piece)
    file.close()

    if as_df:
        return pd.DataFrame(data)
    else:
        return data


def get_spectrum_by_name(source, name):
    file = open(source, 'r')

    line_match = "TITLE=" + name + "\n"
    data_piece = {}
    mz, intensity, ion = [], [], []
    found = False

    for line in file:
        if line == "END IONS\n" and found:
            data_piece['peaks'] = {'mz': mz, 'intensity': intensity, 'ion': ion}
            break
        if line == line_match:  # exact name match
            found = True

        if not found:
            continue  # skip ahead

        if '=' in line:
            key = line.split('=')[0]
            value = line.split('=', 1)[1].strip()
            data_piece[key] = value
        else:
            line_split = line.split(' ')
            mz.append(line_split[0].strip())
            intensity.append(line_split[1].strip())
    file.close()

    return data_piece


'''
Thoughts on format

Every Spectrum becomes a dictionary 

standard = {
    (Id: Number by occurence)
    Name: its name (keylike feature)
    Peptide: --- Extract from Name
    Charge: --- Extract from Name
    OtherAttributes:
    Peaks: pd.DataFrame(['mz', 'intensity', 'ions'])
}

minimal = {Name, pd.DF mz vs intensity}

sparse = (Name, sparse_vector)


'''
