#from pyteomics import mzml
#import regex as re


# import spectrum_utils.spectrum as sus


def read(source, sep=' '):
    file = open(source, 'r')

    data = []
    data_piece = {}
    mz, intensity, ion = [], [], []

    for line in file:
        if 'Name:' == line[0:5] or 'NAME:' == line[0:5]:
            data_piece['peaks'] = {'mz': mz, 'intensity': intensity, 'annotation': ion}
            data.append(data_piece)

            data_piece, mz, intensity, ion = {}, [], [], []

        if ':' in line:
            key = line.split(':')[0]
            value = line.split(':', 1)[1].strip()
            data_piece[key] = value
        else:
            if line == "\n":
                continue
            ls = line.strip()
            line_split = ls.split(sep)
            mz.append(float(line_split[0]))
            intensity.append(float(line_split[1]))
            #ion.append(line_split[2].strip())

    data_piece['peaks'] = {'mz': mz, 'intensity': intensity, 'annotation': ion}
    data.append(data_piece)
    file.close()

    return data[1:]


def read_minimal(source):
    file = open(source, 'r')

    data = []
    data_piece = {}
    mz, intensity, ion = [], [], []

    for line in file:
        if 'Name:' == line[0:5]:
            data_piece['peaks'] = {'mz': mz, 'intensity': intensity}
            data.append(data_piece)
            data_piece = {'Name': line.split(':', 1)[1].strip()}
            mz, intensity = [], []
            continue
        if not (':' in line):
            line_split = line.split('\t')
            mz.append(line_split[0])
            intensity.append(line_split[1])

    data.append(data_piece)
    file.close()

    return data[1:]


def read_peptides(source):
    file = open(source, 'r')

    pep_list = []
    for line in file:
        if 'Name:' == line[0:5]:
            l = line.strip('\n')[5:]
            l = re.sub(r'[\d+ /]', '', l)
            pep_list.append(l)

    file.close()
    return pep_list


def read_sparse(source):
    file = open(source, 'r')

    file.close()


def readOld(source):
    file = open(source, 'r')
    c = 0
    data = []
    active_lines = []
    for line in file:
        if 'Name:' == line[0:5]:
            data.append(make_data_piece(active_lines))
            active_lines = []
        active_lines.append(line.strip('\n'))
    data.append(make_data_piece(active_lines))
    file.close()

    return data[1:]


def make_data_piece(lines):
    data_piece = {}
    mz, intensity, ion = [], [], []

    for line in lines:
        if ':' in line:
            key = line.split(':')[0]
            value = ':'.join(line.split(':')[1:])
            data_piece[key] = value
        else:
            line_split = line.split('\t')
            mz.append(line_split[0])
            intensity.append(line_split[1])
            ion.append(line_split[2])

        data_piece['peaks'] = {'mz': mz, 'intensity': intensity, 'ion': ion}
    return data_piece


def get_spectrum_by_name(source, name):
    file = open(source, 'r')

    line_match = "Name: " + name + "\n"
    data_piece = {}
    mz, intensity, ion = [], [], []
    found = False

    for line in file:
        if line[0:5] == "Name:" and found:
            data_piece['peaks'] = {'mz': mz, 'intensity': intensity, 'ion': ion}
            break
        if line == line_match: #exact name match
            found = True
        if not found: continue
        if ':' in line:
            key = line.split(':')[0]
            value = line.split(':', 1)[1].strip()
            data_piece[key] = value
        else:
            line_split = line.split('\t')
            mz.append(line_split[0])
            intensity.append(line_split[1])
            ion.append(line_split[2].strip())

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
