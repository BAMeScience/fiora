import pandas as pd



# from https://github.com/hcji/PyCFMID/blob/master/PyCFMID/PyCFMID.py
def parser_fraggraph_gen(output_file):
    with open(output_file) as t:
        output = t.readlines()
    output = [s.replace('\n', '') for s in output]
    nfrags = int(output[0])
    frag_index = [int(output[i].split(' ')[0]) for i in range(1, nfrags+1)] 
    frag_mass = [float(output[i].split(' ')[1]) for i in range(1, nfrags+1)] 
    frag_smiles = [output[i].split(' ')[2] for i in range(1, nfrags+1)]
    loss_from = [int(output[i].split(' ')[0]) for i in range(nfrags+2, len(output))] 
    loss_to = [int(output[i].split(' ')[1]) for i in range(nfrags+2, len(output))]
    loss_smiles = [output[i].split(' ')[2] for i in range(nfrags+2, len(output))]
    fragments = pd.DataFrame({'index': frag_index, 'mass': frag_mass, 'smiles': frag_smiles})
    losses = pd.DataFrame({'from': loss_from, 'to': loss_to, 'smiles': loss_smiles})
    return {'fragments': fragments, 'losses': losses}