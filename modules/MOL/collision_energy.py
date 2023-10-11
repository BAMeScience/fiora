charge_factor = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75}
nce_instruments = ["Orbitrap", "LC-ESI-QFT", "LC-APCI-ITFT", "Linear Ion Trap", "LC-ESI-ITFT"] # "Flow-injection QqQ/MS",

def NCE_to_eV(nce, precursor_mz, charge=1):
    return nce * precursor_mz / 500 * charge_factor[charge]

def align_CE(ce, precursor_mz, instrument=None):
    if type(ce) == float:
        if ce > 0.0 and ce < 1.0:
            return str(ce) # REMOVE
        if instrument in nce_instruments:
            return NCE_to_eV(ce, precursor_mz)
        return ce
    if "keV" in ce:
        ce = ce.replace("keV", "")
        return float(ce) * 1000
    if "eV" in ce:
        ce = ce.replace("eV", "")
        try:
            return float(ce)
        except:
            return ce
    elif "V" in ce:
        ce = ce.replace("V", "")
        try:
            return float(ce)
        except:
            return ce
    elif "ev" in ce:
        ce = ce.replace("ev", "")
        try:
            return float(ce)
        except:
            return ce
    elif "% (nominal)" in ce:
        try:
            nce = ce.split('% (nominal)')[0].strip().split(' ')[-1]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "(nominal)" in ce:
        try:
            nce = ce.split('(nominal)')[0].strip().split(' ')[-1]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "(NCE)" in ce:
        try:
            nce = ce.strip().split('(NCE)')[0]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "HCD" in ce:
        try:
            nce = ce.strip().split('HCD')[0]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    elif "%" in ce:
        try:
            nce = ce.split('%')[0].strip().split(' ')[-1]
            nce = float(nce)
            return NCE_to_eV(nce, precursor_mz)
        except:
            return ce
    else:
        try: 
            ce = float(ce)
            if ce > 0.0 and ce < 1.0:
                return str(ce) # REMOVE
            if instrument in nce_instruments:
                return NCE_to_eV(ce, precursor_mz)
            return ce
        except:
            return ce