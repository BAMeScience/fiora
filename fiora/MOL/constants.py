ADDUCT_WEIGHTS = {
    "[M+H]+": 1.007276,
    "[M+NH4]+": 18.033823,
    "[M+Na]+": 22.989218 ,
    "[M-H]-": -1.007276,
    
    #
    # positvie fragment rearrangements
    #
    "[M-H]+": -1.007276, # Double bond replacing 2 hydrogen atoms + H
    "[M-3H]+": -1.007276 * 3, # 2 Double bonds  + H
    # experimental cases
    "[M]+": 0,
    "[M-2H]+": -1.007276 * 2,
    "[M-4H]+": -1.007276 * 4,
    "[M-5H]+": -1.007276 * 5,
    
    
    #
    # negative fragment rearrangements
    # 
    
    "[M]-": 0, # can this even exist?
    "[M-2H]-": -1.007276 * 2,
    "[M-3H]-": -1.007276 * 3,
    "[M-4H]-": -1.007276 * 4
    
    }


PPM = 1/1000000
DEFAULT_PPM = 100 * PPM
DEFAULT_DALTON = 0.05 # equiv: 100ppm of 500 m/z 
MIN_ABS_TOLERANCE = 0.02 # Tolerance aplied for small fragment when relative PPM gets too small
#DEFAULT_MODES = ["[M+H]+", "[M-H]+", "[M-3H]+"]
DEFAULT_MODES = ["[M+H]+", "[M]+", "[M-H]+", "[M-2H]+", "[M-3H]+"] #"[M-4H]+"] #, "[M-5H]+"]
#NEGATIVE_MODES = ["[M]-", "[M-H]-", "[M-2H]-", "[M-3H]-", "[M-4H]-"]


# source: https://fiehnlab.ucdavis.edu/staff/kind/metabolomics/ms-adduct-calculator
