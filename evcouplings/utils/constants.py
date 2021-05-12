"""
Useful values and constants for all of package

Authors:
  Thomas A. Hopf
  Anna G. Green
"""
import numpy as np

# amino acid one-letter code to three-letter code
AA1_to_AA3 = {
    "A": "ALA",
    "B": "ASX",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "X": "XAA",
    "Y": "TYR",
    "Z": "GLX",
}

# amino acid three-letter code to one-letter code
AA3_to_AA1 = {
    v: k for k, v in AA1_to_AA3.items()
}

# Amino acid surface area values from Tien et al, 2013 (empirical values)
AA_SURFACE_AREA = {
    "A": 121,
    "R": 265,
    "D": 187,
    "N": 187,
    "C": 148,
    "E": 214,
    "Q": 214,
    "G": 97,
    "H": 216,
    "I": 195,
    "L": 191,
    "K": 230,
    "M": 103,
    "F": 228,
    "P": 154,
    "S": 143,
    "T": 163,
    "W": 264,
    "Y": 255,
    "V": 165,
    "X": np.nan
}

#Hydropathy index from Lehninger Principles of Biochemistry, 5th Edition, table 3-1
HYDROPATHY_INDEX = {
    "G": -.4,
    "A": 1.8,
    "P": -1.6,
    "V": 4.2,
    "L": 3.8,
    "I": 4.5,
    "M": 1.9,
    "F": 2.8,
    "Y": -1.3,
    "W": -0.9,
    "S": -0.8,
    "T": -0.7,
    "C": 2.5,
    "N": -3.5,
    "Q": -3.5,
    "K": -3.9,
    "H": -3.2,
    "R": -4.5,
    "D": -3.5,
    "E": -3.5,
    "-": 0
}