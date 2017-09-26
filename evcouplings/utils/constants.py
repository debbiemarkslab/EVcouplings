"""
Useful values and constants for all of package

Authors:
  Thomas A. Hopf
"""

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
