"""
Functions for structure prediction using CNSsolve 1.3

Authors:
  Thomas A. Hopf
"""

from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.constants import AA1_to_AA3


def cns_seq_file(sequence, output_file=None, residues_per_line=16):
    """
    Generate a CNS .seq file for a given protein sequence

    Parameters
    ----------
    sequence : str
        Amino acid sequence in one-letter code
    output_file : str, optional (default: None)
        Save 3-letter code sequence to this file
        (if None, will create temporary file)
    residues_per_line : int, optional (default: 16)
        Print this many residues on each line
        of .seq file

    Returns
    -------
    output_file : str
        Path to file with sequence
        (useful if temporary file was
        generated)

    Raises
    ------
    InvalidParameterError
        If sequence contains invalid symbol
    """
    if output_file is None:
        output_file = temp()

    with open(output_file, "w") as f:
        # split sequence into parts per line
        lines = [
            sequence[i: i + residues_per_line]
            for i in range(0, len(sequence), residues_per_line)
        ]

        # go through lines and transform into 3-letter code
        for line in lines:
            try:
                l3 = " ".join(
                    [AA1_to_AA3[aa] for aa in line]
                )
            except KeyError as e:
                raise InvalidParameterError(
                    "Invalid amino acid could not be mapped"
                ) from e

            f.write(l3 + "\n")

    return output_file
