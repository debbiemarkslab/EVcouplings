"""
Distance calculations in PDB 3D structures

Authors:
  Thomas A. Hopf
"""

import numpy as np
import pandas as pd
from numba import jit


def res_name_to_tuple(name):
    """
    Transform PDB residue index into tuple of
    position and insertion code

    Parameters
    ----------
    name : str
        Residue name (e.g. "188" or "189A")
    Returns
    -------
    tuple(int, str)
        Residue name split into position index
        and insertion code (empty string if residue
        has no insertion code)
    """
    if len(name) < 1:
        raise ValueError(
            "Residue name must at least be one character long."
        )

    if name[-1].isdigit():
        return (int(name), "")
    else:
        return (int(name[:-1]), name[-1])
