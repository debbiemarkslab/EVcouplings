"""
General calculation functions.

Authors:
  Thomas A. Hopf
"""

import numpy as np


def entropy(X, normalize=False):
    """
    Calculate entropy of distribution

    Parameters
    ----------
    X : np.array
        Vector for which entropy will be calculated
    normalize:
        Rescale entropy to range from 0 ("variable", "flat")
        to 1 ("conserved")

    Returns
    -------
    float
        Entropy of X
    """
    X_ = X[X > 0]
    H = -np.sum(X_ * np.log(X_))

    if normalize:
        return 1 - (H / np.log(len(X)))
    else:
        return H


def entropy_map(model, normalize=True):
    """
    Parameters
    ----------
    model : CouplingsModel
        Model for which entropy of sequence
        alignment will be computed (based on
        single-site frequencies f_i(A_i)
        contained in model)
    normalize : bool, default: True
        Normalize entropy to range 0 (variable)
        to 1 (conserved) instead of raw values

    Returns
    -------
    dict
        Map from positions in sequence (int) to
        entropy of column (float) in alignment
    """
    cons = np.apply_along_axis(
        lambda x: entropy(x, normalize=normalize),
        axis=1, arr=model.fi()
    )

    return dict(
        zip(model.index_list, cons)
    )
