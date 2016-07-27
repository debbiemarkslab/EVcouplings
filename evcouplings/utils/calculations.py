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
