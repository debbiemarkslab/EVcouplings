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
    H = -np.sum(X_ * np.log2(X_))

    if normalize:
        return 1 - (H / np.log2(len(X)))
    else:
        return H


def entropy_vector(model, normalize=True):
    """
    Compute vector of positional entropies for
    single-site frequencies in a CouplingsModel

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
    np.array
        Vector of length model.L containing
        entropy for each position
    """
    cons = np.apply_along_axis(
        lambda x: entropy(x, normalize=normalize),
        axis=1, arr=model.fi()
    )

    return cons


def entropy_map(model, normalize=True):
    """
    Compute dictionary of positional entropies for
    single-site frequencies in a CouplingsModel

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
    cons = entropy_vector(model, normalize)

    return dict(
        zip(model.index_list, cons)
    )


def dihedral_angle(p0, p1, p2, p3):
    """
    Compute dihedral angle given four points
    
    Adapted from the following source:
    http://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    (answer by user Praxeolitic)
    
    Parameters
    ----------
    p0 : np.array
        Coordinates of first point
    p1 : np.array
        Coordinates of second point
    p2 : np.array
        Coordinates of third point
    p3 : np.array
        Coordinates of fourth point

    Returns
    -------
    numpy.float
        Dihedral angle (in radians)
    """
    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.arctan2(y, x)


def median_absolute_deviation(x, scale=1.4826):
    """
    Compute median absolute deviation of a set of numbers
    (median of deviations from median)

    Parameters
    ----------
    x : list-like of float
        Numbers for which median absolute deviation
        will be computed
    scale : float, optional (default: 1.4826)
        Rescale median absolute deviation by this factor;
        default value is such that median absolute
        deviation will match regular standard deviation
        of Gaussian distribution
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return scale * mad
