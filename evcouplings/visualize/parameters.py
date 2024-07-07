"""
Visualization of pair model parameters

Authors:
  Thomas A. Hopf
"""

import json
import numpy as np
from evcouplings.couplings.pairs import add_mixture_probability


def evzoom_data(model, ec_threshold=0.9, freq_threshold=0.01,
                Jij_threshold=10, score="cn",
                reorder=None):
    """
    Generate data for EVzoom visualization. Use evzoom_json()
    to get final JSON string to use with EVzoom.

    Parameters
    ----------
    model : evcouplings.couplings.model.CouplingsModel
        Parameters of pairwise graphical model
    ec_threshold : float or int, optional (default: 0.9)
        Only display evolutionary couplings above this
        threshold. If float between 0 and 1, this will be
        interpreted as probability cutoff for mixture model.
        Otherwise, will be interpreted as absolute number of couplings.
    freq_threshold : float, optional (default: 0.01)
        Only display coupling parameters for amino acids with
        at least this frequency in the underlying sequence
        alignment
    Jij_threshold : int or float, optional (default: 10)
        Only display coupling parameters above this
        threshold. If float, this number will be interpreted
        as an actual score threshold; if int, this will
        be interpreted as a percentage of the maximum
        absolute score.
    score : str, optional (default: "cn")
        Use this score to determine which couplings to display.
        Valid choices are the score columns contained in the
        CouplingsModel.ecs dataframe
    reorder : str, optional (default: "KRHEDNQTSCGAVLIMPYFW")
        Order of amino acids in displayed coupling matrices

    Returns
    -------
    map_ : dict
        Map containing sequence indices and characters
    logo : list
        List containing information about sequence logos
        for axes of visualization
    matrix : dict
        List containing couplings that will be visualized
    """
    DIGITS = 2
    DIGITS_LOGO = 2
    ecs = model.ecs

    if 0 < ec_threshold <= 1.0:
        ecs = add_mixture_probability(ecs, score=score)
        ecs_sel = ecs.loc[ecs.probability >= ec_threshold]
    else:
        ecs_sel = ecs.iloc[:int(ec_threshold)]

    # if cutoff for couplings is given as int, interpret
    # as percentage of biggest absolute coupling value
    if isinstance(Jij_threshold, int):
        max_val = np.max(np.abs(model.Jij()))
        Jij_threshold = max_val * Jij_threshold / 100

    if reorder is not None:
        alphabet = np.array(list(reorder))
        alphabet_order = [
            model.alphabet_map[c] for c in reorder
        ]
    else:
        alphabet = model.alphabet
        alphabet_order = sorted(
            model.alphabet_map.values()
        )

    # Map containing sequence and indeces
    map_ = {
        "letters": "".join(model.seq()),
        "indices": list(map(int, model.sn())),
    }

    # assemble coupling matrix
    matrix = []

    for idx, r in ecs_sel.iterrows():
        i, j, score_ij = r["i"], r["j"], r[score]
        Jij = model.Jij(i, j)[alphabet_order, :][:, alphabet_order]
        ai_set = np.where(
            np.max(np.abs(Jij), axis=1) > Jij_threshold
        )[0]
        aj_set = np.where(
            np.max(np.abs(Jij), axis=0) > Jij_threshold
        )[0]

        cur_matrix = [
            [round(Jij[ai, aj], DIGITS) for aj in list(aj_set)]
            for ai in list(ai_set)
        ]

        cur_matrix_T = [
            [round(Jij[ai, aj], DIGITS) for ai in list(ai_set)]
            for aj in list(aj_set)
        ]

        cur_row = {
            "i": model.mn(i) + 1,
            "j": model.mn(j) + 1,
            "score": round(score_ij, DIGITS),
            "iC": "".join(alphabet[ai_set]),
            "jC": "".join(alphabet[aj_set]),
            "matrix": cur_matrix,
        }

        cur_row_T = {
            "i": cur_row["j"],
            "j": cur_row["i"],
            "score": cur_row["score"],
            "iC": cur_row["jC"],
            "jC": cur_row["iC"],
            "matrix": cur_matrix_T,
        }

        matrix.append(cur_row)
        matrix.append(cur_row_T)

    # assemble sequence logo
    fi = model.fi()
    q = model.num_symbols

    # copy and blank out fi matrix to avoid numpy warnings
    # taking log of 0 (note where argument does not help)
    fi_no0 = fi.copy()
    fi_no0[fi <= 0] = np.nan
    B = -fi * np.log2(fi_no0)
    B[fi <= 0] = 0
    R = np.log2(q) - B.sum(axis=1)

    logo = []
    for i in range(model.L):
        order = np.argsort(fi[i, :])
        frequent = order[fi[i, order] >= freq_threshold]

        symbols = model.alphabet[frequent]
        fi_row = fi[i, frequent] * R[i]

        logo.append([
            {"code": s, "bits": round(float(h), DIGITS_LOGO)}
            for s, h in zip(symbols, fi_row)
        ])

    return map_, logo, matrix


def evzoom_json(model, **kwargs):
    """
    Create JSON input for visualizing parameters
    of pairwise graphical model using EVzoom

    Parameters
    ----------
    model : evcouplings.couplings.model.CouplingsModel
        Parameters of pairwise graphical model
    **kwargs
        See evzoom_data() for options

    Returns
    -------
    str
        EVzoom-ready JSON input
    """
    map_, logo, matrix = evzoom_data(model, **kwargs)

    data = {
        "map": map_,
        "logo": logo,
        "couplings": matrix,
    }

    return json.dumps(data)
