"""
Index mapping for PDB structures

Authors:
  Thomas A. Hopf
  Charlotta P. Schärfe
"""

import numpy as np
import pandas as pd

from evcouplings.align.alignment import Alignment, parse_header


def map_indices(seq_i, start_i, end_i,
                seq_j, start_j, end_j, gaps=("-", ".")):
    """
    Compute index mapping between positions in two
    aligned sequences

    Parameters
    ----------
    # TODO

    Returns
    -------
    # TODO
    """
    NA = np.nan
    pos_i = start_i
    pos_j = start_j
    mapping = []

    for i, (res_i, res_j) in enumerate(zip(seq_i, seq_j)):
        # Do we match two residues, or residue and a gap?
        # if matching two residues, store 1 to 1 mapping.
        # Store positions as strings, since pandas cannot
        # handle nan values in integer columns
        if res_i not in gaps and res_j not in gaps:
            mapping.append([str(pos_i), res_i, str(pos_j), res_j])
        elif res_i not in gaps:
            mapping.append([str(pos_i), res_i, NA, NA])
        elif res_j not in gaps:
            mapping.append([NA, NA, str(pos_j), res_j])

        # adjust position in sequences if we saw a residue
        if res_i not in gaps:
            pos_i += 1

        if res_j not in gaps:
            pos_j += 1

    assert pos_i - 1 == end_i and pos_j - 1 == end_j

    return pd.DataFrame(
        mapping, columns=["i", "A_i", "j", "A_j"]
    )


def alignment_index_mapping(alignment_file, format="stockholm",
                            target_seq=None):
    """

    Create index mapping table between sequence positions
    based on a sequence alignment.

    Parameters
    ----------
    # TODO

    Returns
    -------
    # TODO
    """
    # read alignment that is basis of mapping
    with open(alignment_file) as a:
        ali = Alignment.from_file(a, format)

    # determine index of target sequence if necessary
    # (default: first sequence in alignment)
    if target_seq is None:
        target_seq_index = 0
    else:
        for i, full_id in enumerate(ali.ids):
            if full_id.startswith(target_seq):
                target_seq_index = i

    # get range and sequence of target
    id_, target_start, target_end = parse_header(
        ali.ids[target_seq_index]
    )
    target_seq = ali.matrix[target_seq_index]

    # now map from target numbering to hit numbering
    full_map = None

    for i, full_id in enumerate(ali.ids):
        if i == target_seq_index:
            continue

        # extract information about sequence we are comparing to
        id_, region_start, region_end = parse_header(full_id)
        other_seq = ali.matrix[i]

        # compute mapping table
        map_df = map_indices(
            target_seq, target_start, target_end,
            other_seq, region_start, region_end,
            [ali._match_gap, ali._insert_gap]
        )

        # adjust column names for non-target sequence
        map_df = map_df.rename(
            columns={
                "j": "i_" + full_id,
                "A_j": "A_i_" + full_id,
            }
        )

        # add to full mapping table, left outer join
        # so all positions in target sequence are kept
        if full_map is None:
            full_map = map_df
        else:
            full_map = full_map.merge(
                map_df, on=("i", "A_i"), how="left"
            )

    return full_map
