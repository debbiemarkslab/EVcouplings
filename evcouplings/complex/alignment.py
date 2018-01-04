"""
Functions for writing concatenated sequence alignments

Authors:
  Anna G. Green
"""
from collections import OrderedDict
import numpy as np
from evcouplings.align import Alignment, write_fasta, parse_header


def write_concatenated_alignment(id_pairing, alignment_1, alignment_2,
                                 target_sequence_1, target_sequence_2):
    """
    Concatenate monomer alignments into a complex alignment
    and output to file.
    
    Parameters
    ----------
    id_pairing : pd.DataFrame
        dataframe with columns id_1 and id_2
        indicating the pairs of sequences to be concatenated
    alignment_1 : str
        Path to alignment file for first monomer alignment
    alignment_2 : str
        Path to alignment file for second monomer alignment
    target_sequence_1 : str
        Target sequence identifier for first monomer alignment
    target_sequence_2 : str
        Target sequence identifier for second monomer alignment

    Returns
    -------
    str
        Header of the concatenated target sequence
    int
        Index of target sequence in the alignment
    Alignment
        the full concatenated alignment
    Alignment
        An alignment of the first monomer sequences with
        only the sequences contained in the concatenated
        alignment
    Alignment
        An alignment of the second monomer sequences with
        only the sequences contained in the concatenated
        aligment
    """

    def _unfilter(string):
        """
        Uppercases all of the letters in string,
        converts all "." to "-"
        """
        string = np.char.upper(string)
        string[string=="."] = "-"
        return string

    def _prepare_header(id1, id2):
        # id1_id2
        header_format = "{}_{}"
        concatenated_header = header_format.format(id1, id2)

        return concatenated_header

    sequences_to_write = []  # list of (header,seq1,seq2) tuples

    # load the monomer alignments
    with open(alignment_1) as f1, open(alignment_2) as f2:
        ali_1 = Alignment.from_file(f1)
        ali_2 = Alignment.from_file(f2)

    ali_1 = ali_1.apply(func=_unfilter,columns=np.array(range(ali_1.matrix.shape[1])))
    ali_2 = ali_2.apply(func=_unfilter,columns=np.array(range(ali_2.matrix.shape[1])))

    target_index_1 = ali_1.id_to_index[target_sequence_1]
    target_index_2 = ali_2.id_to_index[target_sequence_2]

    # prepare the target sequence
    target_sequences = (
        ali_1.matrix[target_index_1, :],
        ali_2.matrix[target_index_2, :]
    )

    # Target header must end with /1-range for correct focus mode
    length = len(target_sequences[0]) + len(target_sequences[1])

    target_header = "{}_{}/1-{}".format(
        parse_header(target_sequence_1)[0],
        parse_header(target_sequence_2)[0],
        length
    )

    # store target sequence for writing
    sequences_to_write.append(
        (target_header, target_sequences[0], target_sequences[1])
    )

    # the target sequence is the first in the output file
    target_seq_idx = 0

    # create other headers and sequences
    for id1, id2 in zip(id_pairing.id_1, id_pairing.id_2):

        # prepare the concatenated header
        concatenated_header = _prepare_header(id1, id2)

        # get indices of the sequences
        index_1 = ali_1.id_to_index[id1]
        index_2 = ali_2.id_to_index[id2]

        # save the information
        sequences_to_write.append(
            (
                concatenated_header,
                ali_1.matrix[index_1, :],
                ali_2.matrix[index_2, :]
            )
        )

    # concatenate strings
    sequences_full = OrderedDict([
        (header, np.concatenate([seq1, seq2])) for header, seq1, seq2 in sequences_to_write
    ])

    sequences_monomer_1 = OrderedDict([
        (header, seq1) for header, seq1, seq2 in sequences_to_write
    ])

    sequences_monomer_2 = OrderedDict([
        (header, seq2) for header, seq1, seq2 in sequences_to_write
    ])

    full_ali = Alignment.from_dict(sequences_full)
    monomer_ali_1 = Alignment.from_dict(sequences_monomer_1)
    monomer_ali_2 = Alignment.from_dict(sequences_monomer_2)

    return target_header, target_seq_idx, full_ali, monomer_ali_1, monomer_ali_2
