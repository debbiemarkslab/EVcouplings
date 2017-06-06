"""
Protocols for writing concatenated sequence alignments

Authors:
  Anna G. Green
"""

from evcouplings.align import Alignment, write_fasta
from collections import defaultdict
from operator import itemgetter


def unfilter(string):
    """
    uppercases all of the letters in string
    converts all "." to "-"
    """
    unf_string = string.upper()
    unf_string = unf_string.replace(".", "-")
    return unf_string


def write_concatenated_alignment(id_pairing,
                                 id_to_full_header_1,
                                 id_to_full_header_2,
                                 alignment_1,
                                 alignment_2,
                                 target_sequence_1,
                                 target_sequence_2,
                                 concatenated_alignment_file,
                                 monomer_alignment_file_1=None,
                                 monomer_alignment_file_2=None):
    """
    Parameters
    ----------
    id_pairing
    alignment_1
    alignment_2
    target_sequence_1
    target_sequence_2
    uniprot_to_id
    concatenated_alignment_file

    """

    def _identity(seq1, seq2):
        id = 0
        for i, j in zip(seq1, seq2):
            if i == j and i != "-":
                id += 1
        return id

    def _get_full_header(id,
                         id_to_header,
                         ali,
                         target_header):
        """
        if id points to unique header, return that header
        if id points to multiple headers, get the header
        that has closest id to target sequence
        TODO: is this the best way to select the real hit? 
        """
        if len(id_to_header[id]) == 1:
            return id_to_header[id][0]
        else:
            sequence_to_identity = []
            target_seq = ali[ali.id_to_index[target_header]]

            for full_id in id_to_header[id]:
                seq = ali[ali.id_to_index[full_id]]
                sequence_to_identity.append((full_id, _identity(target_seq, seq)))

            sequence_to_identity = sorted(sequence_to_identity, key=itemgetter(1), ascending=False)
        return sequence_to_identity[0][0]

    def _prepare_header(id1, id2, full_header_1, full_header_2):
        header_format = "{}_{} {} {}"  # id1_id2 full_header_1 full_header_2
        # header_format = "{}_{}"
        concatenated_header = header_format.format(id1, id2,
                                                   full_header_1,
                                                   full_header_2)
        return concatenated_header

    def _prepare_sequence(ali, full_header):
        """
        extracts the sequence from the alignment
        converts to string
        uppercases
        """
        sequence = ali[ali.id_to_index[full_header]]
        sequence = "".join(sequence)
        sequence = unfilter(sequence)
        return sequence

    sequences_to_write = []  # list of (header,seq1,seq2) tuples

    # load the alignments
    ali_1 = Alignment.from_file(open(alignment_1, "r"))
    ali_2 = Alignment.from_file(open(alignment_2, "r"))

    # create target header and target sequence
    # Format id1_id2 full_header_1 full_header_2
    # target full header is equivalent to target sequence

    target_full_header_1 = id_to_full_header_1[target_sequence_1][0]
    target_full_header_2 = id_to_full_header_2[target_sequence_2][0]

    target_sequences = (_prepare_sequence(ali_1, target_full_header_1),
                        _prepare_sequence(ali_2, target_full_header_2))

    # Target header must end with /1-range for correct focus mode
    length = len(target_sequences[0]) + len(target_sequences[1])

    target_header = "{}_{}/1-{}".format(
        target_sequence_1.split("/")[0],
        target_sequence_2.split("/")[0],
        length
    )

    sequences_to_write.append((target_header,
                               target_sequences[0],
                               target_sequences[1]))
    target_seq_idx = 0  # the target sequence is the first in the output file

    # create other headers and sequences
    for id1, id2 in id_pairing:
        full_header_1 = _get_full_header(id1, id_to_full_header_1, ali_1, target_full_header_1)
        full_header_2 = _get_full_header(id2, id_to_full_header_2, ali_2, target_full_header_2)

        concatenated_header = _prepare_header(id1, id2, full_header_1, full_header_2)

        concatenated_sequences = (_prepare_sequence(ali_1, full_header_1),
                                  _prepare_sequence(ali_2, full_header_2))

        sequences_to_write.append((concatenated_header,
                                   concatenated_sequences[0],
                                   concatenated_sequences[1]))

    sequences = [(a, b + c) for a, b, c in sequences_to_write]
    write_fasta(sequences, open(concatenated_alignment_file, "w"))

    # if monomer 1 filename is given, write monomer 1 seqs
    if monomer_alignment_file_1:
        sequences = [(a, b) for a, b, c in sequences_to_write]
        write_fasta(sequences, open(monomer_alignment_file_1, "w"))

    # if monomer 2 filename is given, write monomer 2 seqs
    if monomer_alignment_file_2:
        sequences = [(a, c) for a, b, c in sequences_to_write]
        write_fasta(sequences, open(monomer_alignment_file_2, "w"))

    return target_header, target_seq_idx
