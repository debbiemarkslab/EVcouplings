"""
Protocols for writing concatenated sequence alignments

Authors:
  Anna G. Green
"""

from operator import itemgetter
from evcouplings.align import Alignment, write_fasta


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
    Concatenate monomer alignments into a complex alignment
    and output to file.
    
    Parameters
    ----------
    id_pairing : pd.DataFrame
        dataframe with columns uniprot_id_1 and uniprot_id_2
        indicating the pairs of sequences to be concatenated
    id_to_full_header_1 : dict (str to list of str)
        Sequence identifiers pointing to list of full headers in
        the alignment corresponding to that sequence identifier
        for first monomer alignment
    id_to_full_header_2 : dict (str to list of str)
        Same for second monomer alignment
    alignment_1 : str
        Path to alignment file for first monomer alignment
    alignment_2 : str
        Path to alignment file for second monomer alignment
    target_sequence_1 : str
        Target sequence identifier for first monomer alignment
    target_sequence_2 : str
        Target sequence identifier for second monomer alignment
    concatenated_alignment_file : str
        Path where concatenated alignment file will be stored
    monomer_alignment_file_1 : str, optional (default=None)
        Path to write first monomer alignment including
        only the sequences contained in the concatenated
        alignment
    monomer_alignment_file_2 : str, optional (default=None)
        Path to write second monomer alignment including
        only the sequences contained in the concatenated
        alignment

    Returns
    -------
    str
        Header of the concatenated target sequence
    int
        Index of target sequence in the alignment
    """

    def _unfilter(string):
        """
        Uppercases all of the letters in string,
        converts all "." to "-"
        """
        unf_string = string.upper()
        unf_string = unf_string.replace(".", "-")
        return unf_string

    def _identity(seq1, seq2):
        id_ = 0
        for i, j in zip(seq1, seq2):
            if i == j and i != "-":
                id_ += 1

        return id_

    def _get_full_header(id_, id_to_header,
                         ali, target_header):
        """
        if id points to unique header, return that header
        if id points to multiple headers, get the header
        that has closest id to target sequence

        # TODO: is this the best way to select the real hit?
        """
        if len(id_to_header[id_]) == 1:
            return id_to_header[id_][0]

        else:
            print(id_,id_to_header[id_])
            sequence_to_identity = []
            target_seq = ali[ali.id_to_index[target_header]]

            for full_id in id_to_header[id_]:
                seq = ali[ali.id_to_index[full_id]]
                sequence_to_identity.append(
                    (full_id, _identity(target_seq, seq))
                )

            sequence_to_identity = sorted(
                sequence_to_identity, key=itemgetter(1), reverse=True
            )

        return sequence_to_identity[0][0]

    def _prepare_header(id1, id2, full_header_1, full_header_2):
        # id1_id2 full_header_1 full_header_2
        header_format = "{}_{} {} {}"
        concatenated_header = header_format.format(
            id1, id2, full_header_1, full_header_2
        )

        return concatenated_header

    def _prepare_sequence(ali, full_header):
        """
        Extracts the sequence from the alignment,
        converts to string, uppercases
        """
        sequence = ali[ali.id_to_index[full_header]]
        sequence = "".join(sequence)
        sequence = _unfilter(sequence)
        return sequence

    sequences_to_write = []  # list of (header,seq1,seq2) tuples

    # load the monomer alignments
    with open(alignment_1) as f1, open(alignment_2) as f2:
        ali_1 = Alignment.from_file(f1)
        ali_2 = Alignment.from_file(f2)

    # create target header and target sequence
    # Format id1_id2 full_header_1 full_header_2
    # target full header is equivalent to target sequence
    target_full_header_1 = id_to_full_header_1[target_sequence_1][0]
    target_full_header_2 = id_to_full_header_2[target_sequence_2][0]

    target_sequences = (
        _prepare_sequence(ali_1, target_full_header_1),
        _prepare_sequence(ali_2, target_full_header_2)
    )

    # Target header must end with /1-range for correct focus mode
    length = len(target_sequences[0]) + len(target_sequences[1])

    target_header = "{}_{}/1-{}".format(
        target_sequence_1.split("/")[0],
        target_sequence_2.split("/")[0],
        length
    )

    # store target sequence for writing
    sequences_to_write.append(
        (target_header, target_sequences[0], target_sequences[1])
    )

    # the target sequence is the first in the output file
    target_seq_idx = 0

    # create other headers and sequences
    for id1, id2 in zip(id_pairing.id_1,id_pairing.id_2):

    	# get the full header for sequence 1 in the original alignment
        full_header_1 = _get_full_header(
            id1, id_to_full_header_1, ali_1, target_full_header_1
        )

		# get the full header for sequence 2 in the original alignment
        full_header_2 = _get_full_header(
            id2, id_to_full_header_2, ali_2, target_full_header_2
        )

        # prepare the concatenated header
        concatenated_header = _prepare_header(
            id1, id2, full_header_1, full_header_2
        )

        # concatenate the seqs
        concatenated_sequences = (
            _prepare_sequence(ali_1, full_header_1),
            _prepare_sequence(ali_2, full_header_2)
        )

        # save the information
        sequences_to_write.append(
            (
                concatenated_header,
                concatenated_sequences[0],
                concatenated_sequences[1]
            )
        )

    # concatenate strings
    sequences = [(a, b + c) for a, b, c in sequences_to_write]

    # write alignment to file
    with open(concatenated_alignment_file, "w") as f:
        write_fasta(sequences, f)

    # if monomer 1 filename is given, write monomer 1 seqs
    if monomer_alignment_file_1:
        sequences = [(a, b) for a, b, c in sequences_to_write]
        with open(monomer_alignment_file_1, "w") as f:
            write_fasta(sequences, f)

    # if monomer 2 filename is given, write monomer 2 seqs
    if monomer_alignment_file_2:
        sequences = [(a, c) for a, b, c in sequences_to_write]
        with open(monomer_alignment_file_2, "w") as f:
            write_fasta(sequences, f)

    return target_header, target_seq_idx
