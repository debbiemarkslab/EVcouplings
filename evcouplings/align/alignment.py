"""
Class and functions for reading, storing and manipulating
multiple sequence alignments.

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple
from copy import deepcopy
import numpy as np
from evcouplings.utils.helpers import DefaultOrderedDict, wrap


def read_fasta(fileobj):
    """
    Generator function to read a FASTA-format file
    (includes aligned FASTA, A2M, A3M formats)

    Parameters
    ----------
    fileobj : file-like object
        FASTA alignment file

    Returns
    -------
    generator of (str, str) tuples
        Returns tuples of (sequence ID, sequence)
    """
    current_sequence = ""
    current_id = None

    for line in fileobj:
        # Start reading new entry. If we already have
        # seen an entry before, return it first.
        if line.startswith(">"):
            if current_id is not None:
                yield current_id, current_sequence

            current_id = line.rstrip()[1:]
            current_sequence = ""

        elif not line.startswith(";"):
            current_sequence += line.rstrip()

    # Also do not forget last entry in file
    yield current_id, current_sequence


# Holds information of a parsed Stockholm alignment file
StockholmAlignment = namedtuple(
    "StockholmAlignment",
    ["seqs", "gf", "gc", "gs", "gr"]
)


def read_stockholm(fileobj, read_annotation=False):
    """
    Generator function to read Stockholm format alignment
    file (e.g. from hmmer).

    Notes:
    Generator iterates over different alignments in the
    same file (but not over individual sequences, which makes
    little sense due to wrapped Stockholm alignments).


    Parameters
    ----------
    fileobj : file-like object
        Stockholm alignment file
    read_annotation : bool, optional (default=False)
        Read annotation columns from alignment

    Returns
    -------
    StockholmAlignment
        namedtuple with the following fields:
        seqs, gf, gc, gs, gr (all DefaultOrderedDict)

    Raises
    ------
    ValueError
    """
    seqs = DefaultOrderedDict(str)

    """
    Markup definition: http://sonnhammer.sbc.su.se/Stockholm.html
    #=GF <feature> <Generic per-File annotation, free text>
    #=GC <feature> <Generic per-Column annotation, exactly 1 char per column>
    #=GS <seqname> <feature> <Generic per-Sequence annotation, free text>
    #=GR <seqname> <feature> <Generic per-Residue annotation, exactly 1 char per residue>
    """
    gf = DefaultOrderedDict(list)
    gc = DefaultOrderedDict(str)
    gs = DefaultOrderedDict(lambda: DefaultOrderedDict(list))
    gr = DefaultOrderedDict(lambda: DefaultOrderedDict(str))

    # line counter within current alignment (can be more than one per file)
    i = 0

    # read alignment
    for line in fileobj:
        if i == 0 and not line.startswith("# STOCKHOLM 1.0"):
            raise ValueError(
                "Not a valid Stockholm alignment: "
                "Header missing. {}".format(line.rstrip())
            )

        # annotation lines
        if line.startswith("#"):
            if read_annotation:
                if line.startswith("#=GF"):
                    # can have multiple lines for the same feature
                    _, feat, val = line.rstrip().split(maxsplit=2)
                    gf[feat].append(val)
                elif line.startswith("#=GC"):
                    # only one line with the same GC label
                    _, feat, seq = line.rstrip().split(maxsplit=2)
                    gc[feat] += seq
                elif line.startswith("#=GS"):
                    # can have multiple lines for the same feature
                    _, seq_id, feat, val = line.rstrip().split(maxsplit=3)
                    gs[seq_id][feat] = val
                elif line.startswith("#=GR"):
                    # per sequence, only one line with a certain GR feature
                    _, seq_id, feat, seq = line.rstrip().split()
                    gr[seq_id][feat] += seq

            i += 1

        # terminator line for current alignment;
        # only yield once we see // to avoid reading
        # truncated alignments
        elif line.startswith("//"):
            yield StockholmAlignment(seqs, gf, gc, gs, gr)

            # reset counter to check for valid start of alignment
            # once we read the next line
            i = 0

        # actual alignment lines
        else:
            splitted = line.rstrip().split(maxsplit=2)
            # there might be empty lines, so check for valid split
            if len(splitted) == 2:
                seq_id, seq = splitted
                seqs[seq_id] += seq

            i += 1

    # Do NOT yield at the end without // to avoid returning truncated alignments


def read_a3m(fileobj):
    """
    Read an alignment in compressed a3m format and expand
    into a2m format.

    Parameters
    ----------
    fileobj : file-like object
        A3M alignment file

    Returns
    -------
    ???

    # TODO: implement
    """
    raise NotImplementedError


class Alignment(object):
    """
    Container to store and manipulate multiple sequence alignments.

    Important:
    (1) Sequence annotation currently is not transformed when
    selecting subsets of columns or positions (e.g. affects GR and GC
    lines in Stockholm alignments)
    (2) Sequence ranges in IDs are not adjusted when selecting
    subsets of positions

    # TODO: missing features
    - changing from uppercase to lowercase
    - add sequence identity calculations back in
    """
    def __init__(self, fileobj, format="fasta"):
        """
        Parameters
        ----------
        fileobj : file-like obj
            Alignment to be read in
        format : {"fasta", "stockholm"}
            Format of input alignment

        Raises
        ------
        ValueError
            For invalid alignments or alignment formats
        """
        # meta annotation (e.g. from Stockholm format)
        self.annotation = {}

        if fileobj is None:
            return

        # read in sequence alignment from file
        if format == "fasta":
            seqs = DefaultOrderedDict()
            for seq_id, seq in read_fasta(fileobj):
                seqs[seq_id] = seq
        elif format == "stockholm":
            # only reads first Stockholm alignment contained in file
            ali = next(read_stockholm(fileobj, read_annotation=True))
            seqs = ali.seqs
            self.annotation["GF"] = ali.gf
            self.annotation["GC"] = ali.gc
            self.annotation["GS"] = ali.gs
            self.annotation["GR"] = ali.gr
        else:
            raise ValueError("Invalid alignment format: {}".format(format))

        # convert to internal matrix representation
        self.matrix, self.N, self.L = Alignment.to_matrix(seqs)

        # store sequence ID mappings
        self.ids = list(seqs.keys())
        self.id_to_index = {
            id_: i for i, id_ in enumerate(self.ids)
        }

    def __getitem__(self, index):
        """
        # TODO: eventually this should allow fancy indexing
        """
        if index in self.id_to_index:
            return self.matrix[self.id_to_index[index], :]
        elif index in range(self.N):
            return self.matrix[index, :]
        else:
            raise KeyError(
                "Not a valid index for sequence alignment: {}".format(index)
            )

    def __len__(self):
        return self.N

    @staticmethod
    def to_matrix(sequences):
        """
        Transforms a dictionary with sequences

        Parameters
        ----------
        sequences : dict(str)
            Dictionary from sequence IDs (keys) to
            sequences (values)

        Returns
        -------
        numpy.array
            2D array containing sequence alignment
            (first axis: sequences, second axis: columns)
        int
            Number of sequences in the alignment
        int
            Number of columns in the alignment
        """
        if len(sequences) == 0:
            raise ValueError("Need at least one sequence")

        N = len(sequences)
        L = len(next(iter(sequences.values())))
        matrix = np.empty((N, L), dtype=np.str)

        for i, (seq_id, seq) in enumerate(sequences.items()):
            if len(seq) != L:
                raise ValueError(
                    "Sequences have differing lengths: i={} L_0={} L_i={}".format(
                        seq_id, L, len(seq)
                    )
                )

            matrix[i] = np.array(list(seq))

        return matrix, N, L

    def count(self, char, axis="pos", normalize=True):
        """
        Count occurrences of a character in the sequence
        alignment.

        Parameters
        ----------
        char : str
            Character which is counted
        axis : {"pos", "seq"}, optional (default="pos")
            Count along positions or sequences
        normalize : bool, optional (default=True)
            Normalize count for length of axis (i.e. relative count)

        Returns
        -------
        np.array
            Vector containing counts of char along the axis

        Raises
        ------
        ValueError
            Upon invalid axis specification
        """
        if axis == "pos":
            naxis = 0
        elif axis == "seq":
            naxis = 1
        else:
            raise ValueError("Invalid axis: {}".format(axis))

        c = np.sum(self.matrix == char, axis=naxis)
        if normalize:
            c = c / self.matrix.shape[naxis]

        return c

    def select(self, columns=None, sequences=None):
        """
        Create a sub-alignment that contains a subset of
        sequences and/or columns.

        Note: This does currently not adjust the indices
        of the sequences. Annotation in the original alignment
        will be lost and not passed on to the new object.

        Parameters
        ----------
        columns : np.array(bool)
            Vector containing True for each column that
            should be retained, False otherwise
        sequences : np.array(bool)
            Vector containing True for each sequence
            that should be retained, False otherwise

        Returns
        -------
        Alignment
            Alignment with selected columns and sequences
            (note this alignment looses annotation)
        """
        if columns is None and sequences is None:
            return self

        # create empty alignment
        selected = Alignment(None)

        if sequences is not None:
            selected.matrix = np.copy(self.matrix[sequences, :])
            selected.ids = [seq_id for (i, seq_id) in enumerate(self.ids) if sequences[i]]
        else:
            selected.ids = deepcopy(self.ids)

        if columns is not None:
            selected.matrix = np.copy(self.matrix[:, columns])

        selected.N, selected.L = selected.matrix.shape
        selected.id_to_index = {
            id_: i for i, id_ in enumerate(selected.ids)
        }

        return selected

    def write(self, fileobj, format="fasta", width=80):
        """
        Write an alignment to a file.

        Parameters
        ----------
        fileobj : file-like object
            File to which alignment is saved
        format : {"fasta", "aln"}
            Output format for alignment
        width : int
            Column width for fasta alignment

        Raises
        ------
        ValueError
            Upon invalid file format specification
        """
        for i, seq_id in enumerate(self.ids):
            seq = "".join(self.matrix[i])

            if format == "fasta":
                fileobj.write(">{}\n".format(seq_id))
                fileobj.write(wrap(seq, width=width) + "\n")
            elif format == "aln":
                fileobj.write(seq + "\n")
            else:
                raise ValueError(
                    "Invalid alignment format: {}".format(format)
                )
