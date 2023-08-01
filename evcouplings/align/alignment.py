"""
Class and functions for reading, storing and manipulating
multiple sequence alignments.

Authors:
  Thomas A. Hopf
"""

import re
from collections import namedtuple, OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from numba import jit

from evcouplings.utils.calculations import entropy
from evcouplings.utils.helpers import DefaultOrderedDict, wrap

# constants
GAP = "-"
MATCH_GAP = GAP
INSERT_GAP = "."

ALPHABET_PROTEIN_NOGAP = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_PROTEIN = GAP + ALPHABET_PROTEIN_NOGAP

# amino acid alphabet ordered by amino acid properties
ALPHABET_PROTEIN_NOGAP_ORDERED = "KRHEDNQTSCGAVLIMPYFW"
# keep in line with convention that first character is gap
ALPHABET_PROTEIN_ORDERED = GAP + ALPHABET_PROTEIN_NOGAP_ORDERED

ALPHABET_DNA_NOGAP = "ACGT"
ALPHABET_DNA = GAP + ALPHABET_DNA_NOGAP

ALPHABET_RNA_NOGAP = "ACGU"
ALPHABET_RNA = GAP + ALPHABET_RNA_NOGAP

HMMER_PREFIX_WARNING = "# WARNING: seq names have been made unique by adding a prefix of"


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


def write_fasta(sequences, fileobj, width=80):
    """
    Write a list of IDs/sequences to a FASTA-format file

    Parameters
    ----------
    sequences : Iterable
        Iterator returning tuples(str, str), where
        first value is header/ID and second the
        sequence
    fileobj : file-like obj
        File to which alignment will be written
    """
    for (seq_id, seq) in sequences:
        fileobj.write(">{}\n".format(seq_id))
        fileobj.write(wrap(seq, width=width) + "\n")


def write_aln(sequences, fileobj, width=80):
    """
    Write a list of IDs/sequences to a ALN-format file

    Currently, the file will not contain headers but simply
    a block matrix of the alignment

    Parameters
    ----------
    sequences : Iterable
        Iterator returning tuples(str, str), where
        first value is header/ID and second the
        sequence
    fileobj : file-like obj
        File to which alignment will be written
    """
    for (seq_id, seq) in sequences:
        fileobj.write(seq + "\n")


# Holds information of a parsed Stockholm alignment file
StockholmAlignment = namedtuple(
    "StockholmAlignment",
    ["seqs", "gf", "gc", "gs", "gr"]
)


def read_stockholm(fileobj, read_annotation=False, raise_hmmer_prefixes=True):
    """
    Generator function to read Stockholm format alignment
    file (e.g. from hmmer).

    .. note::

        Generator iterates over different alignments in the
        same file (but not over individual sequences, which makes
        little sense due to wrapped Stockholm alignments).


    Parameters
    ----------
    fileobj : file-like object
        Stockholm alignment file
    read_annotation : bool, optional (default=False)
        Read annotation columns from alignment
    raise_hmmer_prefixes : bool, optional (default: True)
        HMMER adds number prefixes to sequence identifiers if
        identifiers are not unique. If True, the parser will
        raise an exception if the alignment has such prefixes.

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

        if raise_hmmer_prefixes and line.startswith(HMMER_PREFIX_WARNING):
            raise ValueError(
                "HMMER added identifier prefixes to alignment because of non-unique "
                "sequence identifiers. Either some sequence identifier is present "
                "twice in the sequence database, or your target sequence identifier is "
                "the same as an identifier in the database. In the first case, please fix "
                "your sequence database. In the second case, please choose a different "
                "sequence identifier for your target sequence that does not overlap with "
                "the sequence database."
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


def read_a3m(fileobj, inserts="first"):
    """
    Read an alignment in compressed a3m format and expand
    into a2m format.

    .. note::

        this function is currently not able to keep inserts in all the sequences

    ..todo::

        implement this

    Parameters
    ----------
    fileobj : file-like object
        A3M alignment file
    inserts : {"first", "delete"}
        Keep inserts in first sequence, or delete
        any insert column and keep only match state
        columns.

    Returns
    -------
    OrderedDict
        Sequences in alignment (key: ID, value: sequence),
        in order they appeared in input file

    Raises
    ------
    ValueError
        Upon invalid choice of insert strategy
    """
    seqs = OrderedDict()

    for i, (seq_id, seq) in enumerate(read_fasta(fileobj)):
        # remove any insert gaps that may still be in alignment
        # (just to be sure)
        seq = seq.replace(".", "")

        if inserts == "first":
            # define "spacing" of uppercase columns in
            # final alignment based on target sequence;
            # remaining columns will be filled with insert
            # gaps in the other sequences
            if i == 0:
                uppercase_cols = [
                    j for (j, c) in enumerate(seq)
                    if (c == c.upper() or c == "-")
                ]
                gap_template = np.array(["."] * len(seq))
                filled_seq = seq
            else:
                uppercase_chars = [
                    c for c in seq if c == c.upper() or c == "-"
                ]
                filled = np.copy(gap_template)
                filled[uppercase_cols] = uppercase_chars
                filled_seq = "".join(filled)

        elif inserts == "delete":
            # remove all lowercase letters and insert gaps .;
            # since each sequence must have same number of
            # uppercase letters or match gaps -, this gives
            # the final sequence in alignment
            seq = "".join([c for c in seq if c == c.upper() and c != "."])
        else:
            raise ValueError(
                "Invalid option for inserts: {}".format(inserts)
            )

        seqs[seq_id] = filled_seq

    return seqs


def write_a3m(sequences, fileobj, insert_gap=INSERT_GAP, width=80):
    """
    Write a list of IDs/sequences to a FASTA-format file

    Parameters
    ----------
    sequences : Iterable
        Iterator returning tuples(str, str), where
        first value is header/ID and second the
        sequence
    fileobj : file-like obj
        File to which alignment will be written
    """
    for (seq_id, seq) in sequences:
        fileobj.write(">{}\n".format(seq_id))
        fileobj.write(seq.replace(insert_gap, "") + "\n")


def detect_format(fileobj, filepath=""):
    """
    Detect if an alignment file is in FASTA or
    Stockholm format.

    Parameters
    ----------
    fileobj : file-like obj
        Alignment file for which to detect format
    filepath : string or path-like obj
        Path of alignment file

    Returns
    -------
    format : {"fasta", "a3m", "stockholm", None}
        Format of alignment, None if not detectable
    """
    for i, line in enumerate(fileobj):
        # must be first line of Stockholm file by definition
        if i == 0 and line.startswith("# STOCKHOLM 1.0"):
            return "stockholm"

        # This indicates a FASTA file
        if line.startswith(">"):
            # A3M files have extension .a3m
            if Path(filepath).suffix.lower() == ".a3m":
                return "a3m"
            return "fasta"

        # Skip comment lines and empty lines for FASTA detection
        if line.startswith(";") or line.rstrip() == "":
            continue

        # Arriving here means we could not detect format
        return None


def parse_header(header):
    """
    Extract ID of the (overall) sequence and the
    sequence range fromat a sequence header of the form
    sequenceID/start-end.

    If the header contains any additional information
    after the first whitespace character (e.g. sequence annotation),
    it will be discarded before parsing. If there is no sequence
    range, only the id (part of string before whitespace) will
    be returned but no range.

    Parameters
    ----------
    header : str
        Sequence header

    Returns
    -------
    seq_id : str
        Sequence identifier
    start : int
        Start of sequence range. Will be None if it cannot
        be extracted from the header.
    stop : int
        End of sequence range. Will be None if it cannot
        be extracted from the header.
    """
    # discard anything in header that might come after the
    # first whitespace (by convention this is typically annotation)
    header = header.split()[0]

    # try to extract region from sequence header
    m = re.search("(.+)/(\d+)-(\d+)", header)
    if m:
        id_, start_str, end_str = m.groups()
        region_start, region_end = int(start_str), int(end_str)
        return id_, region_start, region_end
    else:
        # cannot find region, so just give back sequence iD
        return header, None, None


def sequences_to_matrix(sequences):
    """
    Transforms a list of sequences into a
    numpy array.

    Parameters
    ----------
    sequences : list-like (str)
        List of strings containing aligned sequences

    Returns
    -------
    numpy.array
        2D array containing sequence alignment
        (first axis: sequences, second axis: columns)
    """
    if len(sequences) == 0:
        raise ValueError("Need at least one sequence")

    N = len(sequences)
    L = len(next(iter(sequences)))
    matrix = np.empty((N, L), dtype=str)

    for i, seq in enumerate(sequences):
        if len(seq) != L:
            raise ValueError(
                "Sequences have differing lengths: i={} L_0={} L_i={}".format(
                    i, L, len(seq)
                )
            )

        matrix[i] = np.array(list(seq))

    return matrix


def map_from_alphabet(alphabet=ALPHABET_PROTEIN, default=GAP):
    """
    Creates a mapping dictionary from a given alphabet.

    Parameters
    ----------
    alphabet : str
        Alphabet for remapping. Elements will
        be remapped according to alphabet starting
        from 0
    default : Elements in matrix that are not
        contained in alphabet will be treated as
        this character

    Raises
    ------
    ValueError
        For invalid default character
    """
    map_ = {
        c: i for i, c in enumerate(alphabet)
    }

    try:
        default = map_[default]
    except KeyError:
        raise ValueError(
            "Default {} is not in alphabet {}".format(default, alphabet)
        )

    return defaultdict(lambda: default, map_)


def map_matrix(matrix, map_):
    """
    Map elements in a numpy array using alphabet

    Parameters
    ----------
    matrix : np.array
        Matrix that should be remapped
    map_ : defaultdict
        Map that will be applied to matrix elements

    Returns
    -------
    np.array
        Remapped matrix
    """
    return np.vectorize(map_.__getitem__)(matrix)


class Alignment:
    """
    Container to store and manipulate multiple sequence alignments.

    .. note::

        Important:

        1. Sequence annotation currently is not transformed when
           selecting subsets of columns or positions (e.g. affects GR and GC
           lines in Stockholm alignments)
        2. Sequence ranges in IDs are not adjusted when selecting
           subsets of positions
    """
    def __init__(self, sequence_matrix, sequence_ids=None, annotation=None,
                 alphabet=ALPHABET_PROTEIN):
        """
        Create new alignment object from ready-made components.

        .. note::

            Use factory method Alignment.from_file to create alignment from file,
            or Alignment.from_dict from dictionary of sequences.

        Parameters
        ----------
        sequence_matrix : np.array
            N x L array of characters in the alignment
            (N=number of sequences, L=width of alignment)
        sequence_ids : list-like, optional (default=None)
            Sequence names of alignment members (must have N elements).
            If None, defaults sequence IDs to "0", "1", ...
        annotation : dict-like
            Annotation for sequence alignment

        Raises
        ------
        ValueError
            If dimensions of sequence_matrix and sequence_ids
            are inconsistent
        """
        self.matrix = np.array(sequence_matrix)
        self.N, self.L = self.matrix.shape

        # characters coding for gaps in match-state and insert
        # columns of the alignment
        self._match_gap = MATCH_GAP
        self._insert_gap = INSERT_GAP

        # defined alphabet of alignment
        self.alphabet = alphabet
        self.alphabet_default = self._match_gap

        self.alphabet_map = map_from_alphabet(
            self.alphabet, default=self.alphabet_default
        )
        self.num_symbols = len(self.alphabet_map)

        # Alignment matrix remapped into in integers
        # Will only be calculated if necessary for downstream
        # calculations
        self.matrix_mapped = None
        self.num_cluster_members = None
        self.weights = None
        self._frequencies = None
        self._pair_frequencies = None

        if sequence_ids is None:
            # default to numbering sequences if not given
            self.ids = [str(i) for i in range(self.N)]
        else:
            if len(sequence_ids) != self.N:
                raise ValueError(
                    "Number of sequence IDs and length of "
                    "alignment do not match".format(
                        len(sequence_ids), self.L
                    )
                )

            # make sure we get rid of iterators etc.
            self.ids = list(sequence_ids)

        # turn identifiers into numpy array for consistency with previous implementation;
        # but use dtype object to avoid memory usage issues of numpy string datatypes (longest
        # sequence defines memory usage otherwise)
        self.ids = np.array(self.ids, dtype=np.object_)

        self.id_to_index = {
            id_: i for i, id_ in enumerate(self.ids)
        }

        if annotation is not None:
            self.annotation = annotation
        else:
            self.annotation = {}

    @classmethod
    def from_dict(cls, sequences, **kwargs):
        """
        Construct an alignment object from a dictionary
        with sequence IDs as keys and aligned sequences
        as values.

        Parameters
        ----------
        sequences : dict-like
            Dictionary with pairs of sequence ID (key) and
            aligned sequence (value)

        Returns
        -------
        Alignment
            initialized alignment
        """
        matrix = sequences_to_matrix(sequences.values())

        return cls(
            matrix, sequences.keys(), **kwargs
        )

    @classmethod
    def from_file(cls, fileobj, format="fasta",
                  a3m_inserts="first", raise_hmmer_prefixes=True,
                  split_header=False, **kwargs):
        """
        Construct an alignment object by reading in an
        alignment file.

        Parameters
        ----------
        fileobj : file-like obj
            Alignment to be read in
        format : {"fasta", "stockholm", "a3m"}
            Format of input alignment
        a3m_inserts : {"first", "delete"}, optional (default: "first")
            Strategy to deal with inserts in a3m alignment files
            (see read_a3m documentation for details)
        raise_hmmer_prefixes : bool, optional (default: True)
            HMMER adds number prefixes to sequence identifiers in Stockholm
            files if identifiers are not unique. If True, the parser will
            raise an exception if a Stockholm alignment has such prefixes.
        split_header: bool, optional (default: False)
            Only store identifier portion of each header (before first whitespace)
            in identifier list, rather than full header line
        **kwargs
            Additional arguments to be passed to class constructor

        Returns
        -------
        Alignment
            Parsed alignment

        Raises
        ------
        ValueError
            For invalid alignments or alignment formats
        """
        annotation = {}
        # read in sequence alignment from file

        if format == "fasta":
            seqs = OrderedDict()
            for seq_id, seq in read_fasta(fileobj):
                seqs[seq_id] = seq
        elif format == "stockholm":
            # only reads first Stockholm alignment contained in file
            ali = next(
                read_stockholm(
                    fileobj, read_annotation=True,
                    raise_hmmer_prefixes=raise_hmmer_prefixes
                )
            )
            seqs = ali.seqs
            annotation["GF"] = ali.gf
            annotation["GC"] = ali.gc
            annotation["GS"] = ali.gs
            annotation["GR"] = ali.gr
            kwargs["annotation"] = annotation
        elif format == "a3m":
            seqs = read_a3m(fileobj, inserts=a3m_inserts)
        else:
            raise ValueError("Invalid alignment format: {}".format(format))

        # reduce header lines to identifiers if requested
        if split_header:
            seqs = {
                header.split()[0]: seq for header, seq in seqs.items()
            }

        return cls.from_dict(seqs, **kwargs)

    def __getitem__(self, index):
        """
        .. todo::

            eventually this should allow fancy indexing and offer the functionality of select()
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

    def count(self, char, axis="pos", normalize=True):
        """
        Count occurrences of a character in the sequence
        alignment.

        .. note::

            The counts are raw counts not adjusted for
            sequence redundancy.

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

        .. note::

            This does currently not adjust the indices
            of the sequences. Annotation in the original alignment
            will be lost and not passed on to the new object.

        Parameters
        ----------
        columns : np.array(bool) or np.array(int), optional
            Vector containing True for each column that
            should be retained, False otherwise; or the
            indices of columns that should be selected
        sequences : np.array(bool) or np.array(int), optional
            Vector containing True for each sequence that
            should be retained, False otherwise; or the
            indices of sequences that should be selected

        Returns
        -------
        Alignment
            Alignment with selected columns and sequences
            (note this alignment looses annotation)
        """
        if columns is None and sequences is None:
            return self

        sel_matrix = self.matrix
        ids = self.ids

        if columns is not None:
            sel_matrix = sel_matrix[:, columns]

        if sequences is not None:
            sel_matrix = sel_matrix[sequences, :]
            ids = ids[sequences]

        # do not copy annotation since it may become
        # inconsistent
        return Alignment(
            np.copy(sel_matrix), np.copy(ids),
            alphabet=self.alphabet
        )

    def apply(self, columns=None, sequences=None, func=np.char.lower):
        """
        Apply a function along columns and/or rows of alignment matrix,
        or to entire matrix. Note that column and row selections are
        applied independently in this particular order.

        Parameters
        ----------
        columns : np.array(bool) or np.array(int), optional
            Vector containing True for each column that
            should be retained, False otherwise; or the
            indices of columns that should be selected
        sequences : np.array(bool) or np.array(int), optional
            Vector containing True for each sequence that
            should be retained, False otherwise; or the
            indices of sequences that should be selected
        func : callable
            Vectorized numpy function that will be applied to
            the selected subset of the alignment matrix

        Returns
        -------
        Alignment
            Alignment with modified columns and sequences
            (this alignment maintains annotation)
        """
        mod_matrix = np.copy(self.matrix)

        if columns is None and sequences is None:
            return self
        else:
            if columns is not None:
                mod_matrix[:, columns] = func(mod_matrix[:, columns])

            if sequences is not None:
                mod_matrix[sequences, :] = func(mod_matrix[sequences, :])

        return Alignment(
            mod_matrix, deepcopy(self.ids), deepcopy(self.annotation),
            alphabet=self.alphabet
        )

    def replace(self, original, replacement, columns=None, sequences=None):
        """
        Replace character with another in full matrix or
        subset of columns/sequences.

        Parameters
        ----------
        original : char
            Character that should be replaced
        replacement : char
            Replacement character
        columns : numpy index array
            See self.apply for explanation
        sequences : numpy index array
            See self.apply for explanation

        Returns
        -------
        Alignment
            Alignment with replaced characters

        """
        return self.apply(
            columns, sequences,
            func=lambda x: np.char.replace(
                x, original, replacement
            )
        )

    def lowercase_columns(self, columns):
        """
        Change a subset of columns to lowercase character
        and replace "-" gaps with "." gaps, e.g. to exclude
        them from EC calculations

        Parameters
        ----------
        columns : numpy index array
            Subset of columns to make lowercase

        Returns
        -------
        Alignment
            Alignment with lowercase columns
        """
        return self.apply(
            columns=columns, func=np.char.lower
        ).replace(
            self._match_gap, self._insert_gap, columns=columns
        )

    def __ensure_mapped_matrix(self):
        """
        Ensure self.matrix_mapped exists
        """
        if self.matrix_mapped is None:
            self.matrix_mapped = map_matrix(
                self.matrix, self.alphabet_map
            )

    def set_weights(self, identity_threshold=0.8):
        """
        Calculate weights for sequences in alignment by
        clustering all sequences with sequence identity
        greater or equal to the given threshold.

        .. note::

            This method sets self.weights. After this method was called, methods/attributes such as
            self.frequencies or self.conservation()
            will make use of sequence weights.

        .. note::

            (Implementation: cannot use property here since we need identity threshold as a parameter....)

        Parameters
        ----------
        identity_threshold : float, optional (default: 0.8)
            Sequence identity threshold
        """
        self.__ensure_mapped_matrix()

        self.num_cluster_members = num_cluster_members(
            self.matrix_mapped, identity_threshold
        )
        self.weights = 1.0 / self.num_cluster_members

        # reset frequencies, since these were based on
        # different weights before or had no weights at all
        self._frequencies = None
        self._pair_frequencies = None

    @property
    def frequencies(self):
        """
        Returns/calculates single-site frequencies of symbols in alignment.
        Also sets self._frequencies member variable for later reuse.

        Previously calculated sequence weights using self.set_weights()
        will be used to adjust frequency counts; otherwise, each sequence
        will contribute with equal weight.

        Returns
        -------
        np.array
            Reference to self._frequencies
        """
        if self._frequencies is None:
            self.__ensure_mapped_matrix()

            # use precalculated sequence weights, but only
            # if we have explicitly calculated them before
            # (expensive calculation)
            if self.weights is None:
                weights = np.ones((self.N))
            else:
                weights = self.weights

            self._frequencies = frequencies(
                self.matrix_mapped, weights, self.num_symbols
            )

        return self._frequencies

    @property
    def pair_frequencies(self):
        """
        Returns/calculates pairwise frequencies of symbols in alignment.
        Also sets self._pair_frequencies member variable for later reuse.

        Previously calculated sequence weights using self.set_weights()
        will be used to adjust frequency counts; otherwise, each sequence
        will contribute with equal weight.

        Returns
        -------
        np.array
            Reference to self._pair_frequencies
        """
        if self._pair_frequencies is None:
            self.__ensure_mapped_matrix()

            if self.weights is None:
                weights = np.ones((self.N))
            else:
                weights = self.weights

            self._pair_frequencies = pair_frequencies(
                self.matrix_mapped, weights,
                self.num_symbols, self.frequencies
            )

        return self._pair_frequencies

    def identities_to(self, seq, normalize=True):
        """
        Calculate sequence identity between sequence
        and all sequences in the alignment.

        seq : np.array, list-like, or str
            Sequence for comparison
        normalize : bool, optional (default: True)
            Calculate relative identity between 0 and 1
            by normalizing with length of alignment
        """
        self.__ensure_mapped_matrix()

        # make sure this doesnt break with strings
        seq = np.array(list(seq))

        seq_mapped = map_matrix(seq, self.alphabet_map)
        ids = identities_to_seq(seq_mapped, self.matrix_mapped)

        if normalize:
            return ids / self.L
        else:
            return ids

    def conservation(self, normalize=True):
        """
        Calculate per-column conservation of sequence alignment
        based on entropy of single-column frequency distribution.

        If self.set_weights() was called previously, the
        frequencies used to calculate site entropies will be based
        on reweighted sequences.

        Parameters
        ----------
        normalize : bool, optional (default: True)
            Transform column entropy to range 0 (no conservation)
            to 1 (fully conserved)

        Returns
        -------
        np.array
            Vector of length L with conservation scores
        """
        return np.apply_along_axis(
            lambda x: entropy(x, normalize=normalize),
            axis=1, arr=self.frequencies
        )

    def write(self, fileobj, format="fasta", width=80):
        """
        Write an alignment to a file.

        Parameters
        ----------
        fileobj : file-like object
            File to which alignment is saved
        format : {"fasta", "aln", "a3m"}
            Output format for alignment
        width : int
            Column width for fasta alignment

        Raises
        ------
        ValueError
            Upon invalid file format specification
        """
        seqs = (
            (id_, "".join(self.matrix[i]))
            for (i, id_) in enumerate(self.ids)
        )

        if format == "fasta":
            write_fasta(seqs, fileobj, width)
        elif format == "a3m":
            write_a3m(seqs, fileobj, self._insert_gap, width)
        elif format == "aln":
            write_aln(seqs, fileobj, width)
        else:
            raise ValueError(
                "Invalid alignment format: {}".format(format)
            )


@jit(nopython=True)
def frequencies(matrix, seq_weights, num_symbols):
    """
    Calculate single-site frequencies of symbols in alignment

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    seq_weights : np.array
        Vector of length N containing weight for each sequence
    num_symbols : int
        Number of different symbols contained in alignment

    Returns
    -------
    np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters
    """
    N, L = matrix.shape
    fi = np.zeros((L, num_symbols))
    for s in range(N):
        for i in range(L):
            fi[i, matrix[s, i]] += seq_weights[s]

    return fi / seq_weights.sum()


@jit(nopython=True)
def pair_frequencies(matrix, seq_weights, num_symbols, fi):
    """
    Calculate pairwise frequencies of symbols in alignment.

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    seq_weights : np.array
        Vector of length N containing weight for each sequence
    num_symbols : int
        Number of different symbols contained in alignment
    fi : np.array
        Matrix of size L x num_symbols containing relative
        column frequencies of all characters.

    Returns
    -------
    np.array
        Matrix of size L x L x num_symbols x num_symbols containing
        relative pairwise frequencies of all character combinations
    """
    N, L = matrix.shape
    fij = np.zeros((L, L, num_symbols, num_symbols))
    for s in range(N):
        for i in range(L):
            for j in range(i + 1, L):
                fij[i, j, matrix[s, i], matrix[s, j]] += seq_weights[s]
                fij[j, i, matrix[s, j], matrix[s, i]] = fij[i, j, matrix[s, i], matrix[s, j]]

    # normalize frequencies by the number
    # of effective sequences
    fij /= seq_weights.sum()

    # set the frequency of a pair (alpha, alpha)
    # in position i to the respective single-site
    # frequency of alpha in position i
    for i in range(L):
        for alpha in range(num_symbols):
            fij[i, i, alpha, alpha] = fi[i, alpha]

    return fij


@jit(nopython=True)
def identities_to_seq(seq, matrix):
    """
    Calculate number of identities to given target sequence
    for all sequences in the matrix

    Parameters
    ----------
    seq : np.array
        Vector of length L containing mapped sequence
        (using map_matrix function)
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols)
        using map_matrix function

    Returns
    -------
    np.array
        Vector of length N containing number of identities
        to each sequence in matrix
    """
    N, L = matrix.shape
    identities = np.zeros((N, ))

    for i in range(N):
        id_i = 0
        for j in range(L):
            if matrix[i, j] == seq[j]:
                id_i += 1

        identities[i] = id_i

    return identities


@jit(nopython=True)
def num_cluster_members(matrix, identity_threshold):
    """
    Calculate number of sequences in alignment
    within given identity_threshold of each other

    Parameters
    ----------
    matrix : np.array
        N x L matrix containing N sequences of length L.
        Matrix must be mapped to range(0, num_symbols) using
        map_matrix function
    identity_threshold : float
        Sequences with at least this pairwise identity will be
        grouped in the same cluster.

    Returns
    -------
    np.array
        Vector of length N containing number of cluster
        members for each sequence (inverse of sequence
        weight)
    """
    N, L = matrix.shape
    L = 1.0 * L

    # minimal cluster size is 1 (self)
    num_neighbors = np.ones((N))

    # compare all pairs of sequences
    for i in range(N - 1):
        for j in range(i + 1, N):
            pair_id = 0
            for k in range(L):
                if matrix[i, k] == matrix[j, k]:
                    pair_id += 1

            if pair_id / L >= identity_threshold:
                num_neighbors[i] += 1
                num_neighbors[j] += 1

    return num_neighbors
