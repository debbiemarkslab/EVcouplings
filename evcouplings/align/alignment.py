"""
Class and functions for reading, storing and manipulating
multiple sequence alignments.

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple
from evcouplings.utils.helpers import DefaultOrderedDict


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

    # read alignment
    for i, line in enumerate(fileobj):
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

        # terminator line for current alignment;
        # only yield once we see // to avoid reading
        # truncated alignments
        elif line.startswith("//"):
            yield StockholmAlignment(seqs, gf, gc, gs, gr)

        # actual alignment lines
        else:
            splitted = line.rstrip().split(maxsplit=2)
            # there might be empty lines, so check for valid split
            if len(splitted) == 2:
                seq_id, seq = splitted
                seqs[seq_id] += seq

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
    # TODO
    """
    pass
