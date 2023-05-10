"""
Protein sequence alignment creation protocols/workflows.

Authors:
  Thomas A. Hopf
  Anna G. Green - complex protocol, hmm_build_and_search
  Chan Kang - hmm_build_and_search

"""

from collections import OrderedDict
from collections.abc import Iterable
import re
from shutil import copy
import os

import numpy as np
import pandas as pd

from evcouplings.align import tools as at
from evcouplings.align.alignment import (
    detect_format, parse_header, read_fasta,
    write_fasta, Alignment
)

from evcouplings.couplings.mapping import Segment

from evcouplings.utils import BailoutException

from evcouplings.utils.config import (
    check_required, InvalidParameterError, MissingParameterError,
    read_config_file, write_config_file
)

from evcouplings.utils.system import (
    create_prefix_folders, get, valid_file,
    verify_resources, ResourceError
)

from evcouplings.align.ena import (
    extract_embl_annotation,
    extract_cds_ids,
    add_full_header
)


def _verify_sequence_id(sequence_id):
    """
    Verify if a target sequence identifier is in proper
    format for the pipeline to run without errors
    (not none, and contains no whitespace)
        
    Parameters
    ----------
    id : str
        Target sequence identifier to verify

    Raises
    ------
    InvalidParameterError
        If sequence identifier is not valid
    """
    if sequence_id is None:
        raise InvalidParameterError(
            "Target sequence identifier (sequence_id) must be defined and "
            "cannot be None/null."
        )

    try:
        if len(sequence_id.split()) != 1 or len(sequence_id) != len(sequence_id.strip()):
            raise InvalidParameterError(
                "Target sequence identifier (sequence_id) may not contain any "
                "whitespace (spaces, tabs, ...)"
            )
    except AttributeError:
        raise InvalidParameterError(
            "Target sequence identifier (sequence_id) must be a string"
        )


def _make_hmmsearch_raw_fasta(alignment_result, prefix):
    """
    HMMsearch results do not contain the query sequence
    so we must construct a raw_fasta file with the query 
    sequence as the first hit, to ensure proper numbering. 
    The search result is filtered to only contain the columns with
    match states to the HMM, which has a one to one mapping to the
    query sequence.

    Paramters
    ---------
    alignment_result : dict
        Alignment result dictionary, output by run_hmmsearch
    prefix : str
        Prefix for file creation

    Returns
    -------
    str
        path to raw focus alignment file

    """
    def _add_gaps_to_query(query_sequence_ali, ali):

         # get the index of columns that do not contain match states (indicated by an x)
        gap_index = [
            i for i, x in enumerate(ali.annotation["GC"]["RF"]) if x != "x"
        ]
        # get the index of columns that contain match states (indicated by an x)
        match_index = [
            i for i, x in enumerate(ali.annotation["GC"]["RF"]) if x == "x"
        ]

        # ensure that the length of the match states 
        # match the length of the sequence
        if len(match_index) != query_sequence_ali.L:
            raise ValueError(
                "HMMsearch result {} does not have a one-to-one"
                " mapping to the query sequence columns".format(
                    alignment_result["raw_alignment_file"]
                )
            )

        gapped_query_sequence = ""
        seq = list(query_sequence_ali.matrix[0, :])

        # loop through every position in the HMMsearch hits
        for i in range(len(ali.annotation["GC"]["RF"])):
            # if that position should be a gap, add a gap
            if i in gap_index:
                gapped_query_sequence += "-"
            # if that position should be a letter, pop the next
            # letter in the query sequence
            else:
                gapped_query_sequence += seq.pop(0)

        new_sequence_ali = Alignment.from_dict({
            query_sequence_ali.ids[0]: gapped_query_sequence
        })
        return new_sequence_ali

    # open the sequence file
    with open(alignment_result["target_sequence_file"]) as a:
        query_sequence_ali = Alignment.from_file(a, format="fasta")

    # if the provided alignment is empty, just return the target sequence 
    raw_focus_alignment_file = prefix + "_raw.fasta"
    if not valid_file(alignment_result["raw_alignment_file"]):
        # write the query sequence to a fasta file
        with open(raw_focus_alignment_file, "w") as of:
            query_sequence_ali.write(of)

        # return as an alignment object
        return raw_focus_alignment_file

    # else, open the HMM search result
    with open(alignment_result["raw_alignment_file"]) as a:
        ali = Alignment.from_file(a, format="stockholm")

    # make sure that the stockholm alignment contains the match annotation
    if not ("GC" in ali.annotation and "RF" in ali.annotation["GC"]):
        raise ValueError(
            "Stockholm alignment {} missing RF"
            " annotation of match states".format(alignment_result["raw_alignment_file"])
        )
            
    # add insertions to the query sequence in order to preserve correct
    # numbering of match sequences
    gapped_sequence_ali = _add_gaps_to_query(query_sequence_ali, ali)

    # write a new alignment file with the query sequence as 
    # the first entry
    
    with open(raw_focus_alignment_file, "w") as of:
        gapped_sequence_ali.write(of)
        ali.write(of)

    return raw_focus_alignment_file


def fetch_sequence(sequence_id, sequence_file,
                   sequence_download_url, out_file):
    """
    Fetch sequence either from database based on identifier, or from
    input sequence file.

    Parameters
    ----------
    sequence_id : str
        Identifier of sequence that should be retrieved
    sequence_file : str
        File containing sequence. If None, sqeuence will
        be downloaded from sequence_download_url
    sequence_download_url : str
        URL from which to download missing sequence. Must
        contain "{}" at the position where sequence ID will
        be inserted into download URL (using str.format).
    out_file : str
        Output file in which sequence will be stored, if
        sequence_file is not existing.

    Returns
    -------
    str
        Path of file with stored sequence (can be sequence_file
        or out_file)
    tuple (str, str)
        Identifier of sequence as stored in file, and sequence
    """
    if sequence_file is None:
        get(
            sequence_download_url.format(sequence_id),
            out_file,
            allow_redirects=True
        )
    else:
        # if we have sequence file, try to copy it
        try:
            copy(sequence_file, out_file)
        except FileNotFoundError:
            raise ResourceError(
                "sequence_file does not exist: {}".format(
                    sequence_file
                )
            )

    # also make sure input file has something in it
    verify_resources(
        "Input sequence missing", out_file
    )

    with open(out_file) as f:
        seq = next(read_fasta(f))

    return out_file, seq


def cut_sequence(sequence, sequence_id, region=None, first_index=None, out_file=None):
    """
    Cut a given sequence to sub-range and save it in a file

    Parameters
    ----------
    sequence : str
        Full sequence that will be cut
    sequence_id : str
        Identifier of sequence, used to construct header
        in output file
    region : tuple(int, int), optional (default: None)
        Region that will be cut out of full sequence.
        If None, full sequence will be returned.
    first_index : int, optional (default: None)
        Define index of first position in sequence.
        Will be set to 1 if None.
    out_file : str, optional (default: None)
        Save sequence in a FASTA file (header:
        >sequence_id/start_region-end_region)

    Returns
    ------
    str
        Subsequence contained in region
    tuple(int, int)
        Region. If no input region is given, this will be
        (1, len(sequence)); otherwise, the input region is
        returned.

    Raises
    ------
    InvalidParameterError
        Upon invalid region specification (violating boundaries
        of sequence)
    """
    cut_seq = None

    # (not using 1 as default value to allow parameter
    # to be unspecified in config file)
    if first_index is None:
        first_index = 1

    # last index is *inclusive*!
    if region is None:
        region = (first_index, first_index + len(sequence) - 1)
        cut_seq = sequence
    else:
        start, end = region
        str_start = start - first_index
        str_end = end - first_index + 1
        cut_seq = sequence[str_start:str_end]

        # make sure bounds are valid given the sequence that we have
        if str_start < 0 or str_end > len(sequence):
            raise InvalidParameterError(
                "Invalid sequence range: "
                "region={} first_index={} len(sequence)={}".format(
                    region,
                    first_index,
                    len(sequence)
                )
            )

    # save sequence to file
    if out_file is not None:
        with open(out_file, "w") as f:
            header = "{}/{}-{}".format(sequence_id, *region)
            write_fasta([(header, cut_seq)], f)

    return region, cut_seq


def search_thresholds(use_bitscores, seq_threshold, domain_threshold, seq_len):
    """
    Set homology search inclusion parameters.

    HMMER hits get included in the HMM according to a two-step rule
        1. sequence passes sequence-level treshold
        2. domain passes domain-level threshold

    Therefore, search thresholds are set based on the following logic:
        1. If only sequence threshold is given, a MissingParameterException is raised
        2. If only bitscore threshold is given, sequence threshold is set to the same
        3. If both thresholds are given, they are according to defined values

    Valid inputs for bitscore thresholds:
        1. int or str: taken as absolute score threshold
        2. float: taken as relative threshold (absolute threshold derived by
       multiplication with domain length)

    Valid inputs for integer thresholds:
        1. int: Used as negative exponent, threshold will be set to 1E-<exponent>
        2. float or str: Interpreted literally

    Parameters
    ----------
    use_bitscores : bool
        Use bitscore threshold instead of E-value threshold
    domain_threshold : str or int or float
        Domain-level threshold. See rules above.
    seq_threshold : str or int or float
        Sequence-level threshold. See rules above.
    seq_len : int
        Length of sequence. Used to calculate absolute bitscore
        threshold for relative bitscore thresholds.

    Returns
    -------
    tuple (str, str)
        Sequence- and domain-level thresholds ready to be fed into HMMER
    """
    def transform_bitscore(x):
        if isinstance(x, float):
            # float: interpret as relative fraction of length
            return "{:.1f}".format(x * seq_len)
        else:
            # otherwise interpret as absolute score
            return str(x)

    def transform_evalue(x):
        if isinstance(x, int):
            # if integer, interpret as negative exponent
            return "1E{}".format(-x)
        else:
            # otherwise interpret literally
            # (mantissa-exponent string or float)
            return str(x).upper()

    if domain_threshold is None:
        raise MissingParameterError(
            "domain_threshold must be explicitly defined "
            "and may not be None/empty"
        )

    if use_bitscores:
        transform = transform_bitscore
    else:
        transform = transform_evalue

    if seq_threshold is not None:
        seq_threshold = transform(seq_threshold)

    if domain_threshold is not None:
        domain_threshold = transform(domain_threshold)

    # set "outer" sequence threshold so that it matches domain threshold
    if domain_threshold is not None and seq_threshold is None:
        seq_threshold = domain_threshold

    return seq_threshold, domain_threshold


def extract_header_annotation(alignment, from_annotation=True):
    """
    Extract Uniprot/Uniref sequence annotation from Stockholm file
    (as output by jackhmmer). This function may not work for other
    formats.

    Parameters
    ----------
    alignment : Alignment
        Multiple sequence alignment object
    from_annotation : bool, optional (default: True)
        Use annotation line (in Stockholm file) rather
        than sequence ID line (e.g. in FASTA file)

    Returns
    -------
    pandas.DataFrame
        Table containing all annotation
        (one row per sequence in alignment,
        in order of occurrence)
    """
    columns = [
        ("GN", "gene"),
        ("OS", "organism"),
        ("PE", "existence_evidence"),
        ("SV", "sequence_version"),
        ("n", "num_cluster_members"),
        ("Tax", "taxon"),
        ("RepID", "representative_member")
    ]

    col_to_descr = OrderedDict(columns)
    regex = re.compile("\s({})=".format(
        "|".join(col_to_descr.keys()))
    )

    # collect rows for dataframe in here
    res = []

    for i, id_ in enumerate(alignment.ids):
        # annotation line for current sequence
        seq_id = None
        anno = None

        # look for annotation either in separate
        # annotation line or in full sequence ID line
        if from_annotation:
            seq_id = id_
            # query level by level to avoid creating new keys
            # in DefaultOrderedDict
            if ("GS" in alignment.annotation and
                    id_ in alignment.annotation["GS"] and
                    "DE" in alignment.annotation["GS"][id_]):
                anno = alignment.annotation["GS"][id_]["DE"]
        else:
            split = id_.split(maxsplit=1)
            if len(split) == 2:
                seq_id, anno = split
            else:
                seq_id = id_
                anno = None

        # extract info from line if we got one
        if anno is not None:
            # do split on known field names o keep things
            # simpler than a gigantic full regex to match
            # (some fields are allowed to be missing)
            pairs = re.split(regex, anno)
            pairs = ["id", seq_id, "name"] + pairs

            # create feature-value map
            feat_map = dict(zip(pairs[::2], pairs[1::2]))
            res.append(feat_map)
        else:
            res.append({"id": seq_id})

    df = pd.DataFrame(res)
    return df.reindex(
        ["id", "name"] + list(col_to_descr.keys()),
        axis=1
    )


def describe_seq_identities(alignment, target_seq_index=0):
    """
    Calculate sequence identities of any sequence
    to target sequence and create result dataframe.

    Parameters
    ----------
    alignment : Alignment
        Alignment for which description statistics
        will be calculated

    Returns
    -------
    pandas.DataFrame
        Table giving the identity to target sequence
        for each sequence in alignment (in order of
        occurrence)
    """
    id_to_query = alignment.identities_to(
        alignment[target_seq_index]
    )

    return pd.DataFrame(
        {"id": alignment.ids, "identity_to_query": id_to_query}
    )


def describe_frequencies(alignment, first_index, target_seq_index=None):
    """
    Get parameters of alignment such as gaps, coverage,
    conservation and summarize.

    Parameters
    ----------
    alignment : Alignment
        Alignment for which description statistics
        will be calculated
    first_index : int
        Sequence index of first residue in target sequence
    target_seq_index : int, optional (default: None)
        If given, will add the symbol in the target sequence
        into a separate column of the output table

    Returns
    -------
    pandas.DataFrame
        Table detailing conservation and symbol frequencies
        for all positions in the alignment
    """
    fi = alignment.frequencies
    conservation = alignment.conservation()

    # careful not to include any characters that are non-match state (e.g. lowercase letters)
    fi_cols = {
        c: fi[:, alignment.alphabet_map[c]] for c in alignment.alphabet
    }

    if target_seq_index is not None:
        target_seq = alignment[target_seq_index]
    else:
        target_seq = np.full((alignment.L, ), np.nan)

    info = pd.DataFrame(
        {
            "i": range(first_index, first_index + alignment.L),
            "A_i": target_seq,
            "conservation": conservation,
            **fi_cols
        }
    )
    # reorder columns
    info = info.loc[:, ["i", "A_i", "conservation"] + list(alignment.alphabet)]

    # do not report values for lowercase columns
    info.loc[
        info.A_i.str.lower() == info.A_i, ["conservation"] + list(alignment.alphabet)
    ] = np.nan

    return info


def describe_coverage(alignment, prefix, first_index, minimum_column_coverage):
    """
    Produce "classical" buildali coverage statistics, i.e.
    number of sequences, how many residues have too many gaps, etc.

    Only to be applied to alignments focused around the
    target sequence.

    Parameters
    ----------
    alignment : Alignment
        Alignment for which coverage statistics will be calculated
    prefix : str
        Prefix of alignment file that will be stored as identifier in table
    first_index : int
        Sequence index of first position of target sequence
    minimum_column_coverage : Iterable(float) or float
        Minimum column coverage threshold(s) that will be tested
        (creating one row for each threshold in output table).

        .. note::

            ``int`` values given to this function instead of a float will be divided by 100 to create the corresponding
            floating point representation. This parameter is 1.0 - maximum fraction of gaps per column.

    Returns
    -------
    pd.DataFrame
        Table with coverage statistics for different gap thresholds
    """
    res = []
    NO_MEFF = np.nan

    if not isinstance(minimum_column_coverage, Iterable):
        minimum_column_coverage = [minimum_column_coverage]

    pos = np.arange(first_index, first_index + alignment.L)
    f_gap = alignment.frequencies[:, alignment.alphabet_map[alignment._match_gap]]

    for threshold in minimum_column_coverage:
        if isinstance(threshold, int):
            threshold /= 100

        # all positions that have enough sequence information (i.e. little gaps),
        # and their indeces
        uppercase = f_gap <= 1 - threshold
        uppercase_idx = np.nonzero(uppercase)[0]

        # where does coverage of sequence by good alignment start and end?
        cov_first_idx, cov_last_idx = uppercase_idx[0], uppercase_idx[-1]

        # calculate indeces in sequence numbering space
        first, last = pos[cov_first_idx], pos[cov_last_idx]

        # how many lowercase positions in covered region?
        num_lc_cov = np.sum(~uppercase[cov_first_idx:cov_last_idx + 1])

        # total number of upper- and lowercase positions,
        # and relative percentage
        num_cov = uppercase.sum()
        num_lc = (~uppercase).sum()
        perc_cov = num_cov / len(uppercase)

        res.append(
            (prefix, threshold, alignment.N, alignment.L,
             num_cov, num_lc, perc_cov, first, last,
             last - first + 1, num_lc_cov, NO_MEFF)
        )

    df = pd.DataFrame(
        res, columns=[
            "prefix", "minimum_column_coverage", "num_seqs",
            "seqlen", "num_cov", "num_lc", "perc_cov",
            "1st_uc", "last_uc", "len_cov",
            "num_lc_cov", "N_eff",
        ]
    )
    return df


def existing(**kwargs):
    """
    Protocol:

    Use external sequence alignment and extract all relevant
    information from there (e.g. sequence, region, etc.),
    then apply gap & fragment filtering as usual

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * sequence_id (passed through from input)
        * alignment_file
        * raw_focus_alignment_file
        * statistics_file
        * sequence_file
        * first_index
        * target_sequence_file
        * annotation_file (None)
        * frequencies_file
        * identities_file
        * focus_mode
        * focus_sequence
        * segments
    """
    check_required(
        kwargs,
        [
            "prefix", "input_alignment",
            "sequence_id", "first_index",
            "extract_annotation"
        ]
    )

    prefix = kwargs["prefix"]

    # make sure output directory exists
    create_prefix_folders(prefix)

    # this file is starting point of pipeline;
    # check if input alignment actually exists
    input_alignment = kwargs["input_alignment"]
    verify_resources(
        "Input alignment does not exist",
        input_alignment
    )

    # first try to autodetect format of alignment
    with open(input_alignment) as f:
        format = detect_format(f, filepath=input_alignment)
        if format is None:
            raise InvalidParameterError(
                "Format of input alignment {} could not be "
                "automatically detected.".format(
                    input_alignment
                )
            )

    with open(input_alignment) as f:
        ali_raw = Alignment.from_file(f, format)

    # save annotation in sequence headers (species etc.)
    annotation_file = None
    if kwargs["extract_annotation"]:
        annotation_file = prefix + "_annotation.csv"
        from_anno_line = (format == "stockholm")
        annotation = extract_header_annotation(
            ali_raw, from_annotation=from_anno_line
        )
        annotation.to_csv(annotation_file, index=False)

    # Target sequence of alignment
    sequence_id = kwargs["sequence_id"]

    # check if sequence identifier is valid
    _verify_sequence_id(sequence_id)

    # First, find focus sequence in alignment
    focus_index = None
    for i, id_ in enumerate(ali_raw.ids):
        if id_.startswith(sequence_id):
            focus_index = i
            break

    # if we didn't find it, cannot continue
    if focus_index is None:
        raise InvalidParameterError(
            "Target sequence {} could not be found in alignment"
            .format(sequence_id)
        )

    # identify what columns (non-gap) to keep for focus
    focus_seq = ali_raw[focus_index]
    focus_cols = np.array(
        [c not in [ali_raw._match_gap, ali_raw._insert_gap] for c in focus_seq]
    )

    # extract focus alignment
    focus_ali = ali_raw.select(columns=focus_cols)
    focus_seq_nogap = "".join(focus_ali[focus_index])

    # determine region of sequence. If first_index is given,
    # use that in any case, otherwise try to autodetect
    full_focus_header = ali_raw.ids[focus_index]
    focus_id = full_focus_header.split()[0]

    # try to extract region from sequence header
    id_, region_start, region_end = parse_header(focus_id)

    # override with first_index if given
    if kwargs["first_index"] is not None:
        region_start = kwargs["first_index"]
        region_end = region_start + len(focus_seq_nogap) - 1

    if region_start is None or region_end is None:
        raise InvalidParameterError(
            "Could not extract region information " +
            "from sequence header {} ".format(full_focus_header) +
            "and first_index parameter is not given."
        )

    # resubstitute full sequence ID from identifier
    # and region information
    header = "{}/{}-{}".format(
        id_, region_start, region_end
    )

    focus_ali.ids[focus_index] = header

    # write target sequence to file
    target_sequence_file = prefix + ".fa"
    with open(target_sequence_file, "w") as f:
        write_fasta(
            [(header, focus_seq_nogap)], f
        )

    # apply sequence identity and fragment filters,
    # and gap threshold
    mod_outcfg, ali = modify_alignment(
        focus_ali, focus_index, id_, region_start, **kwargs
    )

    # generate output configuration of protocol
    outcfg = {
        **mod_outcfg,
        "sequence_id": sequence_id,
        "sequence_file": target_sequence_file,
        "first_index": region_start,
        "target_sequence_file": target_sequence_file,
        "focus_sequence": header,
        "focus_mode": True,
    }

    if annotation_file is not None:
        outcfg["annotation_file"] = annotation_file

    # dump config to YAML file for debugging/logging
    write_config_file(prefix + ".align_existing.outcfg", outcfg)

    # return results of protocol
    return outcfg


def modify_alignment(focus_ali, target_seq_index, target_seq_id, region_start, **kwargs):
    """
    Apply pairwise identity filtering, fragment filtering, and exclusion
    of columns with too many gaps to a sequence alignment. Also generates
    files describing properties of the alignment such as frequency distributions,
    conservation, and "old-style" alignment statistics files.

    .. note::

        assumes focus alignment (otherwise unprocessed) as input.

    .. todo::

        come up with something more clever  to filter fragments than fixed width
        (e.g. use 95% quantile of length distribution as reference point)

    Parameters
    ----------
    focus_ali : Alignment
        Focus-mode input alignment
    target_seq_index : int
        Index of target sequence in alignment
    target_seq_id : str
        Identifier of target sequence (without range)
    region_start : int
        Index of first sequence position in target sequence
    kwargs : See required arguments in source code

    Returns
    -------
    outcfg : Dict
        File products generated by the function:

        * alignment_file
        * statistics_file
        * frequencies_file
        * identities_file
        * raw_focus_alignment_file
    ali : Alignment
        Final processed alignment
    """
    check_required(
        kwargs,
        [
            "prefix", "seqid_filter", "hhfilter",
            "minimum_sequence_coverage", "minimum_column_coverage",
            "compute_num_effective_seqs", "theta",
        ]
    )

    prefix = kwargs["prefix"]

    create_prefix_folders(prefix)

    focus_fasta_file = prefix + "_raw_focus.fasta"

    outcfg = {
        "alignment_file": prefix + ".a2m",
        "statistics_file": prefix + "_alignment_statistics.csv",
        "frequencies_file": prefix + "_frequencies.csv",
        "identities_file": prefix + "_identities.csv",
        "raw_focus_alignment_file": focus_fasta_file,
    }

    # swap target sequence to first position if it is not
    # the first sequence in alignment;
    # this is particularly important for hhfilter run
    # because target sequence might otherwise be filtered out
    if target_seq_index != 0:
        indices = np.arange(0, len(focus_ali))
        indices[0] = target_seq_index
        indices[target_seq_index] = 0
        target_seq_index = 0
        focus_ali = focus_ali.select(sequences=indices)

    with open(focus_fasta_file, "w") as f:
        focus_ali.write(f, "fasta")

    # apply pairwise identity filter (using hhfilter)
    if kwargs["seqid_filter"] is not None:
        filtered_file = prefix + "_filtered.a3m"

        at.run_hhfilter(
            focus_fasta_file, filtered_file,
            threshold=kwargs["seqid_filter"],
            columns="first", binary=kwargs["hhfilter"]
        )

        with open(filtered_file) as f:
            focus_ali = Alignment.from_file(f, "a3m")

        # final FASTA alignment before applying A2M format modifications
        filtered_fasta_file = prefix + "_raw_focus_filtered.fasta"
        with open(filtered_fasta_file, "w") as f:
            focus_ali.write(f, "fasta")

    ali = focus_ali

    # filter fragments
    # come up with something more clever here than fixed width
    # (e.g. use 95% quantile of length distribution as reference point)
    min_cov = kwargs["minimum_sequence_coverage"]
    if min_cov is not None:
        if isinstance(min_cov, int):
            min_cov /= 100

        keep_seqs = (1 - ali.count("-", axis="seq")) >= min_cov
        ali = ali.select(sequences=keep_seqs)

    # Calculate frequencies, conservation and identity to query
    # on final alignment (except for lowercase modification)
    # Note: running hhfilter might cause a loss of the target seque
    # if it is not the first sequence in the file! To be sure that
    # nothing goes wrong, target_seq_index should always be 0.
    describe_seq_identities(
        ali, target_seq_index=target_seq_index
    ).to_csv(
        outcfg["identities_file"], float_format="%.3f", index=False
    )

    describe_frequencies(
        ali, region_start, target_seq_index=target_seq_index
    ).to_csv(
        outcfg["frequencies_file"], float_format="%.3f", index=False
    )

    coverage_stats = describe_coverage(
        ali, prefix, region_start, kwargs["minimum_column_coverage"]
    )

    # keep list of uppercase sequence positions in alignment
    pos_list = np.arange(region_start, region_start + ali.L, dtype="int32")

    # Make columns with too many gaps lowercase
    min_col_cov = kwargs["minimum_column_coverage"]
    if min_col_cov is not None:
        if isinstance(min_col_cov, int):
            min_col_cov /= 100

        lc_cols = ali.count(ali._match_gap, axis="pos") > 1 - min_col_cov
        ali = ali.lowercase_columns(lc_cols)

        # if we remove columns, we have to update list of positions
        pos_list = pos_list[~lc_cols]
    else:
        lc_cols = None

    # compute effective number of sequences
    # (this is intended for cases where coupling stage is
    # not run, but this number is wanted nonetheless)
    if kwargs["compute_num_effective_seqs"]:
        # make sure we only compute N_eff on the columns
        # that would be used for model inference, dispose
        # the rest
        if lc_cols is None:
            cut_ali = ali
        else:
            cut_ali = ali.select(columns=~lc_cols)

        # compute sequence weights
        cut_ali.set_weights(kwargs["theta"])

        # N_eff := sum of all sequence weights
        n_eff = float(cut_ali.weights.sum())

        # patch into coverage statistics (N_eff column)
        coverage_stats.loc[:, "N_eff"] = n_eff

        # create table with number of cluster members (inverse sequence
        # weights) for each sequence
        inv_seq_weights = pd.DataFrame({
            "id": cut_ali.ids,
            "num_cluster_members": cut_ali.num_cluster_members
        })

        # save sequence weights to file and add to output config
        outcfg["sequence_weights_file"] = prefix + "_inverse_sequence_weights.csv"
        inv_seq_weights.to_csv(
            outcfg["sequence_weights_file"], index=False
        )
    else:
        n_eff = None

    # save coverage statistics to file
    coverage_stats.to_csv(
        outcfg["statistics_file"], float_format="%.3f",
        index=False
    )

    # store description of final sequence alignment in outcfg
    # (note these parameters will be updated by couplings protocol)
    outcfg.update(
        {
            "num_sites": len(pos_list),
            "num_sequences": len(ali),
            "effective_sequences": n_eff,
            "region_start": region_start,
        }
    )

    # create segment in outcfg
    outcfg["segments"] = [
        Segment(
            "aa", target_seq_id, region_start, region_start + ali.L - 1, pos_list
        ).to_list()
    ]

    with open(outcfg["alignment_file"], "w") as f:
        ali.write(f, "fasta")

    return outcfg, ali


def jackhmmer_search(**kwargs):
    """
    Protocol:

    Iterative jackhmmer search against a sequence database.

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    .. todo::
        explain meaning of parameters in detail.

    Returns
    -------
    outcfg : dict
        Output configuration of the protocol, including
        the following fields:

        * sequence_id (passed through from input)
        * first_index (passed through from input)
        * target_sequence_file
        * sequence_file
        * raw_alignment_file
        * hittable_file
        * focus_mode
        * focus_sequence
        * segments
    """
    check_required(
        kwargs,
        [
            "prefix", "sequence_id", "sequence_file",
            "sequence_download_url", "region", "first_index",
            "use_bitscores", "domain_threshold", "sequence_threshold",
            "database", "iterations", "cpu", "nobias", "reuse_alignment",
            "checkpoints_hmm", "checkpoints_ali", "jackhmmer",
            "extract_annotation"
        ]
    )
    prefix = kwargs["prefix"]

    # check if sequence identifier is valid
    _verify_sequence_id(kwargs["sequence_id"])

    # make sure output directory exists
    create_prefix_folders(prefix)

    # store search sequence file here
    target_sequence_file = prefix + ".fa"
    full_sequence_file = prefix + "_full.fa"

    # make sure search sequence is defined and load it
    full_seq_file, (full_seq_id, full_seq) = fetch_sequence(
        kwargs["sequence_id"],
        kwargs["sequence_file"],
        kwargs["sequence_download_url"],
        full_sequence_file
    )

    # cut sequence to target region and save in sequence_file
    # (this is the main sequence file used downstream)
    (region_start, region_end), cut_seq = cut_sequence(
        full_seq,
        kwargs["sequence_id"],
        kwargs["region"],
        kwargs["first_index"],
        target_sequence_file
    )

    # run jackhmmer... allow to reuse pre-exisiting
    # Stockholm alignment file here
    ali_outcfg_file = prefix + ".align_jackhmmer_search.outcfg"

    # determine if to rerun, only possible if previous results
    # were stored in ali_outcfg_file
    if kwargs["reuse_alignment"] and valid_file(ali_outcfg_file):
        ali = read_config_file(ali_outcfg_file)

        # check if the alignment file itself is also there
        verify_resources(
            "Tried to reuse alignment, but empty or "
            "does not exist",
            ali["alignment"], ali["domtblout"]
        )
    else:
        # otherwise, we have to run the alignment
        # modify search thresholds to be suitable for jackhmmer
        seq_threshold, domain_threshold = search_thresholds(
            kwargs["use_bitscores"],
            kwargs["sequence_threshold"],
            kwargs["domain_threshold"],
            len(cut_seq)
        )

        # run search process
        ali = at.run_jackhmmer(
            query=target_sequence_file,
            database=kwargs[kwargs["database"]],
            prefix=prefix,
            use_bitscores=kwargs["use_bitscores"],
            domain_threshold=domain_threshold,
            seq_threshold=seq_threshold,
            iterations=kwargs["iterations"],
            nobias=kwargs["nobias"],
            cpu=kwargs["cpu"],
            checkpoints_hmm=kwargs["checkpoints_hmm"],
            checkpoints_ali=kwargs["checkpoints_ali"],
            binary=kwargs["jackhmmer"],
        )

        # get rid of huge stdout log file immediately
        # (do not use /dev/null option of jackhmmer function
        # to make no assumption about operating system)
        try:
            os.remove(ali.output)
        except OSError:
            pass

        # turn namedtuple into dictionary to make
        # restarting code nicer
        ali = dict(ali._asdict())

        # save results of search for possible restart
        write_config_file(ali_outcfg_file, ali)

    # prepare output dictionary with result files
    outcfg = {
        "sequence_id": kwargs["sequence_id"],
        "target_sequence_file": target_sequence_file,
        "sequence_file": full_sequence_file,
        "first_index": kwargs["first_index"],
        "focus_mode": True,
        "raw_alignment_file": ali["alignment"],
        "hittable_file": ali["domtblout"],
    }

    # define a single protein segment based on target sequence
    outcfg["segments"] = [
        Segment(
            "aa", kwargs["sequence_id"],
            region_start, region_end,
            range(region_start, region_end + 1)
        ).to_list()
    ]

    outcfg["focus_sequence"] = "{}/{}-{}".format(
        kwargs["sequence_id"], region_start, region_end
    )

    return outcfg


def hmmbuild_and_search(**kwargs):
    """
    Protocol:

    Build HMM from sequence alignment using hmmbuild and 
    search against a sequence database using hmmsearch.
    
    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the protocol, including
        the following fields:

        * target_sequence_file
        * sequence_file
        * raw_alignment_file
        * hittable_file
        * focus_mode
        * focus_sequence
        * segments
    """

    def _format_alignment_for_hmmbuild(input_alignment_file, **kwargs):
        # this file is starting point of pipeline;
        # check if input alignment actually exists

        verify_resources(
            "Input alignment does not exist",
            input_alignment_file
        )

        # first try to autodetect format of alignment
        with open(input_alignment_file) as f:
            format = detect_format(f)
            if format is None:
                raise InvalidParameterError(
                    "Format of input alignment {} could not be "
                    "automatically detected.".format(
                        input_alignment_file
                    )
                )

        with open(input_alignment_file) as f:
            ali_raw = Alignment.from_file(f, format)

        # Target sequence of alignment
        sequence_id = kwargs["sequence_id"]

        if sequence_id is None:
            raise InvalidParameterError(
                "Parameter sequence_id must be defined"
            )

        # First, find focus sequence in alignment
        focus_index = None
        for i, id_ in enumerate(ali_raw.ids):
            if id_.startswith(sequence_id):
                focus_index = i
                break

        # if we didn't find it, cannot continue
        if focus_index is None:
            raise InvalidParameterError(
                "Target sequence {} could not be found in alignment"
                .format(sequence_id)
            )

        # identify what columns (non-gap) to keep for focus
        # this should be all columns in the raw_focus_alignment_file
        # but checking anyway
        focus_seq = ali_raw[focus_index]
        focus_cols = np.array(
            [c not in [ali_raw._match_gap, ali_raw._insert_gap] for c in focus_seq]
        )

        # extract focus alignment
        focus_ali = ali_raw.select(columns=focus_cols)
        focus_seq_nogap = "".join(focus_ali[focus_index])

        # determine region of sequence. If first_index is given,
        # use that in any case, otherwise try to autodetect
        full_focus_header = ali_raw.ids[focus_index]
        focus_id = full_focus_header.split()[0]

        # try to extract region from sequence header
        id_, region_start, region_end = parse_header(focus_id)

        # override with first_index if given (but respect region from alignment if defined)
        if kwargs["first_index"] is not None and (region_start is None or region_end is None):
            region_start = kwargs["first_index"]
            region_end = region_start + len(focus_seq_nogap) - 1

        if region_start is None or region_end is None:
            raise InvalidParameterError(
                "Could not extract region information " +
                "from sequence header {} ".format(full_focus_header) +
                "and first_index parameter is not given."
            )

        # resubstitute full sequence ID from identifier
        # and region information
        header = "{}/{}-{}".format(
            id_, region_start, region_end
        )

        focus_ali.ids[focus_index] = header

        # write target sequence to file
        target_sequence_file = prefix + ".fa"
        with open(target_sequence_file, "w") as f:
            write_fasta(
                [(header, focus_seq_nogap)], f
            )

        # swap target sequence to first position if it is not
        # the first sequence in alignment;
        # this is particularly important for hhfilter run
        # because target sequence might otherwise be filtered out
        if focus_index != 0:
            indices = np.arange(0, len(focus_ali))
            indices[0] = focus_index
            indices[focus_index] = 0
            focus_index = 0
            focus_ali = focus_ali.select(sequences=indices)

        # write the raw focus alignment for hmmbuild
        focus_fasta_file = prefix + "_raw_focus_input.fasta"
        with open(focus_fasta_file, "w") as f:
            focus_ali.write(f, "fasta")

        return focus_fasta_file, target_sequence_file, region_start, region_end


    # define the gap threshold for inclusion in HMM's build by HMMbuild. 
    SYMFRAC_HMMBUILD = 0.0

    # check for required options
    check_required(
        kwargs,
        [
            "prefix", "sequence_id", "alignment_file",
            "use_bitscores", "domain_threshold", "sequence_threshold",
            "database", "cpu", "nobias", "reuse_alignment",
            "hmmbuild", "hmmsearch"
        ]
    )
    prefix = kwargs["prefix"]

    # check if sequence identifier is valid
    _verify_sequence_id(kwargs["sequence_id"])

    # make sure output directory exists
    create_prefix_folders(prefix)

    # prepare input alignment for hmmbuild
    focus_fasta_file, target_sequence_file, region_start, region_end = \
        _format_alignment_for_hmmbuild(
            kwargs["alignment_file"], **kwargs
        )

    # run hmmbuild_and_search... allow to reuse pre-exisiting
    # Stockholm alignment file here
    ali_outcfg_file = prefix + ".align_hmmbuild_and_search.outcfg"

    # determine if to rerun, only possible if previous results
    # were stored in ali_outcfg_file
    if kwargs["reuse_alignment"] and valid_file(ali_outcfg_file):
        ali = read_config_file(ali_outcfg_file)

        # check if the alignment file itself is also there
        verify_resources(
            "Tried to reuse alignment, but empty or "
            "does not exist",
            ali["alignment"], ali["domtblout"]
        )
    else:
        # otherwise, we have to run the alignment
        # modify search thresholds to be suitable for hmmsearch
        sequence_length = region_end - region_start + 1 
        seq_threshold, domain_threshold = search_thresholds(
            kwargs["use_bitscores"],
            kwargs["sequence_threshold"],
            kwargs["domain_threshold"],
            sequence_length
        )

        # create the hmm
        hmmbuild_result = at.run_hmmbuild(
            alignment_file=focus_fasta_file,
            prefix=prefix,
            symfrac=SYMFRAC_HMMBUILD,
            cpu=kwargs["cpu"],
            binary=kwargs["hmmbuild"],
        )
        hmmfile = hmmbuild_result.hmmfile

        # run the alignment from the hmm
        ali = at.run_hmmsearch(
            hmmfile=hmmfile,
            database=kwargs[kwargs["database"]],
            prefix=prefix,
            use_bitscores=kwargs["use_bitscores"],
            domain_threshold=domain_threshold,
            seq_threshold=seq_threshold,
            nobias=kwargs["nobias"],
            cpu=kwargs["cpu"],
            binary=kwargs["hmmsearch"], 
        )

        # get rid of huge stdout log file immediately
        try:
            os.remove(ali.output)
        except OSError:
            pass

        # turn namedtuple into dictionary to make
        # restarting code nicer
        ali = dict(ali._asdict())
        # only item from hmmsearch_result to save is the hmmfile
        ali["hmmfile"] = hmmfile

        # save results of search for possible restart
        write_config_file(ali_outcfg_file, ali)

    # prepare output dictionary with result files
    outcfg = {
        "sequence_file": target_sequence_file,
        "first_index": region_start,
        "input_raw_focus_alignment": focus_fasta_file,
        "target_sequence_file": target_sequence_file,
        "focus_mode": True,
        "raw_alignment_file": ali["alignment"],
        "hittable_file": ali["domtblout"],
    }

    # convert the raw output alignment to fasta format 
    # and add the appropriate query sequecne
    raw_focus_alignment_file = _make_hmmsearch_raw_fasta(outcfg, prefix)
    outcfg["raw_focus_alignment_file"] =  raw_focus_alignment_file

    # define a single protein segment based on target sequence
    outcfg["segments"] = [
        Segment(
            "aa", kwargs["sequence_id"],
            region_start, region_end,
            range(region_start, region_end + 1)
        ).to_list()
    ]

    outcfg["focus_sequence"] = "{}/{}-{}".format(
        kwargs["sequence_id"], region_start, region_end
    )

    return outcfg


def standard(**kwargs):
    """
    Protocol:

    Standard buildali4 workflow (run iterative jackhmmer
    search against sequence database, than determine which
    sequences and columns to include in the calculation based
    on coverage and maximum gap thresholds).

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * sequence_id (passed through from input)
        * first_index (passed through from input)
        * alignment_file
        * raw_alignment_file
        * raw_focus_alignment_file
        * statistics_file
        * target_sequence_file
        * sequence_file
        * annotation_file
        * frequencies_file
        * identities_file
        * hittable_file
        * focus_mode
        * focus_sequence
        * segments

    ali : Alignment
        Final sequence alignment

    """
    check_required(
        kwargs,
        [
            "prefix", "extract_annotation",
        ]
    )

    prefix = kwargs["prefix"]

    # make sure output directory exists
    create_prefix_folders(prefix)

    # first step of protocol is to get alignment using
    # jackhmmer; initialize output configuration with
    # results of this search
    jackhmmer_outcfg = jackhmmer_search(**kwargs)
    stockholm_file = jackhmmer_outcfg["raw_alignment_file"]

    segment = Segment.from_list(jackhmmer_outcfg["segments"][0])
    target_seq_id = segment.sequence_id
    region_start = segment.region_start
    region_end = segment.region_end

    # read in stockholm format (with full annotation)
    with open(stockholm_file) as a:
        ali_raw = Alignment.from_file(a, "stockholm")

    # and store as FASTA file first (disabled for now
    # since equivalent information easily be obtained
    # from Stockholm file
    """
    ali_raw_fasta_file = prefix + "_raw.fasta"
    with open(ali_raw_fasta_file, "w") as f:
        ali_raw.write(f, "fasta")
    """

    # save annotation in sequence headers (species etc.)
    if kwargs["extract_annotation"]:
        annotation_file = prefix + "_annotation.csv"
        annotation = extract_header_annotation(ali_raw)
        annotation.to_csv(annotation_file, index=False)
    else:
        annotation_file = None

    # center alignment around focus/search sequence
    focus_cols = np.array([c != "-" for c in ali_raw[0]])
    focus_ali = ali_raw.select(columns=focus_cols)

    target_seq_index = 0
    mod_outcfg, ali = modify_alignment(
        focus_ali, target_seq_index, target_seq_id, region_start, **kwargs
    )

    #  merge results of jackhmmer_search and modify_alignment stage
    outcfg = {
        **jackhmmer_outcfg,
        **mod_outcfg,
    }

    if annotation_file is not None:
        outcfg["annotation_file"] = annotation_file

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".align_standard.outcfg", outcfg)

    if len(ali) <= 1:
        raise BailoutException("align: No sequences found")

    # return results of protocol
    return outcfg


def complex(**kwargs):
    """
    Protocol:

    Run monomer alignment protocol and postprocess it for
    EVcomplex calculations

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the alignment protocol, and
        the following additional field:

        genome_location_file : path to file containing
            the genomic locations for CDs's corresponding to
            identifiers in the alignment.

    """
    check_required(
        kwargs,
        [
            "prefix", "alignment_protocol",
            "uniprot_to_embl_table",
            "ena_genome_location_table"
        ]
    )

    verify_resources(
        "Uniprot to EMBL mapping table does not exist",
        kwargs["uniprot_to_embl_table"]
    )

    verify_resources(
        "ENA genome location table does not exist",
        kwargs["ena_genome_location_table"]
    )

    prefix = kwargs["prefix"]

    # make sure output directory exists
    create_prefix_folders(prefix)

    # run the regular alignment protocol
    # (standard, existing, ...)
    alignment_protocol = kwargs["alignment_protocol"]

    if alignment_protocol not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid choice for alignment protocol: {}".format(
                alignment_protocol
            )
        )

    outcfg = PROTOCOLS[kwargs["alignment_protocol"]](**kwargs)

    # if the user selected the existing alignment protocol
    # they can supply an input annotation file
    # which overwrites the annotation file generated by the existing protocol
    if alignment_protocol == "existing":
        check_required(kwargs, ["override_annotation_file"])

        if kwargs["override_annotation_file"] is not None:
            verify_resources(
                "Override annotation file does not exist",
                kwargs["override_annotation_file"]
            )

            outcfg["annotation_file"] = prefix + "_annotation.csv"
            annotation_data = pd.read_csv(kwargs["override_annotation_file"])
            annotation_data.to_csv(outcfg["annotation_file"])

    # extract cds identifiers for alignment uniprot IDs
    cds_ids = extract_cds_ids(
        outcfg["alignment_file"],
        kwargs["uniprot_to_embl_table"]
    )

    # extract genome location information from ENA
    genome_location_filename = prefix + "_genome_location.csv"

    genome_location_table = extract_embl_annotation(
        cds_ids,
        kwargs["ena_genome_location_table"],
        genome_location_filename
    )

    genome_location_table = add_full_header(
        genome_location_table, outcfg["alignment_file"]
    )

    genome_location_table.to_csv(genome_location_filename)
    outcfg["genome_location_file"] = genome_location_filename

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".align_complex.outcfg", outcfg)

    return outcfg


# list of available alignment protocols
PROTOCOLS = {
    # standard buildali protocol (iterative hmmer search)
    "standard": standard,

    # build raw multiple sequence alignment using jackmmer
    "jackhmmer_search": jackhmmer_search,

    # start from an existing (external) alignment
    "existing": existing,

    # run alignment protocol and postprocess output for
    # complex pipeline
    "complex": complex,
}


def run(**kwargs):
    """
    Run alignment protocol to generate multiple sequence
    alignment from input sequence.

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: Alignment protocol to run
        prefix: Output prefix for all generated files

    Optional:

    Returns
    -------
    Alignment
    Dictionary with results of stage in following fields (in brackets - not returned by all protocols):

        * alignment_file
        * [raw_alignment_file]
        * statistics_file
        * target_sequence_file
        * sequence_file
        * [annotation_file]
        * frequencies_file
        * identities_file
        * [hittable_file]
        * focus_mode
        * focus_sequence
        * segments
    """
    check_required(kwargs, ["protocol"])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)
