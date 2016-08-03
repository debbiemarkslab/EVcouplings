"""
Protein sequence alignment creation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

from collections import OrderedDict, Iterable
import re

import numpy as np
import pandas as pd

import evcouplings.align.tools as at
from evcouplings.align.alignment import (
    read_fasta, write_fasta, Alignment
)
from evcouplings.utils.config import (
    check_required, MissingParameterError, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, get, file_not_empty, ResourceError
)


def fetch_sequence(sequence_id, sequence_file,
                   sequence_download_url, out_file):
    """
    Get sequence.

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
        # if we have sequence file, pass it through
        out_file = sequence_file

    # also make sure input file has something in it
    if not file_not_empty(out_file):
        raise ResourceError(
            "Input sequence missing: {}".format(out_file)
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


def create_segment(sequence_id, region_start, region_end,
                   segment_id="1", segment_type="aa"):
    """
    Create a segment for a monomer sequence search run

    Parameters
    ----------
    sequence_id : str
        Identifier of sequence
    region_start : int
        Start index of sequence segment
    region_end : int
        End index of sequence segment
    segment_id : str
        Identifier for segment (must be unique)
    segment_type : {"aa", "dna", "rna"}
        Type of sequence

    Returns
    -------
    tuple
        Segment description
    """
    return (
        segment_type,
        segment_id,
        sequence_id,
        region_start,
        region_end
    )


def search_thresholds(use_bitscores, seq_threshold, domain_threshold, seq_len):
    """
    Set homology search inclusion parameters.

    HMMER hits get included in the HMM according to a two-step rule
    1) sequence passes sequence-level treshold
    2) domain passes domain-level threshold

    Therefore, search thresholds are set based on the following logic:
    1) If only sequence threshold is given, a MissingParameterException is raised
    2) If only bitscore threshold is given, sequence threshold is set to the same
    3) If both thresholds are given, they are according to defined values

    Valid inputs for bitscore thresholds:
    1) int or str: taken as absolute score threshold
    2) float: taken as relative threshold (absolute threshold derived by
       multiplication with domain length)

    Valid inputs for integer thresholds:
    1) int: Used as negative exponent, threshold will be set to 1E-<exponent>
    2) float or str: Interpreted literally

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


def extract_header_annotation(alignment):
    """
    Extract Uniprot/Uniref sequence annotation from Stockholm file
    (as output by jackhmmer). This function may not work for other
    formats.

    Parameters:
    -----------
    alignment : Alignment
        Multiple sequence alignment object

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
        # query level by level to avoid creating new keys
        # in DefaultOrderedDict
        if ("GS" in alignment.annotation and
                id_ in alignment.annotation["GS"] and
                "DE" in alignment.annotation["GS"][id_]):
            anno = alignment.annotation["GS"][id_]["DE"]

            # do split on known field names o keep things
            # simpler than a gigantic full regex to match
            # (some fields may be missing)
            pairs = re.split(regex, anno)
            pairs = ["id", id_, "name"] + pairs

            # create feature-value map
            feat_map = dict(zip(pairs[::2], pairs[1::2]))
            res.append(feat_map)
        else:
            res.append({"id": id_})

    df = pd.DataFrame(res)
    return df.loc[:, ["id", "name"] + list(col_to_descr.keys())]


def modify_alignment(config):
    """
    Prepare alignment to be ready for EC calculation
    """
    return


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
    fi = alignment.frequencies()
    conservation = alignment.conservation()

    fi_cols = {c: fi[:, i] for c, i in alignment.alphabet_map.items()}
    if target_seq_index is not None:
        target_seq = alignment[target_seq_index]
    else:
        target_seq = np.full((alignment.L), np.nan)

    info = pd.DataFrame(
        {
            "pos": range(first_index, first_index + alignment.L),
            "target_seq": target_seq,
            "conservation": conservation,
            **fi_cols
        }
    )
    # reorder columns
    info = info.loc[:, ["pos", "target_seq", "conservation"] + list(alignment.alphabet)]

    return info


def describe_coverage(alignment, prefix, first_index, max_gaps_per_column):
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
    max_gaps_per_column : Iterable(float) or float
        Gap threshold(s) that will be tested (creating one row for each
        threshold in output table). Note that int values given to this
        function instead of a float will be divided by 100 to create
        the corresponding floating point representation.

    Returns
    -------
    pd.DataFrame
        Table with coverage statistics for different gap thresholds
    """
    res = []
    NO_MEFF = np.nan

    if not isinstance(max_gaps_per_column, Iterable):
        max_gaps_per_column = [max_gaps_per_column]

    pos = np.arange(first_index, first_index + alignment.L)
    f_gap = alignment.frequencies()[:, alignment.alphabet_map[alignment._match_gap]]

    for threshold in max_gaps_per_column:
        if isinstance(threshold, int):
            threshold /= 100

        # all positions that have enough sequence information (i.e. little gaps),
        # and their indeces
        uppercase = f_gap < threshold
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
            "prefix", "max_gaps_per_column", "num_seqs",
            "seqlen", "num_cov", "num_lc", "perc_cov",
            "1st_uc", "last_uc", "len_cov",
            "num_lc_cov", "N_eff",
        ]
    )
    return df


def external(**kwargs):
    """
    Protocol:
    Use external sequence alignment and extract all relevant
    information from there (e.g. sequence, region, etc.),
    then apply gap & fragment filtering as usual

    # Note: input alignment may already have lowercase positions
    """
    print("Start from existing alignment")
    return {}


def standard(**kwargs):
    """
    Protocol:
    Standard buildali4 workflow

    Parameters
    ----------
    # TODO

    If skip is given, ...
    If callback is given, ...

    Returns
    -------
    # TODO

    """
    check_required(
        kwargs,
        [
            "prefix", "sequence_id", "sequence_file",
            "sequence_download_url", "reuse_alignment",
            "region", "first_index",
            "use_bitscores", "domain_threshold", "sequence_threshold",
            "database", "iterations", "cpu", "nobias", "reuse_alignment",
            "checkpoints_hmm", "checkpoints_ali", "jackhmmer",
            "seqid_filter", "minimum_coverage", "max_gaps_per_column",
            "extract_annotation"
        ]
    )

    prefix = kwargs["prefix"]

    # prepare output dictionary with result files
    outcfg = {
        "alignment_file": prefix + ".a2m",
        "statistics_file": prefix + "_alignment_statistics.csv",
        "sequence_file": prefix + ".fa",
        "specieslist_file": prefix + "_specieslist.csv",
        "focus_mode": True
    }

    # check if stage should be skipped and if so, return
    if kwargs.get("skip", False):
        # get information about sequence range from existing file

        # TODO: add proper exception handling here if any of
        # the following goes wrong:
        # 1) check if sequence file is valid
        # 2) check if region is valid
        # 3) check if alignment is valid
        with open(outcfg["sequence_file"]) as f:
            seq_id, seq = next(read_fasta(f))
            start, end = seq_id.split("/", maxsplit=1)[1].split("-")

        outcfg["segments"] = [
            create_segment(kwargs["sequence_id"], start, end)
        ]

        return outcfg

    # Otherwise, now run the protocol...
    # make sure output directory exists
    # TODO: Exception handling here if this fails
    create_prefix_folders(prefix)

    # make sure search sequence is defined and load it
    full_seq_file, (full_seq_id, full_seq) = fetch_sequence(
        kwargs["sequence_id"],
        kwargs["sequence_file"],
        kwargs["sequence_download_url"],
        kwargs["prefix"] + "_full.fa"
    )

    # cut sequence to target region and save in sequence_file
    # (this is the main sequence file used downstream)
    region, cut_seq = cut_sequence(
        full_seq,
        kwargs["sequence_id"],
        kwargs["region"],
        kwargs["first_index"],
        outcfg["sequence_file"]
    )

    # define a single protein segment based on target sequence
    outcfg["segments"] = [
        create_segment(kwargs["sequence_id"], *region)
    ]

    # run jackhmmer... allow to reuse pre-exisiting
    # Stockholm alignment file here
    if not kwargs["reuse_alignment"]:
        # run iterative jackhmmer search
        check_required(kwargs, [kwargs["database"]])

        seq_threshold, domain_threshold = search_thresholds(
            kwargs["use_bitscores"],
            kwargs["sequence_threshold"],
            kwargs["domain_threshold"],
            len(cut_seq)
        )

        ali = at.run_jackhmmer(
            query=outcfg["sequence_file"],
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
        ali_raw_file = ali.alignment
    else:
        ali_raw_file = prefix + ".sto"

        if not file_not_empty(ali_raw_file):
            raise ResourceError(
                "Tried to reuse alignment, but file does not exist "
                "or have any contents: {}".format(ali_raw_file)
            )

    # read in stockholm format (with full annotation)
    with open(ali_raw_file) as a:
        ali_raw = Alignment.from_file(a, "stockholm")

    # save annotation in sequence headers (species etc.)
    if kwargs["extract_annotation"]:
        annotation = extract_header_annotation(ali_raw)
        annotation.to_csv(prefix + "_annotation.csv", index=False)

    ali_raw_fasta_file = prefix + "_raw.fasta"
    with open(ali_raw_fasta_file, "w") as f:
        ali_raw.write(f, "fasta")

    # center alignment around focus/search sequence
    focus_cols = np.array([c != "-" for c in ali_raw[0]])
    focus_ali = ali_raw.select(columns=focus_cols)
    focus_fasta_file = prefix + "_raw_focus.fasta"
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
    # TODO: come up with something more clever here than fixed width
    # (e.g. use 95% quantile of length distribution as reference point)
    min_cov = kwargs["minimum_coverage"]
    if min_cov is not None:
        if isinstance(min_cov, int):
            min_cov /= 100

        keep_seqs = (1 - ali.count("-", axis="seq")) >= min_cov
        ali = ali.select(sequences=keep_seqs)

    # Calculate frequencies, conservation and identity to query
    # on final alignment (except for lowercase modification)
    describe_seq_identities(ali, target_seq_index=0).to_csv(
        prefix + "_identities.csv", float_format="%.3f", index=False
    )

    describe_frequencies(ali, region[0], target_seq_index=0).to_csv(
        prefix + "_frequencies.csv", float_format="%.3f", index=False
    )

    describe_coverage(
        ali, prefix, region[0], kwargs["max_gaps_per_column"]
    ).to_csv(
        prefix + "_alignment_statistics.csv", float_format="%.3f",
        index=False
    )

    # Make columns with too many gaps lowercase
    max_gaps = kwargs["max_gaps_per_column"]
    if max_gaps is not None:
        if isinstance(max_gaps, int):
            max_gaps /= 100

        lc_cols = ali.count(ali._match_gap, axis="pos") >= max_gaps
        ali = ali.lowercase_columns(lc_cols)

    final_a2m_file = prefix + ".a2m"
    with open(final_a2m_file, "w") as f:
        ali.write(f, "fasta")

    # TODO: visualize statistics
    # TODO: how to merge alignment statistics and plots across
    # different runs?

    # TODO: dump config to YAML file for debugging/logging?

    # run callback function if given (e.g. to merge alignment
    # or update database status)
    if kwargs.get("callback", None) is not None:
        kwargs["callback"]({**kwargs, **outcfg})

    # in the end, return both alignment object (if in memory)
    # and path to final alignment file
    return outcfg, ali


# list of available alignment protocols
PROTOCOLS = {
    # standard buildali protocol (iterative hmmer search)
    "standard": standard,

    # start from an existing (external) alignment
    "external": external,
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
        skip: If True, only return stage results but do
        not run actual calculation.

    Returns
    -------
    Alignment
    Dictionary with results of stage in following fields:
        alignment_file
        statistics_file
        sequence_file
        search_sequence_file
        sequence_id
        segments
        focus_mode
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
