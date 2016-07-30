"""
Protein sequence alignment creation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

import evcouplings.align.tools as at
from evcouplings.align.alignment import read_fasta, write_fasta
from evcouplings.utils.config import check_required
from evcouplings.utils.system import (
    create_prefix_folders, get, file_not_empty, ResourceError
)


# list of available alignment protocols
PROTOCOLS = {
    "standard": standard
}


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
    ValueError
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
            raise ValueError(
                "Illegal sequence range: "
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


def search_jackhmmer(**kwargs):
    """
    Get alignment by homology search from query
    """
    check_required(kwargs, [""])

    x = at.run_jackhmmer(
        "../../test_data/RASH_HUMAN.fa", "../../databases/pdb_seqres.txt", "bla/test",
        True, 100, 200, iterations=1,
        binary="../../software/hmmer-3.1b2-macosx-intel/binaries/jackhmmer",
        checkpoints_hmm=False
    )

    return x


def modify_alignment(config):
    """
    Prepare alignment to be ready for EC calculation
    """
    return


def describe(config):
    """
    Get parameters of alignment such as gaps, coverage,
    conservation
    """
    return


def dummy(**kwargs):
    """
    Dummy protocol if stage is not run
    """
    return


def standard(**kwargs):
    """
    Standard buildali4 workflow

    Parameters
    ----------

    Returns
    -------

    """
    check_required(
        kwargs,
        [
            "prefix", "sequence_id", "sequence_file",
            "sequence_download_url", "reuse_alignment",
            "region", "first_index"
        ]
    )

    prefix = kwargs["prefix"]
    create_prefix_folders(prefix)

    # prepare output dictionary with result files
    outcfg = {
        "alignment_file": prefix + ".a2m",
        "statistics_file": prefix + "_alignment_statistics.csv"
    }

    # make sure search sequence is defined and load it
    outcfg["sequence_file"], (full_seq_id, full_seq) = fetch_sequence(
        kwargs["sequence_id"], kwargs["sequence_file"],
        kwargs["sequence_download_url"],
        kwargs["prefix"] + ".fa"
    )

    # cut sequence to target region
    # TODO: store this in output directory as region
    region, cut_seq = cut_sequence(
        full_seq,
        kwargs["sequence_id"],
        kwargs["region"],
        kwargs["first_index"],
        kwargs["prefix"] + "_searchseq.fa"
    )

    print(cut_seq)
    return outcfg

    # run jackhmmer... allow to plonk in pre-exisiting sto/a2m file here
    # make sure bitscores are expanded correctly
    if kwargs["reuse_alignment"] is None:
        # TODO: update outcfg?
        # ali = search_jackhmmer(kwargs)
        pass
    else:
        pass

    # parse to a2m... autodetect which format we are starting from

    # apply id filter, gap threshold

    # set correct headers (make ready for plmc)

    # generate specieslist (copy into sequence headers?)

    # output gap statistics, conservation of columns

    # visualize distributions?

    # if realign:
    #    realign(config)

    # TODO: how to get alignment statistics and plots?
    # (modularize this into an independent function too)

    # in the end, return both alignment object (if in memory)
    # and path to final alignment file
    return outcfg


def run(**kwargs):
    """
    Run alignment protocol to generate multiple sequence
    alignment from input sequence.

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: Alignment protocol to run
        prefix: Output prefix for all generated files

    Returns
    -------
    Dictionary with results of stage in following fields:
        alignment_file
        statistics_file
        sequence_file
        search_sequence_file
        sequence_id
        segments
    """
    check_required(kwargs, ["protocol"])

    return PROTOCOLS[kwargs["protocol"]](**kwargs)
