"""
Protein sequence alignment creation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

import evcouplings.align.tools as at
from evcouplings.utils.config import check_required
from evcouplings.utils.system import (
    create_prefix_folders, get, file_not_empty, ResourceError
)


# list of available alignment protocols
PROTOCOLS = {
    "standard": standard
}


def fetch_sequence(**kwargs):
    """
    Get sequence.

    Parameters
    ----------

    Returns
    -------
    str:
        Path of file with sequence
    """
    check_required(
        kwargs,
        ["prefix", "sequence_id", "sequence_file"]
    )

    if kwargs["sequence_file"] is None:
        # Only check sequence download URL if we actually
        # need to download something
        check_required(kwargs, ["sequence_download_url"])

        outfile = kwargs["prefix"] + ".fa"
        get(
            kwargs["sequence_download_url"].format(kwargs["sequence_id"]),
            outfile,
            allow_redirects=True
        )
    else:
        # if we have sequence file, pass it through
        outfile = kwargs["sequence_file"]

    # also make sure input file has something in it
    if not file_not_empty(outfile):
        raise ResourceError(
            "Input sequence missing: {}".format(outfile)
        )

    return outfile


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
        ["prefix", "sequence_id", "sequence_file"]
    )

    prefix = kwargs["prefix"]
    create_prefix_folders(prefix)

    # prepare output dictionary with result files
    outcfg = {
        "alignment_file": prefix + ".a2m",
        "statistics_file": prefix + "_alignment_statistics.csv"
    }

    # make sure search sequence is defined
    outcfg["sequence_file"] = fetch_sequence(**kwargs)

    return outcfg

    # run jackhmmer... allow to plonk in pre-exisiting sto file here

    # parse to a2m

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
