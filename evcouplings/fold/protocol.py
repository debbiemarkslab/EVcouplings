"""
Protocols for predicting protein 3D structure from ECs

Authors:
  Thomas A. Hopf
"""

from os import path
from math import ceil
import multiprocessing as mp
from functools import partial
import pandas as pd

from evcouplings.align.alignment import (
    read_fasta, parse_header
)
from evcouplings.couplings.mapping import Segment
from evcouplings.fold.tools import (
    run_psipred, read_psipred_prediction
)
from evcouplings.fold.cns import cns_dgsa_fold
from evcouplings.fold.filter import secstruct_clashes
from evcouplings.utils.config import (
    check_required, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources,
    valid_file, insert_dir
)


def secondary_structure(**kwargs):
    """
    Predict or load secondary structure for an
    input sequence

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    residues : pandas.DataFrame
        Table with sequence and secondary structure
        in columns i, A_i and sec_struct_3state
    """
    check_required(
        kwargs,
        [
            "prefix", "target_sequence_file",
            "segments", "sec_struct_method",
            "sec_struct_file", "psipred",
        ]
    )

    prefix = kwargs["prefix"]
    create_prefix_folders(prefix)

    secstruct_file = kwargs["sec_struct_file"]
    if secstruct_file is not None:
        verify_resources(
            "Secondary structure prediction file does not exist/is empty",
            secstruct_file
        )
        residues = pd.read_csv(secstruct_file)
    else:
        # make sure target sequence file is there so we can
        # predict secondary structure
        target_seq_file = kwargs["target_sequence_file"]
        verify_resources(
            "Sequence file does not exist/is empty", target_seq_file
        )

        # we need to figure out what the index of the first residue
        # in the target sequence is; obtain first index from segment
        # information if possible
        if kwargs["segments"] is not None:
            s = Segment.from_list(kwargs["segments"][0])
            first_index = s.region_start
        else:
            # otherwise try to get it from sequence file
            first_index = None

            with open(target_seq_file) as f:
                header, _ = next(read_fasta(f))
                if header is not None:
                    _, first_index, _ = parse_header(header)

                # if we cannot identify first index from header,
                # do not make guesses but fail
                if first_index is None:
                    raise InvalidParameterError(
                        "Could not unambiguously identify sequence range from "
                        "FASTA header, needs to specified as id/start-end: {}".format(
                            header
                        )
                    )

        # finally, run secondary structure prediction
        if kwargs["sec_struct_method"] == "psipred":
            # store psipred output in a separate directory
            output_dir = path.join(prefix, "psipred")

            # run psipred
            ss2_file, horiz_file = run_psipred(
                target_seq_file, output_dir, binary=kwargs["psipred"]
            )

            # parse output, renumber to first index
            residues = read_psipred_prediction(
                horiz_file, first_index=first_index
            )
        else:
            raise InvalidParameterError(
                "Secondary structure prediction method not implemented: "
                "{}. Valid choices: psipred".format(kwargs["sec_struct_method"])
            )

    # return predicted table
    return residues


def standard(**kwargs):
    """
    Protocol:
    Predict 3D structure from evolutionary couplings

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        sec_struct_file
        folding_ec_file
        folded_structure_files
    """
    check_required(
        kwargs,
        [
            "prefix", "engine", "ec_file", "target_sequence_file",
            "segments", "folding_config_file", "cut_to_alignment_region",
            "sec_struct_method", "reuse_sec_struct",
            "sec_struct_file", "filter_sec_struct_clashes",
            "min_sequence_distance", "fold_probability_cutoffs",
            "fold_lowest_count", "fold_highest_count", "fold_increase",
            "num_models", "psipred", "cpu",
        ]
    )

    prefix = kwargs["prefix"]

    # make sure output directory exists
    create_prefix_folders(prefix)

    outcfg = {
        "folding_ec_file": prefix + "_CouplingScores_with_clashes.csv",
        "sec_struct_file": prefix + "_secondary_structure.csv",
    }

    # get secondary structure prediction
    # check if we should (and can) reuse output file from previous run
    if kwargs["reuse_sec_struct"] and valid_file(outcfg["sec_struct_file"]):
        residues = pd.read_csv(outcfg["sec_struct_file"])
    else:
        residues = secondary_structure(**kwargs)

    # load ECs and filter for long-range pairs
    verify_resources(
        "EC file does not exist", kwargs["ec_file"]
    )
    ecs_all = pd.read_csv(kwargs["ec_file"])
    ecs = ecs_all.query("abs(i - j) > {}".format(
        kwargs["min_sequence_distance"])
    )

    # find secondary structure clashes
    ecs = secstruct_clashes(ecs, residues)
    ecs.to_csv(outcfg["folding_ec_file"], index=False)

    # if requested, filter clashes out before folding
    if kwargs["filter_sec_struct_clashes"]:
        ecs_fold = ecs.loc[~ecs.ss_clash]
    else:
        ecs_fold = ecs

    # cut modelled region to aligned region, if selected
    if kwargs["cut_to_alignment_region"]:
        segments = kwargs["segments"]
        # infer region from segment positions if we have it
        if segments is not None:
            positions = Segment.from_list(segments[0]).positions
        else:
            # otherwise get from EC values (could be misleading if
            # EC list is truncated, so only second option)
            positions = set(ecs.i.unique()).union(ecs.j.unique())

        # limit modelled positions to covered region
        first_pos, last_pos = min(positions), max(positions)
        residues.loc[:, "in_model"] = False
        residues.loc[
            (residues.i >= first_pos) & (residues.i <= last_pos),
            "in_model"
        ] = True
    else:
        # otherwise include all positions in model
        residues.loc[:, "in_model"] = True

    # save secondary structure prediction
    residues.to_csv(outcfg["sec_struct_file"], index=False)

    # only use the residues that will be in model for folding
    residues_fold = residues.loc[residues.in_model]

    # after all the setup, now fold the structures...
    # to speed things up, parallelize this to the number of
    # available CPUs
    num_procs = kwargs["cpu"]
    if num_procs is None:
        num_procs = 1

    # first define all the sub-runs...
    folding_runs = []

    # ... based on mixture model probability
    cutoffs = kwargs["fold_probability_cutoffs"]
    if cutoffs is not None and "probability" in ecs_fold.columns:
        if not isinstance(cutoffs, list):
            cutoffs = [cutoffs]

        for c in cutoffs:
            sig_ecs = ecs_fold.query("probability >= @c")
            if len(sig_ecs) > 0:
                folding_runs.append(
                    (sig_ecs,
                     "_significant_ECs_{}".format(c))
                )

    # ... and on simple EC counts/bins
    flc = kwargs["fold_lowest_count"]
    fhc = kwargs["fold_highest_count"]
    fi = kwargs["fold_increase"]
    if flc is not None and fhc is not None and fi is not None:
        num_sites = len(
            set.union(set(ecs.i.unique()), set(ecs.j.unique()))
        )

        # transform fraction of number of sites into discrete number of ECs
        def _discrete_count(x):
            if isinstance(x, float):
                x = ceil(x * num_sites)
            return int(x)

        # range of plots to make
        lowest = _discrete_count(flc)
        highest = _discrete_count(fhc)
        step = _discrete_count(fi)

        # append to list of jobs to run
        folding_runs += [
            (
                ecs_fold.iloc[:c],
                "_{}".format(c)
            )
            for c in range(lowest, highest + 1, step)
        ]

    # set up method to drive the folding of each job
    method = kwargs["engine"]

    # put folding output in a subdirectory
    # update paths in folding jobs accordingly
    job_path = insert_dir(prefix, "fold")
    create_prefix_folders(job_path)

    folding_runs = [
        (job_ecs, job_path + job_suffix)
        for (job_ecs, job_suffix) in folding_runs
    ]

    if method == "cns_dgsa":
        folder = partial(
            cns_dgsa_fold,
            residues_fold,
            config_file=kwargs["folding_config_file"],
            num_structures=kwargs["num_models"],
            log_level=None,
            binary=kwargs["cns"]
        )
    else:
        raise InvalidParameterError(
            "Invalid folding engine: {} ".format(method) +
            "Valid selections are: cns_dgsa"
        )

    # then apply folding function to each sub-run
    pool = mp.Pool(processes=num_procs)
    results = pool.starmap(folder, folding_runs)

    # merge result dictionaries into one dict
    outcfg["folded_structure_files"] = {
        k: v for subres in results for k, v in subres.items()
    }

    return outcfg


# list of available folding protocols
PROTOCOLS = {
    # standard EVfold protocol
    "standard": standard,
}


def run(**kwargs):
    """
    Run mutation protocol

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: Folding protocol to run
        prefix: Output prefix for all generated files

    Returns
    -------
    outcfg : dict
        Output configuration of stage
        (see individual protocol for fields)
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
