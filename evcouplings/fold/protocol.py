"""
Protocols for predicting protein 3D structure from ECs

Authors:
  Thomas A. Hopf
  Anna G. Green (complex_dock)
"""

from os import path
from math import ceil
import billiard as mp
from functools import partial
import shutil

import pandas as pd

from evcouplings.align.alignment import (
    read_fasta, parse_header
)
from evcouplings.couplings.mapping import Segment
from evcouplings.compare.pdb import ClassicPDB

from evcouplings.fold.tools import (
    run_psipred, read_psipred_prediction,
    run_maxcluster_cluster, run_maxcluster_compare
)
from evcouplings.fold.cns import cns_dgsa_fold
from evcouplings.fold.filter import secstruct_clashes
from evcouplings.fold.ranking import dihedral_ranking
from evcouplings.fold.haddock import haddock_dist_restraint
from evcouplings.fold.restraints import docking_restraints
from evcouplings.utils.config import (
    check_required, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources,
    valid_file, insert_dir, temp
)
from evcouplings.visualize.pymol import pymol_secondary_structure


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
            output_dir = path.join(path.dirname(prefix), "psipred")

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


def compare_models_maxcluster(experiments, predictions, norm_by_intersection=True,
                              distance_cutoff=None, binary="maxcluster"):
    """
    Compare predicted models to a set of experimental structures
    using maxcluster
    
    Parameters
    ----------
    experiments : list(str)
        Paths to files with experimental structures
    predictions : list(str)
        Paths to files with predicted structures
    norm_by_intersection : bool, optional (default: True)
        If True, use the number of positions that exist
        in both experiment and predictions for normalizing
        TM scores (assumes all predicted structures have the
        same positions). If False, use length of experimental
        structure.
    distance_cutoff : float, optional (default: None)
        Distance cutoff for MaxSub search (-d option of maxcluster).
        If None, will use maxcluster auto-calibration.
    binary : str, optional (default: "maxcluster")
        Path to maxcluster binary

    Returns
    -------
    full_result : pandas.DataFrame
        Comparison results across all experimental structures
    single_results : dict
        Mapping from experimental structure filename to
        a pandas.DataFrame containing the comparison
        result for that particular structure.
    """
    # determine list of positions in a structure
    # (maxcluster can only handle single model, single chain
    # structures, so we check that here and fail otherwise)
    def _determine_pos(filename):
        structure = ClassicPDB.from_file(filename)
        if len(structure.model_to_chains) == 0:
            raise InvalidParameterError(
                "Structure contains no model (is empty): " +
                filename +
                " - please verify that no problems occurred during structure mapping"
            )
        elif len(structure.model_to_chains) > 1:
            raise InvalidParameterError(
                "Structure contains more than one model: " +
                filename
            )

        model = list(structure.model_to_chains.keys())[0]
        chains = structure.model_to_chains[model]
        if len(chains) != 1:
            raise InvalidParameterError(
                "Structure must contain exactly one chain, but contains: " +
                ",".join(chains)
            )
        chain_name = chains[0]
        chain = structure.get_chain(chain_name, model)
        return chain.residues.id.astype(str).values, chain

    # remove alternative atom locations since maxcluster
    # can only handle one unique atoms
    def _eliminate_altloc(chain):
        # if multiple locations, select the one with the
        # highest occupancy
        chain.coords = chain.coords.loc[
            chain.coords.groupby(
                ["residue_index", "atom_name"]
            ).occupancy.idxmax()
        ]

        # save cut chain to temporary file
        temp_filename = temp()
        with open(temp_filename, "w") as f:
            chain.to_file(f)
        return temp_filename

    # check we have at least one prediction
    if len(predictions) == 0:
        raise InvalidParameterError(
            "Need at least one predicted structure."
        )

    # determine positions in predicted structure from first model
    pred_pos, _ = _determine_pos(predictions[0])

    # collect results of all comparisons here
    full_result = pd.DataFrame()
    single_results = {}

    for exp_file in experiments:
        # determine what number of position to normalize
        # TM score over (either experiment, or only positions
        # that were modelled and are also present in experiment)
        exp_pos, exp_chain = _determine_pos(exp_file)

        # remove alternative atom locations
        exp_file_cleaned = _eliminate_altloc(exp_chain)

        # compute set of positions both in prediction and expeirment
        joint_pos = set(exp_pos).intersection(pred_pos)

        if norm_by_intersection:
            normalization_length = len(joint_pos)
        else:
            normalization_length = len(exp_pos)

        # run comparison
        comp = run_maxcluster_compare(
            predictions, exp_file_cleaned,
            normalization_length=normalization_length,
            distance_cutoff=distance_cutoff, binary=binary
        )

        # store lengths of experiment, prediction,
        # and what was used for computing TM scores
        comp.loc[:, "filename_experimental"] = exp_file
        comp.loc[:, "L_experiment"] = len(exp_pos)
        comp.loc[:, "L_prediction"] = len(pred_pos)
        comp.loc[:, "L_joint"] = len(joint_pos)
        comp.loc[:, "L_normalization"] = normalization_length

        comp = comp.sort_values("tm", ascending=False)
        single_results[exp_file] = comp

        full_result = pd.concat([full_result, comp])

    return full_result, single_results


def maxcluster_clustering_table(structures, binary):
    """
    Create table of clustering results for all possible
    maxcluster clustering modes

    Parameters
    ----------
    structures : list(str)
        List of structure files
    binary : str, optional (default: "maxcluster")
        Path to maxcluster binary
            
    Returns
    -------
    pandas.DataFrame
        Table with clustering results for all structures
    """
    clust_all = None
    for method in ["single", "average", "maximum", "pairs_min", "pairs_abs"]:
        # run maxcluster with current clustering method
        clust = run_maxcluster_cluster(
            structures, method=method, binary=binary
        )

        # rename columns to contain current clustering method
        clust = clust.rename(
            columns={
                "cluster": "cluster_" + method,
                "cluster_size": "cluster_size_" + method
            }
        )

        # join into one big table
        if clust_all is None:
            clust_all = clust
        else:
            clust_all = clust_all.merge(
                clust, on="filename", how="outer"
            )

    return clust_all


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

        * sec_struct_file
        * folding_ec_file
        * folded_structure_files
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
            "num_models", "psipred", "cpu", "remapped_pdb_files",
            "cleanup",
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

    # make pymol secondary structure assignment script
    outcfg["secondary_structure_pml_file"] = prefix + "_ss_draw.pml"
    pymol_secondary_structure(
        residues, outcfg["secondary_structure_pml_file"]
    )

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

    # store structures in an auxiliary subdirectory, after folding
    # final models will be moved to main folding dir. Depending
    # on cleanup setting, the aux directory will be removed
    aux_prefix = insert_dir(prefix, "aux", rootname_subdir=False)
    aux_dir = path.dirname(aux_prefix)

    folding_runs = [
        (job_ecs, aux_prefix + job_suffix)
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

    # make double sure that the pool is cleaned up,
    # or SIGTERM upon exit will interfere with
    # interrupt signal interception
    pool.close()
    pool.join()

    # merge result dictionaries into one dict
    folded_files = {
        k: v for subres in results for k, v in subres.items()
    }

    # move structures from aux into main folding dir
    fold_dir = path.dirname(prefix)
    prediction_files = []
    for name, file_path in folded_files.items():
        # move file (use copy to allow overwriting)
        shutil.copy(file_path, fold_dir)

        # update file path to main folding dir,
        # and put in a flat list of result files
        prediction_files.append(
            file_path.replace(aux_prefix, prefix)
        )

    outcfg["folded_structure_files"] = prediction_files

    # remove aux dir if cleanup is requested
    if kwargs["cleanup"]:
        shutil.rmtree(aux_dir)

    # apply ranking to predicted models
    ranking = dihedral_ranking(prediction_files, residues)

    # apply clustering (all available methods), but only
    # if we have something to cluster
    if len(prediction_files) > 1:
        clustering = maxcluster_clustering_table(
            prediction_files, binary=kwargs["maxcluster"]
        )

        # join ranking with clustering
        ranking = ranking.merge(clustering, on="filename", how="left")

    # sort by score (best models first)
    ranking = ranking.sort_values(by="ranking_score", ascending=False)

    # store as file
    outcfg["folding_ranking_file"] = prefix + "_ranking.csv"
    ranking.to_csv(outcfg["folding_ranking_file"], index=False)

    # apply comparison to existing structures
    if kwargs["remapped_pdb_files"] is not None and len(kwargs["remapped_pdb_files"]) > 0:
        experimental_files = kwargs["remapped_pdb_files"]

        comp_all, comp_singles = compare_models_maxcluster(
            list(experimental_files.keys()), prediction_files,
            norm_by_intersection=True, distance_cutoff=None,
            binary=kwargs["maxcluster"]
        )

        # merge with ranking and save
        comparison = ranking.merge(
            comp_all, on="filename", how="left"
        ).sort_values(by="tm", ascending=False)
        outcfg["folding_comparison_file"] = prefix + "_comparison.csv"
        comparison.to_csv(outcfg["folding_comparison_file"], index=False)

        # also store comparison to structures in individual files
        ind_comp_files = {}
        for filename, comp_single in comp_singles.items():
            comparison_s = ranking.merge(
                comp_single, on="filename", how="left"
            ).sort_values(by="tm", ascending=False)
            basename = path.splitext(path.split(filename)[1])[0]
            ind_file = path.join(fold_dir, basename + ".csv")

            # map back to original key from remapped_pdb_files as a key for this list
            ind_comp_files[ind_file] = experimental_files[filename]
            comparison_s.to_csv(ind_file, index=False)

        outcfg["folding_individual_comparison_files"] = ind_comp_files

    return outcfg


def complex_dock(**kwargs):
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

        * docking_restraints_files
    """
    check_required(
        kwargs,
        [
            "prefix", "ec_file",
            "segments", "dock_probability_cutoffs",
            "dock_lowest_count", "dock_highest_count", "dock_increase",
        ]
    )

    prefix = kwargs["prefix"]
    outcfg = {}

    # make sure output directory exists
    create_prefix_folders(prefix)

    verify_resources(
        "EC file does not exist and/or is empty",
        kwargs["ec_file"]
    )

    ecs_all = pd.read_csv(kwargs["ec_file"])
    ecs_dock = ecs_all.query("segment_i != segment_j")

    # define the sub-runs ...
    folding_runs = []

    # ... based on mixture model probability
    cutoffs = kwargs["dock_probability_cutoffs"]

    if cutoffs is not None and "probability" in ecs_dock.columns:
        if not isinstance(cutoffs, list):
            cutoffs = [cutoffs]

        for c in cutoffs:
            sig_ecs = ecs_dock.query("probability >= @c")
            if len(sig_ecs) > 0:
                folding_runs.append(
                    (sig_ecs,
                     "_significant_ECs_{}_restraints.tbl".format(c))
                )

    # ... and on simple EC counts/bins
    flc = kwargs["dock_lowest_count"]
    fhc = kwargs["dock_highest_count"]
    fi = kwargs["dock_increase"]
    if flc is not None and fhc is not None and fi is not None:
        num_sites = len(set(ecs_dock.i.unique())) + len(set(ecs_dock.j.unique()))

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
                ecs_dock.iloc[:c],
                "_{}_restraints.tbl".format(c)
            )
            for c in range(lowest, highest + 1, step)
        ]

    outcfg["docking_restraint_files"] = []
    for job_ecs, job_suffix in folding_runs:
        job_filename = prefix + job_suffix
        docking_restraints(job_ecs, job_filename, haddock_dist_restraint)
        outcfg["docking_restraint_files"].append(job_filename)

    return outcfg


# list of available folding protocols
PROTOCOLS = {
    # standard EVfold protocol
    "standard": standard,

    # create docking restraint for complexes
    "complex_dock": complex_dock
}


def run(**kwargs):
    """
    Run folding protocol

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
