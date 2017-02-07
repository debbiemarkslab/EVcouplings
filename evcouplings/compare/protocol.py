"""
EC to 3D structure comparison protocols/workflows.

Authors:
  Thomas A. Hopf
"""

from copy import deepcopy
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    read_config_file
)

from evcouplings.utils.system import (
    create_prefix_folders, valid_file, verify_resources
)
from evcouplings.compare.pdb import load_structures
from evcouplings.compare.distances import (
    intra_dists, multimer_dists, _prepare_chain
)
from evcouplings.compare.sifts import SIFTS, SIFTSResult
from evcouplings.compare.ecs import (
    add_distances, coupling_scores_compared
)
from evcouplings.visualize import pairs, misc


def identify_structures(**kwargs):
    """
    Identify set of 3D structures for comparison

    Parameters
    ----------
    # TODO

    Returns
    -------
    # TODO
    """
    def _filter_by_id(x, id_list):
        x = deepcopy(x)
        x.hits = x.hits.loc[
            x.hits.pdb_id.isin(id_list)
        ]
        return x

    check_required(
        kwargs,
        [
            "prefix", "pdb_ids", "compare_multimer",
            "max_num_hits", "max_num_structures",
            "pdb_mmtf_dir",
            "sifts_mapping_table", "sifts_sequence_db",
            "by_alignment", "alignment_min_overlap",
            "sequence_id", "sequence_file", "region",
            "use_bitscores", "domain_threshold",
            "sequence_threshold", "jackhmmer",
        ]
    )

    # get SIFTS mapping object/sequence DB
    s = SIFTS(
        kwargs["sifts_mapping_table"],
        kwargs["sifts_sequence_db"]
    )

    reduce_chains = not kwargs["compare_multimer"]

    # determine if we need to find structures
    # by sequence search or just fetching
    # based on Uniprot/PDB identifier
    if kwargs["by_alignment"]:
        sifts_map = s.by_alignment(
            reduce_chains=reduce_chains,
            min_overlap=kwargs["alignment_min_overlap"],
            **kwargs
        )
    else:
        sifts_map = s.by_uniprot_id(
            kwargs["sequence_id"], reduce_chains=reduce_chains
        )

        if kwargs["pdb_ids"] is not None:
            pdb_ids = kwargs["pdb_ids"]

            # make sure we have a list of PDB IDs
            if not isinstance(pdb_ids, list):
                pdb_ids = [pdb_ids]

            pdb_ids = [x.lower() for x in pdb_ids]

            sifts_map = _filter_by_id(sifts_map, pdb_ids)

    # limit number of hits and structures
    if kwargs["max_num_hits"] is not None:
        sifts_map.hits = sifts_map.hits.iloc[:kwargs["max_num_hits"]]

    if kwargs["max_num_structures"] is not None:
        keep_ids = sifts_map.hits.pdb_id.unique()
        keep_ids = keep_ids[:kwargs["max_num_structures"]]
        sifts_map = _filter_by_id(sifts_map, keep_ids)

    return sifts_map


def plot_cm(ecs, d_intra, d_mult, output_file=None, boundaries="union", secstruct=None):
    """
    #TODO
    """
    with misc.plot_context("Arial"):
        fig = plt.figure(figsize=(8, 7))
        pairs.plot_contact_map(
            ecs, d_intra, d_mult,
            secstruct_style={"helix_turn_length": 4, "width": 0.5},
            secondary_structure=secstruct,
            show_secstruct=True,
            margin=10,
            boundaries=boundaries
        )

        plt.suptitle("{} evolutionary couplings".format(len(ecs)), fontsize=14)

        if output_file is not None:
            plt.savefig(output_file, bbox_inches="tight")
            # plt.close(fig)  # TODO: reenable


def standard(**kwargs):
    """
    Protocol:
    Compare ECs for single proteins (or domains)
    to 3D structure information

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        # TODO
    """
    check_required(
        kwargs,
        [
            "prefix", "ec_file", "min_sequence_distance",
            "pdb_mmtf_dir", "atom_filter", "compare_multimer",
            "distance_cutoff", "plot_probability_cutoffs",
            "compare_multimer", "plot_probability_cutoffs",
            "boundaries", "plot_lowest_count",
            "plot_highest_count", "plot_increase",
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        "ec_file_compared_all": prefix + "_CouplingScoresCompared_all.csv",
        "ec_file_compared_longrange": prefix + "_CouplingScoresCompared_longrange.csv",
        "pdb_structure_hits": prefix + "_structure_hits.csv",
        "distmap_monomer": prefix + "_distance_map_monomer",
        "distmap_multimer": prefix + "_distance_map_multimer",
    }

    # make sure EC file exists
    verify_resources(
        "EC file does not exist",
        kwargs["ec_file"]
    )

    # make sure output directory exists
    # TODO: Exception handling here if this fails
    create_prefix_folders(prefix)

    # Step 1: Identify 3D structures for comparison
    sifts_map = identify_structures(**{
        **kwargs,
        "prefix": prefix + "/compare_find"
    })

    sifts_map.hits.to_csv(
        outcfg["pdb_structure_hits"], index=False
    )

    # Step 2: Compute distance maps
    # load all structures at one
    structures = load_structures(
        sifts_map.hits.pdb_id,
        kwargs["pdb_mmtf_dir"]
    )

    # compute distance maps and save
    d_intra = intra_dists(
        sifts_map, structures, atom_filter=kwargs["atom_filter"],
        output_prefix=prefix + "/compare_distmap_intra"
    )
    d_intra.to_file(outcfg["distmap_monomer"])

    # compute multimer distances, if requested
    # note that d_multimer can be None if there
    # are no structures with multiple chains
    if kwargs["compare_multimer"]:
        d_multimer = multimer_dists(
            sifts_map, structures, atom_filter=kwargs["atom_filter"],
            output_prefix=prefix + "/compare_distmap_multimer"
        )
    else:
        d_multimer = None

    if d_multimer is not None:
        d_multimer.to_file(outcfg["distmap_multimer"])
    else:
        outcfg["distmap_multimer"] = None

    # Step 3: Compare ECs to distance maps
    ec_table = pd.read_csv(kwargs["ec_file"])

    for out_file, min_seq_dist in [
        (outcfg["ec_file_compared_longrange"], kwargs["min_sequence_distance"]),
        (outcfg["ec_file_compared_all"], 0),
    ]:
        coupling_scores_compared(
            ec_table, d_intra, d_multimer,
            dist_cutoff=kwargs["distance_cutoff"],
            output_file=out_file,
            score="cn",
            min_sequence_dist=min_seq_dist
        )

    # Step 4: Make contact map plots
    cm_files = []

    ecs_longrange = ec_table.query(
        "abs(i - j) >= {}".format(kwargs["min_sequence_distance"])
    )

    # get secondary structure (for now from first hit)
    if len(sifts_map.hits) > 0:
        hit = sifts_map.hits.iloc[0]
        res_ss = structures[hit.pdb_id].get_chain(
            hit.pdb_chain, model=0
        ).remap(
            sifts_map.mapping[hit.mapping_index]
        ).residues
    else:
        res_ss = None

    # based on significance cutoff
    if kwargs["plot_probability_cutoffs"]:
        cutoffs = kwargs["plot_probability_cutoffs"]
        if not isinstance(cutoffs, list):
            cutoffs = []

        for c in cutoffs:
            ec_set = ecs_longrange.query("probability >= @c")
            output_file = prefix + "_significant_ECs_{}.pdf".format(c)
            plot_cm(
                ec_set, d_intra, d_multimer,
                output_file=output_file,
                boundaries=kwargs["boundaries"],
                secstruct=res_ss
            )
            cm_files.append(output_file)

    # based on number of long-range ECs
    num_sites = len(set.union(set(ec_table.i.unique()), set(ec_table.j.unique())))

    def _discrete_count(x):
        if isinstance(x, float):
            x = ceil(x * num_sites)
        return int(x)

    lowest = _discrete_count(kwargs["plot_lowest_count"])
    highest = _discrete_count(kwargs["plot_highest_count"])
    step = _discrete_count(kwargs["plot_increase"])

    for c in range(lowest, highest + 1, step):
        ec_set = ecs_longrange.iloc[:c]
        output_file = prefix + "_{}_ECs.pdf".format(c)
        plot_cm(
            ec_set, d_intra, d_multimer,
            output_file=output_file,
            boundaries=kwargs["boundaries"],
            secstruct=res_ss
        )
        cm_files.append(output_file)

    outcfg["contact_map_files"] = cm_files

    return outcfg


# list of available EC comparison protocols
PROTOCOLS = {
    # standard monomer comparison protocol
    "standard": standard,
}


def run(**kwargs):
    """
    Run inference protocol to calculate ECs from
    input sequence alignment.

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: EC protocol to run
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
