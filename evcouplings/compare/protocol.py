"""
EC to 3D structure comparison protocols/workflows.

Authors:
  Thomas A. Hopf
"""

from copy import deepcopy
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt

from evcouplings.align.alignment import (
    read_fasta, parse_header
)
from evcouplings.utils.config import (
    check_required, InvalidParameterError
)

from evcouplings.utils.system import (
    create_prefix_folders, insert_dir, verify_resources,
)
from evcouplings.compare.pdb import load_structures
from evcouplings.compare.distances import (
    intra_dists, multimer_dists, remap_chains
)
from evcouplings.compare.sifts import SIFTS, SIFTSResult
from evcouplings.compare.ecs import coupling_scores_compared
from evcouplings.visualize import pairs, misc


def _identify_structures(**kwargs):
    """
    Identify set of 3D structures for comparison

    Parameters
    ----------
    **kwargs
        See check_required in code below

    Returns
    -------
    SIFTSResult
        Identified structures and residue index mappings
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
            "sequence_threshold",
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

    sifts_map_full = deepcopy(sifts_map)

    # filter ID list down to manually selected PDB entries
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

    return sifts_map, sifts_map_full


def _make_contact_maps(ec_table, d_intra, d_multimer, **kwargs):
    """
    Plot contact maps with all ECs above a certain probability threshold,
    or a given count of ECs

    Parameters
    ----------
    ec_table : pandas.DataFrame
        Full set of evolutionary couplings (all pairs)
    d_intra : DistanceMap
        Computed residue-residue distances inside chain
    d_multimer : DistanceMap
        Computed residue-residue distances between homomultimeric
        chains
    **kwargs
        Further plotting parameters, see check_required in code
        for necessary values.

    Returns
    -------
    cm_files : list(str)
        Paths of generated contact map files
    """
    def plot_cm(ecs, output_file=None):
        """
        Simple wrapper for contact map plotting
        """
        with misc.plot_context("Arial"):
            fig = plt.figure(figsize=(8, 8))
            if kwargs["scale_sizes"]:
                ecs = ecs.copy()
                ecs.loc[:, "size"] = ecs.cn.values / ecs.cn.max()

            pairs.plot_contact_map(
                ecs, d_intra, d_multimer,
                distance_cutoff=kwargs["distance_cutoff"],
                show_secstruct=kwargs["draw_secondary_structure"],
                margin=5,
                boundaries=kwargs["boundaries"]
            )

            plt.suptitle("{} evolutionary couplings".format(len(ecs)), fontsize=14)

            if output_file is not None:
                plt.savefig(output_file, bbox_inches="tight")
                plt.close(fig)

    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
            "plot_probability_cutoffs",
            "boundaries", "plot_lowest_count",
            "plot_highest_count", "plot_increase",
            "draw_secondary_structure",
        ]
    )

    prefix = kwargs["prefix"]

    cm_files = []

    ecs_longrange = ec_table.query(
        "abs(i - j) >= {}".format(kwargs["min_sequence_distance"])
    )

    # based on significance cutoff
    if kwargs["plot_probability_cutoffs"]:
        cutoffs = kwargs["plot_probability_cutoffs"]
        if not isinstance(cutoffs, list):
            cutoffs = [cutoffs]

        for c in cutoffs:
            ec_set = ecs_longrange.query("probability >= @c")
            # only can plot if we have any significant ECs above threshold
            if len(ec_set) > 0:
                output_file = prefix + "_significant_ECs_{}.pdf".format(c)
                plot_cm(ec_set, output_file=output_file)
                cm_files.append(output_file)

    # based on number of long-range ECs

    # identify number of sites in EC model
    num_sites = len(
        set.union(set(ec_table.i.unique()), set(ec_table.j.unique()))
    )

    # transform fraction of number of sites into discrete number of ECs
    def _discrete_count(x):
        if isinstance(x, float):
            x = ceil(x * num_sites)
        return int(x)

    # range of plots to make
    lowest = _discrete_count(kwargs["plot_lowest_count"])
    highest = _discrete_count(kwargs["plot_highest_count"])
    step = _discrete_count(kwargs["plot_increase"])

    # create individual plots
    for c in range(lowest, highest + 1, step):
        ec_set = ecs_longrange.iloc[:c]
        output_file = prefix + "_{}_ECs.pdf".format(c)
        plot_cm(ec_set, output_file=output_file)
        cm_files.append(output_file)

    # give back list of all contact map file names
    return cm_files


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

        * ec_file_compared_all
        * ec_file_compared_all_longrange
        * pdb_structure_hits
        * distmap_monomer
        * distmap_multimer
        * contact_map_files
        * remapped_pdb_files
    """
    check_required(
        kwargs,
        [
            "prefix", "ec_file", "min_sequence_distance",
            "pdb_mmtf_dir", "atom_filter", "compare_multimer",
            "distance_cutoff", "target_sequence_file",
            "scale_sizes",
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        "ec_compared_all_file": prefix + "_CouplingScoresCompared_all.csv",
        "ec_compared_longrange_file": prefix + "_CouplingScoresCompared_longrange.csv",
        "pdb_structure_hits_file": prefix + "_structure_hits.csv",
        "pdb_structure_hits_unfiltered_file": prefix + "_structure_hits_unfiltered.csv",
        # cannot have the distmap files end with "_file" because there are
        # two files (.npy and .csv), which would cause problems with automatic
        # checking if those files exist
        "distmap_monomer": prefix + "_distance_map_monomer",
        "distmap_multimer": prefix + "_distance_map_multimer",
    }

    # make sure EC file exists
    verify_resources(
        "EC file does not exist",
        kwargs["ec_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # store auxiliary files here (too much for average user)
    aux_prefix = insert_dir(prefix, "aux", rootname_subdir=False)
    create_prefix_folders(aux_prefix)

    # Step 1: Identify 3D structures for comparison
    sifts_map, sifts_map_full = _identify_structures(**{
        **kwargs,
        "prefix": aux_prefix,
    })

    # save selected PDB hits
    sifts_map.hits.to_csv(
        outcfg["pdb_structure_hits_file"], index=False
    )

    # also save full list of hits
    sifts_map_full.hits.to_csv(
        outcfg["pdb_structure_hits_unfiltered_file"], index=False
    )

    # Step 2: Compute distance maps

    # load all structures at once
    structures = load_structures(
        sifts_map.hits.pdb_id,
        kwargs["pdb_mmtf_dir"],
        raise_missing=False
    )

    # compute distance maps and save
    # (but only if we found some structure)
    if len(sifts_map.hits) > 0:
        d_intra = intra_dists(
            sifts_map, structures, atom_filter=kwargs["atom_filter"],
            output_prefix=aux_prefix + "_distmap_intra"
        )
        d_intra.to_file(outcfg["distmap_monomer"])

        # save contacts to separate file
        outcfg["monomer_contacts_file"] = prefix + "_contacts_monomer.csv"
        d_intra.contacts(
            kwargs["distance_cutoff"]
        ).to_csv(
            outcfg["monomer_contacts_file"], index=False
        )

        # compute multimer distances, if requested;
        # note that d_multimer can be None if there
        # are no structures with multiple chains
        if kwargs["compare_multimer"]:
            d_multimer = multimer_dists(
                sifts_map, structures, atom_filter=kwargs["atom_filter"],
                output_prefix=aux_prefix + "_distmap_multimer"
            )
        else:
            d_multimer = None

        # if we have a multimer contact mapin the end, save it
        if d_multimer is not None:
            d_multimer.to_file(outcfg["distmap_multimer"])
            outcfg["multimer_contacts_file"] = prefix + "_contacts_multimer.csv"

            # save contacts to separate file
            d_multimer.contacts(
                kwargs["distance_cutoff"]
            ).to_csv(
                outcfg["multimer_contacts_file"], index=False
            )
        else:
            outcfg["distmap_multimer"] = None

        # at this point, also create remapped structures (e.g. for
        # later comparison of folding results)
        verify_resources(
            "Target sequence file does not exist",
            kwargs["target_sequence_file"]
        )

        # create target sequence map for remapping structure
        with open(kwargs["target_sequence_file"]) as f:
            header, seq = next(read_fasta(f))

        seq_id, seq_start, seq_end = parse_header(header)
        seqmap = dict(zip(range(seq_start, seq_end + 1), seq))

        # remap structures, swap mapping index and filename in
        # dictionary so we have a list of files in the dict keys
        outcfg["remapped_pdb_files"] = {
            filename: mapping_index for mapping_index, filename in
            remap_chains(sifts_map, aux_prefix, seqmap).items()
        }
    else:
        # if no structures, can not compute distance maps
        d_intra = None
        d_multimer = None
        outcfg["distmap_monomer"] = None
        outcfg["distmap_multimer"] = None
        outcfg["remapped_pdb_files"] = None

    # Step 3: Compare ECs to distance maps

    ec_table = pd.read_csv(kwargs["ec_file"])

    # identify number of sites in EC model
    num_sites = len(
        set.union(set(ec_table.i.unique()), set(ec_table.j.unique()))
    )

    for out_file, min_seq_dist in [
        ("ec_compared_longrange_file", kwargs["min_sequence_distance"]),
        ("ec_compared_all_file", 0),
    ]:
        # compare ECs only if we minimally have intra distance map
        if d_intra is not None:
            coupling_scores_compared(
                ec_table, d_intra, d_multimer,
                dist_cutoff=kwargs["distance_cutoff"],
                output_file=outcfg[out_file],
                score="cn",
                min_sequence_dist=min_seq_dist
            )
        else:
            outcfg[out_file] = None

    # also create line-drawing script if we made the csv
    if outcfg["ec_compared_longrange_file"] is not None:
        ecs_longrange = pd.read_csv(outcfg["ec_compared_longrange_file"])

        outcfg["ec_lines_compared_pml_file"] = prefix + "_draw_ec_lines_compared.pml"
        pairs.ec_lines_pymol_script(
            ecs_longrange.iloc[:num_sites, :],
            outcfg["ec_lines_compared_pml_file"],
            distance_cutoff=kwargs["distance_cutoff"]
        )

    # Step 4: Make contact map plots
    # if no structures available, defaults to EC-only plot

    outcfg["contact_map_files"] = _make_contact_maps(
        ec_table, d_intra, d_multimer, **kwargs
    )

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
