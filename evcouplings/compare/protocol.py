"""
EC to 3D structure comparison protocols/workflows.

Authors:
  Thomas A. Hopf
  Anna G. Green (complex and _make_complex_contact_maps)
"""
import joblib
from copy import deepcopy
from math import ceil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from evcouplings.align.alignment import (
    read_fasta, parse_header
)
from evcouplings.utils.config import (
    check_required, InvalidParameterError
)

from evcouplings.utils.system import (
    create_prefix_folders, insert_dir, verify_resources, valid_file
)
from evcouplings.couplings import Segment, add_mixture_probability, enrichment
from evcouplings.compare.pdb import load_structures
from evcouplings.compare.distances import (
    intra_dists, multimer_dists, remap_chains,
    inter_dists, remap_complex_chains
)
from evcouplings.compare.sifts import SIFTS, SIFTSResult
from evcouplings.compare.ecs import (
    coupling_scores_compared, add_precision
)
from evcouplings.visualize import pairs, misc
from evcouplings.compare.asa import combine_asa, add_asa

from evcouplings.align import ALPHABET_PROTEIN

#Hydropathy index from Lehninger Principles of Biochemistry, 5th Edition, table 3-1
HYDROPATHY_INDEX = {
    "G": -.4,
    "A": 1.8,
    "P": -1.6,
    "V": 4.2,
    "L": 3.8,
    "I": 4.5,
    "M": 1.9,
    "F": 2.8,
    "Y": -1.3,
    "W": -0.9,
    "S": -0.8,
    "T": -0.7,
    "C": 2.5,
    "N": -3.5,
    "Q": -3.5,
    "K": -3.9,
    "H": -3.2,
    "R": -4.5,
    "D": -3.5,
    "E": -3.5,
    "-": 0
}

SIFTS_TABLE_FORMAT_STR = "{pdb_id}:{pdb_chain} ({coord_start}-{coord_end})"


def print_pdb_structure_info(sifts_result, format_string=SIFTS_TABLE_FORMAT_STR,
                             header_text=None, hits_per_row=4, separator=", ",
                             location=(0.5, -0.08), text_kwargs=None, ax=None):
    """
    Add PDB structure information text to plot (e.g. contact map)

    Parameters
    ----------
    sifts_result : SIFTSResult
        Structure table that will be basis of information in plot
    format_string : str (optional, default: SIFTS_TABLE_FORMAT_STR)
        Python format string to create text for each PDB hit.
        Can use any column name from SIFTSResult.hits as key in format string
        (see default value for example).
    header_text : str, optional (default: None)
        Additional header line to show before structure information
    hits_per_row : int, optional (default: 4)
        Number of PDB chains to print per row of text
    separator : str, optional (default: ", "
        Separator text that will be printed between different PDB chain
        strings
    location : tuple(int, int), optional (default: (0.5, -0.08)
        x- and y-location where text will be printed (in matplotlib
        axes coordinates)
    text_kwargs : dict, optional (default: None)
        Keyword arguments that will be passed to matplotlib ax.text()
        for printing the text on the plot. If None, will use
        {"ha": "center", "va": "top"} as default setting.
    ax : Matplotlib Axes object, optional (default: None)
        Axes to print text on. If None, will use current matplotlib
        axies (plt.gca())
    """
    # get axis to annotate
    ax = ax or plt.gca()

    # set text plotting kwargs unless override value is supplied
    if text_kwargs is None:
        text_kwargs = {
            "ha": "center",
            "va": "top"
        }

    # if no hits, do not print anything
    if len(sifts_result.hits) == 0:
        return

    # format individual hits
    pdb_texts = [
        format_string.format(**r) for idx, r in sifts_result.hits.iterrows()
    ]

    # add hits into one list per line
    pdb_lines = [
        separator.join(
            pdb_texts[i:i + hits_per_row]
        ) for i in range(
            0, len(pdb_texts), hits_per_row
        )
    ]

    # add header text if supplied
    if header_text is not None:
        pdb_lines = [header_text] + pdb_lines

    # join into multi-line text
    joined_pdb_text = "\n".join(pdb_lines)

    # print text to plot
    ax.text(
        *location, joined_pdb_text,
        transform=ax.transAxes,
        **text_kwargs
    )

X_STRUCFREE = [
    "Z_score",
    "conservation_max",
    "f_hydrophilicity",
    "intra_enrich_max",
    "inter_relative_rank_longrange"

]
X_STRUCAWARE = [
    "Z_score",
    "asa_min",
    "precision",
    "conservation_max",
    "intra_enrich_max",
    "inter_relative_rank_longrange",
    "f_hydrophilicity"
]

X_COMPLEX_STRUCFREE = [0, 2]

X_COMPLEX_STRUCAWARE = [0, 1, 4, 6]

def fit_model(data, model_file, X, column_name):
    """
    Fits a model to predict p(residue interaction)

    data: pd.DataFrame
        has columns X used as features in model
    model_file: str
        path to file containing joblib dumped model (here, an sklearn logistic regression)
    X: list of str
        the columns to be input as features to the model. N.B., MUST be in the same order
        as when the model was originally fit
    column_name: str
        name of column to create

    Returns
        pd.DataFrame of ECs with new column column_name containing the fit model,
        or np.nan if the model could not be fit due to missing data
    """

    model = joblib.load(model_file)

    # the default score is np.nan
    data[column_name] = np.nan

    # if any of the needed columns are missing, return data
    for col in X:
        if not col in data.columns:
            return data

    # drop rows with missing info
    subset_data = data.dropna(subset=X)
    if len(subset_data) == 0:
        return data

    X_var = subset_data[X]
    predicted = model.predict_proba(X_var)[:,1]

    # make prediction and save to correct row
    data.loc[subset_data.index, column_name] = predicted

    return data

def fit_complex_model(ecs, model_file, scaler_file, residue_score_column, output_column, scores_to_use):
    """
    Fits a model to predict p(protein interaction)

    data: pd.DataFrame
        has columns X used as features in model
    model_file: str
        path to file containing joblib dumped model (here, an sklearn logistic regression)
    scaler_file: str
        path to file containing joblib dumped Scaler object
    residue_score_column: str
        a column name in data to be used as input to model
    output_column: str
        name of column to create
    scores_to_use: list of int
        name of column to create

    Returns
        pd.DataFrame of ECs with new column column_name containing the fit model,
        or np.nan if the model could not be fit due to missing data
    """
    #load the model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # sort by the residue score column, and take the instances input
    ecs = ecs.sort_values(residue_score_column, ascending=False)
    X = list(ecs[residue_score_column].iloc[scores_to_use]) + \
            [ecs.inter_relative_rank_longrange.min()]

    # reshape and clean the data
    X = np.array(X).astype(float)
    X = X.transpose()
    X = np.array(X).reshape(1, -1)
    X = np.nan_to_num(X)

    # transform with the scaler
    X = scaler.transform(X)

    ecs.loc[:,output_column]=model.predict_proba(X)[:,1]
    return ecs

def complex_probability(ecs, scoring_model, use_all_ecs=False,
                        score="cn", N_effL=None):
    """
    Adds confidence measure for complex evolutionary couplings

    Parameters
    ----------
    ecs : pandas.DataFrame
        Table with evolutionary couplings
    scoring_model : {"skewnormal", "normal", "evcomplex"}
        Use this scoring model to assign EC confidence measure
    use_all_ecs : bool, optional (default: False)
        If true, fits the scoring model to all ECs;
        if false, fits the model to only the inter ECs.
    score : str, optional (default: "cn")
        Use this score column for confidence assignment

    Returns
    -------
    ecs : pandas.DataFrame
        EC table with additional column "probability"
        containing confidence measure
    """
    from evcouplings.couplings.pairs import add_mixture_probability

    if use_all_ecs:
        ecs = add_mixture_probability(
            ecs, model=scoring_model
        )
    else:
        inter_ecs = ecs.query("segment_i != segment_j")
        intra_ecs = ecs.query("segment_i == segment_j")

        intra_ecs = add_mixture_probability(
            intra_ecs, model=scoring_model, score=score, N_effL=N_effL
        )

        inter_ecs = add_mixture_probability(
            inter_ecs, model=scoring_model, score=score, N_effL=N_effL
        )

        ecs = pd.concat(
            [intra_ecs, inter_ecs]
        ).sort_values(
            score, ascending=False
        )

    return ecs


def _filter_structures(sifts_map, pdb_ids=None, max_num_hits=None, max_num_structures=None):

    """
    Filters input SIFTSResult for specific pdb ids and/or number of hits

    Parameters
    ----------
    sifts_map: SIFTSResult
        Identified structures and residue index mappings
    pdb_ids: list of str, optional (default: None)
        List of PDB ids to be used for comparison
    max_num_hits: int, optional (default: None)
        Number of PDB hits to be used for comparison.
        Different chains from the same PDB count as multiple hits.
    max_num_structures: int, optional (default: None)
        Number of unique PDB ids to be used for comparison.
        Different chains from the same PDB count as one hit.

    Returns
    -------
    SIFTSResult
        Filtered identified structures and residue index mappings
    """
    def _filter_by_id(x, id_list):
        x = deepcopy(x)
        x.hits = x.hits.loc[
            x.hits.pdb_id.isin(id_list)
        ]
        return x


    # filter ID list down to manually selected PDB entries
    if pdb_ids is not None:
        pdb_ids = pdb_ids

        # make sure we have a list of PDB IDs
        if not isinstance(pdb_ids, list):
            pdb_ids = [pdb_ids]

        pdb_ids = [x.lower() for x in pdb_ids]
        sifts_map = _filter_by_id(sifts_map, pdb_ids)

    # limit number of hits and structures
    if max_num_hits is not None:
        sifts_map.hits = sifts_map.hits.iloc[:max_num_hits]

    if max_num_structures is not None:
        keep_ids = sifts_map.hits.pdb_id.unique()
        keep_ids = keep_ids[:max_num_structures]
        sifts_map = _filter_by_id(sifts_map, keep_ids)

    return sifts_map

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

    check_required(
        kwargs,
        [
            "prefix", "pdb_ids", "compare_multimer",
            "max_num_hits", "max_num_structures",
            "pdb_mmtf_dir",
            "sifts_mapping_table", "sifts_sequence_db",
            "by_alignment", "pdb_alignment_method",
            "alignment_min_overlap",
            "sequence_id", "sequence_file", "region",
            "use_bitscores", "domain_threshold",
            "sequence_threshold"
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

        # if searching by alignment, verify that
        # user selected jackhmmer or hmmsearch
        SEARCH_METHODS = ["jackhmmer", "hmmsearch"]

        if kwargs["pdb_alignment_method"] not in SEARCH_METHODS:
            raise InvalidParameterError(
                "Invalid pdb search method: " +
                "{}. Valid selections are: {}".format(
                    ", ".join(SEARCH_METHODS.keys())
                )
            )

        sifts_map = s.by_alignment(
            reduce_chains=reduce_chains,
            min_overlap=kwargs["alignment_min_overlap"],
            **kwargs
        )
    else:
        sifts_map = s.by_uniprot_id(
            kwargs["sequence_id"], reduce_chains=reduce_chains
        )

    # Save the pre-filtered SIFTs map
    sifts_map_full = deepcopy(sifts_map)

    #Filter the SIFTS map
    sifts_map = _filter_structures(
       sifts_map, kwargs["pdb_ids"], kwargs["max_num_hits"], kwargs["max_num_structures"]
    )

    return sifts_map, sifts_map_full


def _make_contact_maps(ec_table, d_intra, d_multimer, sifts_map, **kwargs):
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
            fig = plt.figure(figsize=(10, 10))
            if kwargs["scale_sizes"]:
                ecs = ecs.copy()
                ecs.loc[:, "size"] = ecs.score.values / ecs.score.max()
                # avoid negative sizes
                ecs.loc[ecs["size"] < 0, "size"] = 0

            # draw PDB structure and alignment/EC coverage information on contact map if selected
            # (for now, not a required parameter, default to True)
            if kwargs.get("draw_coverage", True):
                additional_plot_kwargs = {
                    "show_structure_coverage": True,
                    "margin": 0,
                    "ec_coverage": ec_table,
                }
            else:
                additional_plot_kwargs = {
                    "show_structure_coverage": False,
                    "margin": 5,
                    "ec_coverage": None,
                }

            pairs.plot_contact_map(
                ecs, d_intra, d_multimer,
                distance_cutoff=kwargs["distance_cutoff"],
                show_secstruct=kwargs["draw_secondary_structure"],
                boundaries=kwargs["boundaries"],
                **additional_plot_kwargs
            )

            # print PDB information if selected as parameter
            # (for now, not a required parameter, default to True)
            if kwargs.get("print_pdb_information", True) and sifts_map is not None and len(sifts_map.hits) > 0:
                print_pdb_structure_info(
                    sifts_map,
                    ax=plt.gca(),
                    header_text="PDB structures:",
                )

            plt.suptitle("{} evolutionary couplings".format(len(ecs)), fontsize=14)

            if output_file is not None:
                plt.savefig(output_file, bbox_inches="tight")
                plt.close(fig)

    # TODO: eventually add draw_coverage and print_pdb_information as required parameters
    # (used above in plot_cm())
    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
            "plot_probability_cutoffs",
            "boundaries", "plot_lowest_count",
            "plot_highest_count", "plot_increase",
            "draw_secondary_structure"
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


def _make_complex_contact_maps(ec_table, d_intra_i, d_multimer_i,
                               d_intra_j, d_multimer_j,
                               d_inter, first_segment_name,
                               second_segment_name, inter_ecs_model_prediction_file, **kwargs):
    """
    Plot contact maps with all ECs above a certain probability threshold,
    or a given count of ECs

    Parameters
    ----------
    ec_table : pandas.DataFrame
        Full set of evolutionary couplings (all pairs)
    d_intra_i, d_intra_j: DistanceMap
        Computed residue-residue distances within chains for
        monomers i and j
    d_multimer_i, d_multimer_j : DistanceMap
        Computed residue-residue distances between homomultimeric
        chains for monomers i and j
    d_inter: DistanceMap
        Computed residue-residue distances between heteromultimeric
        chains i and j
    first_segment_name, second_segment_name: str
        Name of segment i and segment j in the ec_table
    **kwargs
        Further plotting parameters, see check_required in code
        for necessary values.

    Returns
    -------
    cm_files : list(str)
        Paths of generated contact map files
    """

    def plot_complex_cm(ecs_i, ecs_j, ecs_inter,
                        first_segment_name,
                        second_segment_name,
                         output_file=None):
        """
        Simple wrapper for contact map plotting
        """
        with misc.plot_context("Arial"):
            if kwargs["scale_sizes"]:
                # to scale sizes, combine all ecs to rescale together
                ecs = pd.concat([ecs_i, ecs_j, ecs_inter])
                ecs.loc[:, "size"] = ecs.cn.values / ecs.cn.max()

                # split back into three separate DataFrames
                ecs_i = ecs.query("segment_i == segment_j == @first_segment_name")
                ecs_j = ecs.query("segment_i == segment_j == @second_segment_name")
                ecs_inter = ecs.query("segment_i != segment_j")

                # if any of these groups are entry, replace with None
                if len(ecs_i) == 0:
                    ecs_i = None
                if len(ecs_j) == 0:
                    ecs_j = None
                if len(ecs_inter) == 0:
                    ecs_inter = None

            # Currently, we require at least one of the monomer
            # to have either ECs or distances in order to make a plot
            if ((ecs_i is None or ecs_i.empty) and d_intra_i is None and d_multimer_i is None) \
                    or ((ecs_j is None or ecs_j.empty) and d_intra_j is None and d_multimer_j is None):
                return False

            fig = plt.figure(figsize=(8, 8))

            # create the contact map
            pairs.complex_contact_map(
                ecs_i, ecs_j, ecs_inter,
                d_intra_i, d_multimer_i,
                d_intra_j, d_multimer_j,
                d_inter,
                margin=5,
                boundaries=kwargs["boundaries"],
                scale_sizes=kwargs["scale_sizes"],
                show_secstruct=kwargs["draw_secondary_structure"],
                distance_cutoff=kwargs["distance_cutoff"]
            )

            # Add title to the plot
            if ecs_inter is None:
                ec_len = '0'
            else:
                ec_len = len(ecs_inter)
            plt.suptitle(
                "{} inter-molecule evolutionary couplings".format(ec_len),
                fontsize=14
            )

            # save to output
            if output_file is not None:
                plt.savefig(output_file, bbox_inches="tight")
                plt.close(fig)

            return True

    # transform fraction of number of sites into discrete number of ECs
    def _discrete_count(x):
        if isinstance(x, float):
            num_sites = 0
            for seg_name in [first_segment_name, second_segment_name]:
                num_sites += len(
                    set.union(
                        set(ec_table.query("segment_i == @seg_name").i.unique()),
                        set(ec_table.query("segment_j == @seg_name").j.unique())
                    )
                )

            x = ceil(x * num_sites)

        return int(x)

    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
            "plot_probability_cutoffs",
            "boundaries",
            "draw_secondary_structure", "plot_lowest_count",
            "plot_highest_count", "plot_increase",
            "scale_sizes"
        ]
    )

    prefix = kwargs["prefix"]

    cm_files = []

    ecs_longrange = ec_table.query(
        "abs(i - j) >= {} or segment_i != segment_j".format(kwargs["min_sequence_distance"])
    )

    # create plots based on significance cutoff
    if kwargs["plot_probability_cutoffs"]:
        cutoffs = kwargs["plot_probability_cutoffs"]
        if not isinstance(cutoffs, list):
            cutoffs = [cutoffs]

        for c in cutoffs:
            ec_set = ecs_longrange.query("probability >= @c")

            # only can plot if we have any significant ECs above threshold
            if len(ec_set) > 0:
                ec_set_i = ec_set.query("segment_i == segment_j == @first_segment_name")
                ec_set_j = ec_set.query("segment_i == segment_j == @second_segment_name")
                ec_set_inter = ec_set.query("segment_i != segment_j")

                output_file = prefix + "_significant_ECs_{}.pdf".format(c)
                plot_completed = plot_complex_cm(
                    ec_set_i, ec_set_j, ec_set_inter,
                    first_segment_name, second_segment_name,
                    output_file=output_file
                )
                if plot_completed:
                    cm_files.append(output_file)

    # range of plots to make
    lowest = _discrete_count(kwargs["plot_lowest_count"])
    highest = _discrete_count(kwargs["plot_highest_count"])
    step = _discrete_count(kwargs["plot_increase"])

    for c in range(lowest, highest + 1, step):
        # get the inter ECs to plot
        ec_set_inter = ecs_longrange.query("segment_i != segment_j")[0:c]

        # if there are no inter ecs to be plotted, continue
        if ec_set_inter.empty:
            continue

        # get the index of the lowest inter EC
        last_inter_index = ec_set_inter.index[-1]

        # take all intra-protein ECs that score higher than the lowest plotted inter-protein EC
        ec_set_i = ecs_longrange.iloc[0:last_inter_index].query(
            "segment_i == segment_j == @first_segment_name"
        )
        ec_set_j = ecs_longrange.iloc[0:last_inter_index].query(
            "segment_i == segment_j == @second_segment_name"
        )

        output_file = prefix + "_{}_ECs.pdf".format(c)
        plot_completed = plot_complex_cm(
            ec_set_i, ec_set_j, ec_set_inter,
            first_segment_name, second_segment_name,
            output_file=output_file
        )
        if plot_completed:
            cm_files.append(output_file)

    # give back list of all contact map file names
    return cm_files


def _individual_distance_map_config_result(individual_distance_map_table):
    """
    Create output items for individual distance maps computed by
    intra_dists, multimer_dists and inter_dists

    Parameters
    ----------
    individual_distance_map_table : pd.DataFrame
        Table as returned by intra_dists, multimer_idsts and inter_dists

    Returns
    -------
    individual_maps_result : dict
        Mapping from filename to distance map file type (residue table,
        or distance matrix) and additional annotation (mapping index,
        SIFTS table index)
    """
    individual_maps_result = {}
    file_keys = ["residue_table", "distance_matrix"]

    # create on individual entry per file type (distance matrix or residue table)
    for file_key in file_keys:
        # entry is a mapping from file name to dictionary containing file type and all
        # other remaining entries in the input table
        current_key_results = {
            r[file_key]: {
                **{"file_type": file_key},
                # keep any non-file key attributes from the table
                **{
                    k: v for k, v in r.items() if k not in file_keys
                }
            }
            for _, r in individual_distance_map_table.iterrows()
        }

        # update full table
        individual_maps_result = {
            **individual_maps_result,
            **current_key_results
        }

    return individual_maps_result


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
        * distmap_monomer (prefix of distance matrix and residue table filenames)
        * distmap_monomer_files (explicit listing of distance matrix and residue table filenames=
        * distmap_multimer (prefix of distance matrix and residue table filenames)
        * distmap_multimer_files (explicit listing of distance matrix and residue table filenames=
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

        # residue map for all individual distance maps before aggregation
        "distmap_monomer_residues_file": prefix + "_distance_map_monomer_residues.csv",
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
        outcfg["pdb_structure_hits_file"], index=True
    )

    # also save full list of hits
    sifts_map_full.hits.to_csv(
        outcfg["pdb_structure_hits_unfiltered_file"], index=True
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

        residue_table_filename, dist_mat_filename = d_intra.to_file(outcfg["distmap_monomer"])

        # store residue map (monomer)
        d_intra.aggregated_residue_maps.to_csv(
            outcfg["distmap_monomer_residues_file"], index=False
        )

        # TODO: for now, create additional entries rather than removing distmap_monomer for compatibility reasons,
        # but eventually drop the one above
        outcfg["distmap_monomer_files"] = {
            residue_table_filename: {"file_type": "residue_table"},
            dist_mat_filename: {"file_type": "distance_matrix"}
        }

        d_intra_individual_maps = d_intra.individual_distance_map_table
        # also store individual intra distance matrices (should always be present)
        if d_intra_individual_maps is not None:
            outcfg["distmap_monomer_individual_files"] = _individual_distance_map_config_result(
                d_intra_individual_maps
            )

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

            # if we have a multimer contact map in the end, save it
        if d_multimer is not None:
            residue_table_filename, dist_mat_filename = d_multimer.to_file(outcfg["distmap_multimer"])
            # TODO: for now, create additional entries rather than removing distmap_multimer for compatibility reasons,
            outcfg["distmap_multimer_files"] = {
                residue_table_filename: {"file_type": "residue_table"},
                dist_mat_filename: {"file_type": "distance_matrix"}
            }

            d_multimer_individual_maps = d_multimer.individual_distance_map_table
            # also store individual multimer distance matrices
            if d_multimer_individual_maps is not None:
                outcfg["distmap_multimer_individual_files"] = _individual_distance_map_config_result(
                    d_multimer_individual_maps
                )

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
        #
        # remapped structures have side chains taken off and changed
        # residue types, since e.g. maxcluster cannot handle mismatches
        # well. Also create structures that are just renumbered (but have
        # original side chains and residue names) for visualization asf.
        for name, sequence_map, atom_filter in [
            ("remapped", seqmap, ("N", "CA", "C", "O")),
            ("renumbered", None, None)
        ]:
            outcfg[name + "_pdb_files"] = {
                filename: mapping_index for mapping_index, filename in
                remap_chains(
                    sifts_map,
                    "{}_{}".format(aux_prefix, name),
                    sequence=sequence_map,
                    atom_filter=atom_filter
                ).items()
            }
    else:
        # if no structures, can not compute distance maps
        d_intra = None
        d_multimer = None
        outcfg["distmap_monomer"] = None
        outcfg["distmap_multimer"] = None
        outcfg["remapped_pdb_files"] = None
        outcfg["renumbered_pdb_files"] = None
        outcfg["distmap_monomer_residues_file"] = None

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
                min_sequence_dist=min_seq_dist,
                score="score"
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
            distance_cutoff=kwargs["distance_cutoff"],
            score_column="score"
        )

    # Step 4: Make contact map plots
    # if no structures available, defaults to EC-only plot

    outcfg["contact_map_files"] = _make_contact_maps(
        ec_table, d_intra, d_multimer, sifts_map, **kwargs
    )

    return outcfg


def complex(**kwargs):
    """
    Protocol:
    Compare ECs for a complex to
    3D structure

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
            "pdb_mmtf_dir", "atom_filter",
            "first_compare_multimer", "second_compare_multimer",
            "distance_cutoff", "segments",
            "first_sequence_id", "second_sequence_id",
            "first_sequence_file", "second_sequence_file",
            "first_target_sequence_file", "second_target_sequence_file",
            "scale_sizes", "structurefree_model_file", "structureaware_model_file"
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        # initialize output EC files
        "ec_compared_all_file": prefix + "_CouplingScoresCompared_all.csv",
        "ec_compared_longrange_file": prefix + "_CouplingScoresCompared_longrange.csv",
        "ec_compared_inter_file": prefix + "_CouplingsScoresCompared_inter.csv",

        # initialize output inter distancemap files
        "distmap_inter": prefix + "_distmap_inter",
        "inter_contacts_file": prefix + "_inter_contacts.csv"
    }

    # Add PDB comparison files for first and second monomer
    for monomer_prefix in ["first", "second"]:
        outcfg = {
            **outcfg,
            monomer_prefix + "_pdb_structure_hits_file":
                "{}_{}_structure_hits.csv".format(prefix, monomer_prefix),
            monomer_prefix + "_pdb_structure_hits_unfiltered_file":
                "{}_{}_structure_hits_unfitered.csv".format(prefix, monomer_prefix),
            monomer_prefix + "_distmap_monomer":
                "{}_{}_distance_map_monomer".format(prefix, monomer_prefix),
            monomer_prefix + "_distmap_multimer":
                "{}_{}_distance_map_multimer".format(prefix, monomer_prefix),
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

    # store auxiliary files here (too much for average user)
    first_aux_prefix = insert_dir(aux_prefix, "first_monomer", rootname_subdir=False)
    create_prefix_folders(first_aux_prefix)

    # store auxiliary files here (too much for average user)
    second_aux_prefix = insert_dir(aux_prefix, "second_monomer", rootname_subdir=False)
    create_prefix_folders(second_aux_prefix)

    # Step 1: Identify 3D structures for comparison
    def _identify_monomer_structures(name_prefix, outcfg, aux_prefix):
        # create a dictionary with kwargs for just the current monomer
        # any prefix that starts with a name_prefix will overwrite prefixes that do not start
        # eg, "first_sequence_file" will overwrite "sequence_file"
        monomer_kwargs = deepcopy(kwargs)
        for k,v in kwargs.items():
            if name_prefix + "_" in k:
                # only replace first occurrence of name_prefix
                monomer_kwargs[k.replace(name_prefix + "_", "", 1)] = v

        # remove the "prefix" kwargs so that we can replace with the
        # aux prefix when calling _identify_structures
        monomer_kwargs = {
            k: v for k, v in monomer_kwargs.items() if "prefix" not in k
        }

        # identify structures for that monomer
        sifts_map, sifts_map_full = _identify_structures(
            **monomer_kwargs,
            prefix=aux_prefix
        )

        # save full list of hits
        sifts_map_full.hits.to_csv(
            outcfg[name_prefix + "_pdb_structure_hits_unfiltered_file"], index=False
        )

        return outcfg, sifts_map, sifts_map_full

    outcfg, first_sifts_map, first_sifts_map_full = _identify_monomer_structures("first", outcfg, first_aux_prefix)
    outcfg, second_sifts_map, second_sifts_map_full = _identify_monomer_structures("second", outcfg, second_aux_prefix)

    # Determine the inter-protein PDB hits based on the full sifts map for each monomer
    inter_protein_hits_full = first_sifts_map_full.hits.merge(
        second_sifts_map_full.hits, on="pdb_id", how="inner", suffixes=["_1", "_2"]
    )
    outcfg["structure_hits_unfiltered_file"] = prefix + "_inter_structure_hits_unfiltered.csv"
    inter_protein_hits_full.to_csv(outcfg["structure_hits_unfiltered_file"])

    # Filter for the number of PDB ids to use
    inter_protein_sifts = SIFTSResult(hits=inter_protein_hits_full, mapping=None)
    inter_protein_sifts = _filter_structures(
        inter_protein_sifts,
        kwargs["inter_pdb_ids"],
        kwargs["inter_max_num_hits"],
        kwargs["inter_max_num_structures"]
    )

    outcfg["structure_hits_file"] = prefix + "_inter_structure_hits.csv"
    inter_protein_sifts.hits.to_csv(outcfg["structure_hits_file"])

    def _add_inter_pdbs(inter_protein_sifts, sifts_map, sifts_map_full, name_prefix,):
        """
        ensures that all pdbs used for inter comparison end up in the monomer SIFTS hits
        """

        lines_to_keep = sifts_map_full.hits.query("pdb_id in @inter_protein_sifts.hits.pdb_id").index
        sifts_map.hits = pd.concat([
            sifts_map.hits, sifts_map_full.hits.loc[lines_to_keep, :]
        ]).drop_duplicates()

        # save selected PDB hits
        sifts_map.hits.to_csv(
            outcfg[name_prefix + "_pdb_structure_hits_file"], index=False
        )
        return sifts_map

    first_sifts_map = _add_inter_pdbs(inter_protein_sifts, first_sifts_map, first_sifts_map_full, "first")
    second_sifts_map = _add_inter_pdbs(inter_protein_sifts, second_sifts_map, second_sifts_map_full, "second")

    # get the segment names from the kwargs
    segment_list = kwargs["segments"]

    # Make sure user provided exactly two segments
    if len(segment_list) != 2:
        raise InvalidParameterError(
            "Compare stage for protein complexes requires exactly two segments"
        )

    first_segment_name = Segment.from_list(kwargs["segments"][0]).segment_id
    second_segment_name = Segment.from_list(kwargs["segments"][1]).segment_id

    first_chain_name = Segment.from_list(kwargs["segments"][0]).default_chain_name()
    second_chain_name = Segment.from_list(kwargs["segments"][1]).default_chain_name()

    # load all structures at once
    all_ids = set(first_sifts_map.hits.pdb_id).union(
        set(second_sifts_map.hits.pdb_id)
    )

    structures = load_structures(
        all_ids,
        kwargs["pdb_mmtf_dir"],
        raise_missing=False
    )

    # Step 2: Compute distance maps
    def _compute_monomer_distance_maps(sifts_map, name_prefix, chain_name):

        # prepare a sequence map to remap the structures we have found
        verify_resources(
            "Target sequence file does not exist",
            kwargs[name_prefix + "_target_sequence_file"]
        )

        # create target sequence map for remapping structure
        with open(kwargs[name_prefix + "_target_sequence_file"]) as f:
            header, seq = next(read_fasta(f))

        # create target sequence map for remapping structure
        seq_id, seq_start, seq_end = parse_header(header)
        seqmap = dict(zip(range(seq_start, seq_end + 1), seq))

        # compute distance maps and save
        # (but only if we found some structure)
        if len(sifts_map.hits) > 0:
            d_intra = intra_dists(
                sifts_map, structures, atom_filter=kwargs["atom_filter"],
                output_prefix=aux_prefix + "_" + name_prefix + "_distmap_intra"
            )
            d_intra.to_file(outcfg[name_prefix + "_distmap_monomer"])

            # save contacts to separate file
            outcfg[name_prefix + "_monomer_contacts_file"] = prefix + "_" + name_prefix + "_contacts_monomer.csv"
            d_intra.contacts(
                kwargs["distance_cutoff"]
            ).to_csv(
                outcfg[name_prefix + "_monomer_contacts_file"], index=False
            )

            # compute multimer distances, if requested;
            # note that d_multimer can be None if there
            # are no structures with multiple chains
            if kwargs[name_prefix + "_compare_multimer"]:
                d_multimer = multimer_dists(
                    sifts_map, structures, atom_filter=kwargs["atom_filter"],
                    output_prefix=aux_prefix + "_" + name_prefix + "_distmap_multimer"
                )
            else:
                d_multimer = None

            # if we have a multimer contact map, save it
            if d_multimer is not None:
                d_multimer.to_file(outcfg[name_prefix + "_distmap_multimer"])
                outcfg[name_prefix + "_multimer_contacts_file"] = prefix + "_" + name_prefix + "_contacts_multimer.csv"

                # save contacts to separate file
                d_multimer.contacts(
                    kwargs["distance_cutoff"]
                ).to_csv(
                    outcfg[name_prefix + "_multimer_contacts_file"], index=False
                )
            else:
                outcfg[name_prefix + "_distmap_multimer"] = None

            # create remapped structures (e.g. for
            # later comparison of folding results)
            # remap structures, swap mapping index and filename in
            # dictionary so we have a list of files in the dict keys
            outcfg[name_prefix + "_remapped_pdb_files"] = {
                filename: mapping_index for mapping_index, filename in
                remap_chains(
                    sifts_map, aux_prefix, None, chain_name=chain_name,
                    raise_missing=kwargs["raise_missing"], atom_filter=None
                ).items()
            }

        else:
            # if no structures, cannot compute distance maps
            d_intra = None
            d_multimer = None
            outcfg[name_prefix + "_distmap_monomer"] = None
            outcfg[name_prefix + "_distmap_multimer"] = None
            outcfg[name_prefix + "remapped_pdb_files"] = None

        return d_intra, d_multimer, seqmap

    d_intra_i, d_multimer_i, seqmap_i = _compute_monomer_distance_maps(
        first_sifts_map, "first", first_chain_name
    )
    d_intra_j, d_multimer_j, seqmap_j = _compute_monomer_distance_maps(
        second_sifts_map, "second", second_chain_name
    )

    # compute inter distance map if sifts map for each monomer exists
    if len(first_sifts_map.hits) > 0 and len(second_sifts_map.hits) > 0:
        d_inter = inter_dists(
            first_sifts_map, second_sifts_map,
            raise_missing=kwargs["raise_missing"]
        )
        # if there were overlapping PDBs, save the results
        if d_inter is not None:
            d_inter.to_file(outcfg["distmap_inter"])

            # save contacts to separate file
            d_inter.contacts(
                kwargs["distance_cutoff"]
            ).to_csv(
                outcfg["inter_contacts_file"], index=False
            )

    else:
        outcfg["inter_contacts_file"] = None
        d_inter = None

    # # Step 3: Compare ECs to distance maps
    ec_table = pd.read_csv(kwargs["ec_file"])

    for out_file, min_seq_dist in [
        ("ec_compared_longrange_file", kwargs["min_sequence_distance"]),
        ("ec_compared_all_file", 0),
    ]:

        # compare ECs only if we have an intra distance map
        # for at least one monomer - inter can't exist unless
        # we have both monomers
        if (d_intra_i is not None) or (d_intra_j is not None):
            # compare distances individually for each segment pair
            ecs_intra_i = ec_table.query("segment_i == segment_j == @first_segment_name")
            if d_intra_i is not None:
                ecs_intra_i_compared = coupling_scores_compared(
                    ecs_intra_i, d_intra_i, d_multimer_i,
                    dist_cutoff=kwargs["distance_cutoff"],
                    output_file=None,
                    min_sequence_dist=min_seq_dist
                )
            else:
                # If no distance map, the distance is saved as np.nan
                ecs_intra_i_compared = ecs_intra_i.assign(dist=np.nan)

            ecs_intra_j = ec_table.query("segment_i == segment_j == @second_segment_name")
            if d_intra_j is not None:
                ecs_intra_j_compared = coupling_scores_compared(
                    ecs_intra_j, d_intra_j, d_multimer_j,
                    dist_cutoff=kwargs["distance_cutoff"],
                    output_file=None,
                    min_sequence_dist=min_seq_dist
                )
            else:
                ecs_intra_j_compared = ecs_intra_j.assign(dist=np.nan)

            ecs_inter = ec_table.query("segment_i != segment_j")
            if d_inter is not None:
                ecs_inter_compared = coupling_scores_compared(
                    ecs_inter, d_inter, dist_map_multimer=None,
                    dist_cutoff=kwargs["distance_cutoff"],
                    output_file=None,
                    min_sequence_dist=None  # does not apply for inter-protein ECs
                )
            else:
                ecs_inter_compared = ecs_inter.assign(dist=np.nan)

            # combine the tables
            ec_table_compared = pd.concat([
                ecs_inter_compared,
                ecs_intra_i_compared,
                ecs_intra_j_compared
            ])

            # rename the precision column to "segmentwise_precision"
            # because we calculated precision for each segment independently
            ec_table_compared = ec_table_compared.rename(
                columns={"precision": "segmentwise_precision"}
            )
            # TODO: change "cn" to "score" eventually
            ec_table_compared = ec_table_compared.sort_values("cn", ascending=False)

            # add the total precision
            # TODO: implement different cutoffs for intra vs inter contacts
            ec_table_compared = add_precision(
                ec_table_compared,
                dist_cutoff=kwargs["distance_cutoff"]
            )

            # save to file
            # all ecs
            ec_table_compared.to_csv(outcfg[out_file])

            # save the inter ECs to a file
            ecs_inter_compared.to_csv(outcfg["ec_compared_inter_file"])

    # create an inter-ecs file with extra information for calibration purposes
    def _calibration_file(prefix, ec_file, outcfg):

        """
        Adds values to the dataframe of ECs that will later be used
        for score fitting
        """

        # If there's no EC file, don't bother
        if not valid_file(ec_file):
            return None

        ecs = pd.read_csv(ec_file)

        # calculate intra-protein enrichment
        def _add_enrichment(ecs):

            # Calculate the intra-protein enrichment
            intra1_ecs = ecs.query("segment_i == segment_j == 'A_1'")
            intra2_ecs = ecs.query("segment_i == segment_j == 'B_1'")

            intra1_enrichment = enrichment(intra1_ecs, min_seqdist=6)
            intra1_enrichment["segment_i"] = "A_1"

            intra2_enrichment = enrichment(intra2_ecs, min_seqdist=6)
            intra2_enrichment["segment_i"] = "B_1"

            enrichment_table = pd.concat([intra1_enrichment, intra2_enrichment])

            def _seg_to_enrich(enrich_df, ec_df, enrichment_column):
                """
                combines the enrichment table with the EC table
                """
                s_to_e = {(x,y):z for x,y,z in zip(enrich_df.i, enrich_df.segment_i, enrich_df[enrichment_column])}

                # enrichment for residues in column i
                ec_df["enrichment_i"] =[s_to_e[(x,y)] if (x,y) in s_to_e else 0 for x,y in zip(ec_df.i, ec_df.segment_i)]

                # enrichment for residues in column j
                ec_df["enrichment_j"] =[s_to_e[(x,y)] if (x,y) in s_to_e else 0 for x,y in zip(ec_df.j, ec_df.segment_j)]

                return ec_df

            #add the intra-protein enrichment to the EC table
            ecs = _seg_to_enrich(enrichment_table, ecs, "enrichment")
            # larger of two enrichment values
            ecs["intra_enrich_max"] = ecs[["enrichment_i", "enrichment_j"]].max(axis=1)
            # smaller of two enrichment values
            ecs["intra_enrich_min"] = ecs[["enrichment_i", "enrichment_j"]].min(axis=1)

            return ecs

        ecs = _add_enrichment(ecs)

        # get just the inter ECs and calculate Z-score
        ecs = ecs.query("segment_i != segment_j")
        mean_ec = ecs.cn.mean()
        std_ec = ecs.cn.std()
        ecs["Z_score"] = (ecs.cn - mean_ec) / std_ec

        # get only the top 100 inter ECs
        ecs = ecs[0:100]

        # add rank
        L = len(ecs.i.unique()) + len(ecs.j.unique())
        ecs["inter_relative_rank_longrange"] = ecs.index / L

        # accessible surface area
        if not "first_remapped_pdb_files" in outcfg:
            outcfg["first_remapped_pdb_files"] = []

        if not "second_remapped_pdb_files" in outcfg:
            outcfg["second_remapped_pdb_files"] = []

        # calculate the ASA for the first and second segments by combining asa from all remapped pdb files
        first_asa, outcfg = combine_asa(outcfg["first_remapped_pdb_files"], kwargs["dssp"], outcfg)
        first_asa["segment_i"] = "A_1"

        second_asa, outcfg = combine_asa(outcfg["second_remapped_pdb_files"], kwargs["dssp"], outcfg)
        second_asa["segment_i"] = "B_1"

        # save the ASA to a file
        asa = pd.concat([first_asa, second_asa])
        outcfg["asa_file"] = prefix + "_surface_area.csv"
        asa.to_csv(outcfg["asa_file"])

        # Add the ASA to the ECs and compute the max and min for each position pair
        ecs = add_asa(ecs, asa, asa_column="mean")
        ecs["asa_max"] = ecs[["asa_i", "asa_j"]].max(axis=1)
        ecs["asa_min"] = ecs[["asa_i", "asa_j"]].min(axis=1)

        # Add min and max conservation to EC file
        #frequency_file = prefix.replace("compare", "concatenate") + "_frequencies.csv"
        frequency_file = kwargs["frequencies_file"]
        d = pd.read_csv(frequency_file)
        conservation = {(x,y):z for x,y,z in zip(d.segment_i, d.i, d.conservation)}

        ecs["conservation_i"] = [conservation[(x,y)] if (x,y) in conservation else np.nan for x,y in zip(ecs.segment_i, ecs.i)]
        ecs["conservation_j"] = [conservation[(x,y)] if (x,y) in conservation else np.nan for x,y in zip(ecs.segment_j, ecs.j)]
        ecs["conservation_max"] = ecs[["conservation_i", "conservation_j"]].max(axis=1)
        ecs["conservation_min"] = ecs[["conservation_i", "conservation_j"]].min(axis=1)

        # amino acid frequencies
        for char in list(ALPHABET_PROTEIN):
            # Frequency of amino acid 'char' in position i
            ecs = ecs.merge(d[["i", "segment_i", char]], on=["i","segment_i"], how="left", suffixes=["", "_1"])
            ecs = ecs.rename({char: f"f{char}_i"}, axis=1)
            if "i_1" in ecs.columns:
                ecs = ecs.drop(columns=["i_1", "segment_i_1"])

            # Frequency of amino acid 'char' in position j
            ecs = ecs.merge(
                d[["i", "segment_i", char]], left_on=["j", "segment_j"],
                right_on=["i", "segment_i"], how="left", suffixes=["", "_1"]
            )
            ecs = ecs.rename({char: f"f{char}_j"}, axis=1)
            if "j_1" in ecs.columns:
                ecs = ecs.drop(columns=["j_1", "segment_j_1"])

        # summed frequency of amino acid char in both positions i and j
        # ie, each pair i,j now gets one combined frequency
        for char in list(ALPHABET_PROTEIN):
            ecs[f"f{char}"] = ecs[f"f{char}_i"] + ecs[f"f{char}_j"]

        # Compute the weighted sum of hydropathy for pair i, j
        hydrophilicity = []

        # For each EC
        for _, row in ecs.iterrows():
            # frequncy of amino acid char * hydopathy index of that AA
            hydro = sum([
                HYDROPATHY_INDEX[char] * float(row[[f'f{char}']]) for char in list(ALPHABET_PROTEIN)
            ])
            hydrophilicity.append(hydro)
        ecs["f_hydrophilicity"] = hydrophilicity

        #save the calibration file
        outcfg["calibration_file"] = prefix + "_CouplingScores_inter_calibration.csv"
        ecs.to_csv(outcfg["calibration_file"])

    # Compute the calibration file
    if valid_file(outcfg["ec_compared_longrange_file"]):
        _calibration_file(prefix, outcfg["ec_compared_longrange_file"], outcfg)
    else:
        _calibration_file(prefix, kwargs["ec_longrange_file"], outcfg)

    # If calibration file was correctly computed
    if valid_file(outcfg["calibration_file"]):
        calibration_ecs = pd.read_csv(outcfg["calibration_file"],index_col=0)

        # Fit the structure free model file
        calibration_ecs = fit_model(
            calibration_ecs,
            kwargs["structurefree_model_file"],
            X_STRUCFREE,
            "residue_prediction_strucfree"
        )

        # Fit the structure aware model prediction file
        calibration_ecs = fit_model(
            calibration_ecs,
            kwargs["structureaware_model_file"],
            X_STRUCAWARE,
            "residue_prediction_strucaware"
        )

        # Fit the structure free complex model
        calibration_ecs = fit_complex_model(
            calibration_ecs,
            kwargs["complex_strucfree_model_file"],
            kwargs["complex_strucfree_scaler_file"],
            "residue_prediction_strucfree",
            "complex_prediction_strucfree",
            X_COMPLEX_STRUCFREE
        )

        # Fit the structure aware complex model
        calibration_ecs = fit_complex_model(
            calibration_ecs,
            kwargs["complex_strucaware_model_file"],
            kwargs["complex_strucaware_scaler_file"],
            "residue_prediction_strucaware",
            "complex_prediction_strucaware",
            X_COMPLEX_STRUCAWARE
        )

        outcfg["inter_ecs_model_prediction_file"] = prefix +"_CouplingScores_inter_prediction.csv"
        calibration_ecs[[
            "inter_relative_rank_longrange", "i", "A_i", "j", "A_j",
            "segment_i", "segment_j", "cn", "dist", "precision",  "residue_prediction_strucaware", "residue_prediction_strucfree",
            "complex_prediction_strucaware", "complex_prediction_strucfree", "Z_score", "asa_min", "conservation_max",
            "f_hydrophilicity"
        ]].to_csv(outcfg["inter_ecs_model_prediction_file"])


    # create the inter-ecs line drawing script
    if outcfg["ec_compared_inter_file"] is not None and kwargs["plot_highest_count"] is not None:
        inter_ecs = ec_table.query("segment_i != segment_j")

        outcfg["ec_lines_compared_pml_file"] = prefix + "_draw_ec_lines_compared.pml"

        pairs.ec_lines_pymol_script(
            inter_ecs.iloc[:kwargs["plot_highest_count"], :],
            outcfg["ec_lines_compared_pml_file"],
            distance_cutoff=kwargs["distance_cutoff"],
            chain={
                first_segment_name: first_chain_name,
                second_segment_name: second_chain_name
            }
        )

    # Remap the complex crystal structures, if available
    if len(first_sifts_map.hits) > 0 and len(second_sifts_map.hits) > 0:
        outcfg["complex_remapped_pdb_files"] = {
            filename: mapping_index for mapping_index, filename in
            remap_complex_chains(
                first_sifts_map, second_sifts_map,
                seqmap_i, seqmap_j, output_prefix=aux_prefix,
                raise_missing=kwargs["raise_missing"], atom_filter=None
            ).items()
        }

    # Step 4: Make contact map plots
    # if no structures available, defaults to EC-only plot
    outcfg["contact_map_files"] = _make_complex_contact_maps(
        ec_table, d_intra_i, d_multimer_i,
        d_intra_j, d_multimer_j,
        d_inter, first_segment_name,
        second_segment_name, outcfg["inter_ecs_model_prediction_file"], **kwargs
    )

    return outcfg

# list of available EC comparison protocols
PROTOCOLS = {
    # standard monomer comparison protocol
    "standard": standard,

    # comparison for protein complexes
    "complex": complex,
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
