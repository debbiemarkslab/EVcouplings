"""
Fix EVcomplex2 models to inter-protein ECs

Authors:
  Anna G. Green
"""
import joblib
import pandas as pd
import numpy as np

from evcouplings.utils.config import (
    check_required, InvalidParameterError
)

from evcouplings.utils.system import (
    valid_file
)
from evcouplings.couplings import enrichment, Segment
from evcouplings.score_interaction.asa import combine_asa, add_asa
from evcouplings.compare.distances import DistanceMap
from evcouplings.compare.protocol import complex_probability, plot_complex_cm
from evcouplings.align import ALPHABET_PROTEIN
from evcouplings.utils.constants import HYDROPATHY_INDEX

X_STRUCFREE = [
    "evcomplex_normed",
    "conservation_max",
    "intra_enrich_max",
    "inter_relative_rank_longrange"

]
X_STRUCAWARE = [
    "evcomplex_normed",
    "asa_min",
    "precision",
    "conservation_max",
    "intra_enrich_max",
    "inter_relative_rank_longrange",
]

X_COMPLEX_STRUCFREE = [0, 2]

X_COMPLEX_STRUCAWARE = [0, 3, 4, 7]


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
        if col not in data.columns:
            print("missing", col)
            return data

    # drop rows with missing info
    subset_data = data.dropna(subset=X)
    if len(subset_data) == 0:
        return data

    X_var = subset_data[X]
    predicted = model.predict_proba(X_var)[:, 1]

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
    # load the model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # sort by the residue score column, and take the instances input
    ecs = ecs.sort_values(residue_score_column, ascending=False)
    X = list(
        ecs[residue_score_column].iloc[scores_to_use]
    ) + [ecs.inter_relative_rank_longrange.min()]

    # reshape and clean the data
    X = np.array(X).astype(float)
    X = X.transpose()
    X = np.array(X).reshape(1, -1)

    X = np.nan_to_num(X)

    # transform with the scaler
    X = scaler.transform(X)

    ecs.loc[:, output_column] =model.predict_proba(X)[:, 1]
    return ecs


def _make_complex_contact_maps_probability(ec_table, d_intra_i, d_multimer_i,
                               d_intra_j, d_multimer_j,
                               d_inter, first_segment_name,
                               second_segment_name,  **kwargs):
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
    check_required(
        kwargs,
        [
            "prefix",
            "plot_probability_cutoffs",
            "boundaries",
            "draw_secondary_structure",
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
        for column, cutoffs in kwargs["plot_probability_cutoffs"].items():

            if not isinstance(cutoffs, list):
                cutoffs = [cutoffs]

            for cutoff in cutoffs:
                ec_set_i = ecs_longrange.query("segment_i == segment_j == @first_segment_name")
                ec_set_j = ecs_longrange.query("segment_i == segment_j == @second_segment_name")
                ec_set = ecs_longrange.loc[ecs_longrange[column] >= cutoff, :]

                ec_set_inter = ec_set.query("segment_i != segment_j")

                output_file = prefix + "_{}_significant_ECs_{}.pdf".format(column, cutoff)
                plot_completed = plot_complex_cm(
                    ec_set_i, ec_set_j, ec_set_inter,
                    d_intra_i, d_multimer_i,
                    d_intra_j, d_multimer_j,
                    d_inter,
                    first_segment_name, second_segment_name,
                    output_file=output_file, **kwargs
                )

                if plot_completed:
                    cm_files.append(output_file)

    # give back list of all contact map file names
    return cm_files


def standard(**kwargs):
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
            "structurefree_model_file", "structureaware_model_file",
            "ec_compared_longrange_file", "ec_longrange_file",
            "first_remapped_pdb_files", "second_remapped_pdb_files",
            "frequencies_file"
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        # initialize output EC files
        "ec_calibration_file": prefix + "_inter_calibration.csv",
        "ec_prediction_file": prefix + "_inter_prediction.csv",
    }

    # create an inter-ecs file with extra information for calibration purposes
    def _calibration_file(prefix, ec_file, outcfg):

        """
        Adds values to the dataframe of ECs that will later be used
        for score fitting
        """

        ecs = pd.read_csv(ec_file, index_col=0)

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
                s_to_e = {(x, y):z for x, y, z in zip(
                    enrich_df.i, enrich_df.segment_i, enrich_df[enrichment_column]
                )}

                # enrichment for residues in column i
                ec_df["enrichment_i"] = [s_to_e[(x, y)] if (x, y) in s_to_e else 0 for x, y in zip(
                    ec_df.i, ec_df.segment_i
                )]

                # enrichment for residues in column j
                ec_df["enrichment_j"] = [s_to_e[(x, y)] if (x, y) in s_to_e else 0 for x, y in zip(
                    ec_df.j, ec_df.segment_j
                )]

                return ec_df

            # add the intra-protein enrichment to the EC table
            ecs = _seg_to_enrich(enrichment_table, ecs, "enrichment")
            # larger of two enrichment values
            ecs["intra_enrich_max"] = ecs[["enrichment_i", "enrichment_j"]].max(axis=1)
            # smaller of two enrichment values
            ecs["intra_enrich_min"] = ecs[["enrichment_i", "enrichment_j"]].min(axis=1)

            return ecs

        ecs = _add_enrichment(ecs)
        ecs = ecs.reset_index()

        # get just the inter ECs and calculate Z-score
        ecs = ecs.query("segment_i != segment_j")
        mean_ec = ecs.cn.mean()
        std_ec = ecs.cn.std()
        ecs.loc[:, 'Z_score'] = (ecs.cn - mean_ec) / std_ec

        # add the evcomplex score with neffL correction
        N_effL = kwargs["effective_sequences"] / kwargs["num_sites"]
        ecs = complex_probability(ecs, "evcomplex", use_all_ecs=False,
                        score="cn", Neff_over_L=N_effL)
        ecs.loc[:, "evcomplex_normed"] = ecs.loc[:, "probability"]

        # get only the top 100 inter ECs
        L = len(ecs.i.unique()) + len(ecs.j.unique())
        ecs = ecs[0:100]

        # add rank
        ecs["inter_relative_rank_longrange"] = ecs.index / L
        print(ecs.head())
        # calculate the ASA for the first and second segments by combining asa from all remapped pdb files
        first_asa, outcfg = combine_asa(kwargs["first_remapped_pdb_files"], kwargs["dssp"], prefix, outcfg)
        first_asa["segment_i"] = "A_1"

        second_asa, outcfg = combine_asa(kwargs["second_remapped_pdb_files"], kwargs["dssp"], prefix, outcfg)
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
        frequency_file = kwargs["frequencies_file"]
        d = pd.read_csv(frequency_file)
        conservation = {(x,y):z for x,y,z in zip(d.segment_i, d.i, d.conservation)}

        ecs["conservation_i"] = [
            conservation[(x, y)] if (x, y) in conservation else np.nan for x, y in zip(ecs.segment_i, ecs.i)
        ]
        ecs["conservation_j"] = [
            conservation[(x, y)] if (x, y) in conservation else np.nan for x, y in zip(ecs.segment_j, ecs.j)
        ]
        ecs["conservation_max"] = ecs[["conservation_i", "conservation_j"]].max(axis=1)
        ecs["conservation_min"] = ecs[["conservation_i", "conservation_j"]].min(axis=1)

        # # amino acid frequencies
        # for char in list(ALPHABET_PROTEIN):
        #     # Frequency of amino acid 'char' in position i
        #     ecs = ecs.merge(d[["i", "segment_i", char]], on=["i","segment_i"], how="left", suffixes=["", "_1"])
        #     ecs = ecs.rename({char: f"f{char}_i"}, axis=1)
        #     if "i_1" in ecs.columns:
        #         ecs = ecs.drop(columns=["i_1", "segment_i_1"])
        #
        #     # Frequency of amino acid 'char' in position j
        #     ecs = ecs.merge(
        #         d[["i", "segment_i", char]], left_on=["j", "segment_j"],
        #         right_on=["i", "segment_i"], how="left", suffixes=["", "_1"]
        #     )
        #     ecs = ecs.rename({char: f"f{char}_j"}, axis=1)
        #     if "j_1" in ecs.columns:
        #         ecs = ecs.drop(columns=["j_1", "segment_j_1"])
        #
        # # summed frequency of amino acid char in both positions i and j
        # # ie, each pair i,j now gets one combined frequency
        # print("computing frequencies")
        # for char in list(ALPHABET_PROTEIN):
        #     ecs[f"f{char}"] = ecs[f"f{char}_i"] + ecs[f"f{char}_j"]
        #
        # # Compute the weighted sum of hydropathy for pair i, j
        # hydrophilicity = []
        #
        # # For each EC
        # print("computing hydrophilicty")
        # for _, row in ecs.iterrows():
        #     # frequncy of amino acid char * hydopathy index of that AA
        #     hydro = sum([
        #         HYDROPATHY_INDEX[char] * float(row[[f'f{char}']]) for char in list(ALPHABET_PROTEIN)
        #     ])
        #     hydrophilicity.append(hydro)
        #
        # ecs["f_hydrophilicity"] = hydrophilicity

        # save the calibration file
        ecs.to_csv(outcfg["ec_calibration_file"])

    # Compute the calibration file
    if valid_file(kwargs["ec_compared_longrange_file"]):
        _calibration_file(prefix, kwargs["ec_compared_longrange_file"], outcfg)
    elif valid_file(kwargs["ec_longrange_file"]):
        _calibration_file(prefix, kwargs["ec_longrange_file"], outcfg)
    else:
        raise InvalidParameterError("No valid EC file provided as input for modeling")

    calibration_ecs = pd.read_csv(outcfg["ec_calibration_file"], index_col=0)
    calibration_ecs = calibration_ecs.sort_values("cn", ascending=False)[0:20]

    # Fit the structure free model file
    print('fitting the model')
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

    calibration_ecs[[
        "inter_relative_rank_longrange", "i", "A_i", "j", "A_j",
        "segment_i", "segment_j", "cn", "dist", "precision",  "evcomplex_normed",
        "Z_score", "asa_min", "conservation_max", "intra_enrich_max",
        "residue_prediction_strucaware", "residue_prediction_strucfree",
        "complex_prediction_strucaware", "complex_prediction_strucfree"
    ]].to_csv(outcfg["ec_prediction_file"])

    # Step 4: Make contact map plots
    # if no structures available, defaults to EC-only plot
    def _load_distmap(distmap_file):
        if distmap_file is not None and valid_file(distmap_file + ".csv"):
            distmap = DistanceMap.from_file(distmap_file)
        else:
            distmap = None
        return distmap

    d_intra_i = _load_distmap(kwargs["first_distmap_monomer"])
    d_multimer_i = _load_distmap(kwargs["first_distmap_multimer"])
    d_intra_j = _load_distmap(kwargs["second_distmap_monomer"])
    d_multimer_j = _load_distmap(kwargs["second_distmap_multimer"])
    d_inter = _load_distmap(kwargs["distmap_inter"])

    first_segment_name = Segment.from_list(kwargs["segments"][0]).segment_id
    second_segment_name = Segment.from_list(kwargs["segments"][1]).segment_id

    # Add the top L intra ECs to the ec_table
    ecs = pd.read_csv(kwargs["ec_file"]).query("segment_i == segment_j")
    ecs = ecs.query("abs(i - j) >= {} or segment_i != segment_j".format(kwargs["min_sequence_distance"]))
    ecs_intra = ecs.iloc[0:kwargs["num_sites"]]

    calibration_ecs = pd.concat([calibration_ecs, ecs_intra])

    outcfg["contact_map_files"] = _make_complex_contact_maps_probability(
        calibration_ecs, d_intra_i, d_multimer_i,
        d_intra_j, d_multimer_j,
        d_inter, first_segment_name,
        second_segment_name, **kwargs
    )

    return outcfg


# list of available EC comparison protocols
PROTOCOLS = {
    # standard complex model fitting protocol
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
