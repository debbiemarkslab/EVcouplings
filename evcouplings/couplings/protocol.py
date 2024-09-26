"""
Evolutionary couplings calculation protocols/workflows.

Authors:
  Thomas A. Hopf
  Anna G. Green (complex couplings)
  Benjamin Schubert
"""

import string
import pandas as pd
import numpy as np

from evcouplings.couplings import tools as ct
from evcouplings.couplings import pairs, mapping
from evcouplings.couplings.mean_field import MeanFieldDCA
from evcouplings.couplings.model import CouplingsModel
from evcouplings.visualize.parameters import evzoom_json
from evcouplings.visualize.pairs import (
    ec_lines_pymol_script, enrichment_pymol_script
)

from evcouplings.align.alignment import (
    read_fasta, ALPHABET_PROTEIN, ALPHABET_PROTEIN_NOGAP,
    ALPHABET_PROTEIN_ORDERED, ALPHABET_PROTEIN_NOGAP_ORDERED,
    ALPHABET_DNA, ALPHABET_RNA, Alignment
)

from evcouplings.utils import BailoutException

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    read_config_file, write_config_file
)

from evcouplings.utils.system import (
    create_prefix_folders, valid_file,
    verify_resources,
)

# symbols for common sequence alphabets
ALPHABET_MAP = {
    "aa": ALPHABET_PROTEIN,
    "dna": ALPHABET_DNA,
    "rna": ALPHABET_RNA,
}

# models for assigning confidence scores to ECs
SCORING_MODELS = (
    "skewnormal",
    "normal",
    "evcomplex",
)


def infer_plmc(**kwargs):
    """
    Run EC computation on alignment. This function contains
    the functionality shared between monomer and complex EC
    inference.
    
    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
    
    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        raw_ec_file
        model_file
        num_sites
        num_sequences
        effective_sequences

        focus_mode (passed through)
        focus_sequence (passed through)
        segments (passed through)

    """
    check_required(
        kwargs,
        [
            "prefix", "alignment_file",
            "focus_mode", "focus_sequence", "theta",
            "alphabet", "segments", "ignore_gaps", "iterations",
            "lambda_h", "lambda_J", "lambda_group",
            "scale_clusters",
            "cpu", "plmc", "reuse_ecs",
        ]
    )

    prefix = kwargs["prefix"]

    # for now disable option to not save model, since
    # otherwise mutate stage will crash. To remove model
    # file at end, use delete option in management section.
    """
    if kwargs["save_model"]:
        model = prefix + ".model"
    else:
        model = None
    """
    model = prefix + ".model"

    outcfg = {
        "model_file": model,
        "raw_ec_file": prefix + "_ECs.txt",
        "ec_file": prefix + "_CouplingScores.csv",
        # the following are passed through stage...
        "focus_mode": kwargs["focus_mode"],
        "focus_sequence": kwargs["focus_sequence"],
        "segments": kwargs["segments"],
    }

    # make sure input alignment exists
    verify_resources(
        "Input alignment does not exist",
        kwargs["alignment_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # regularization strength on couplings J_ij
    lambda_J = kwargs["lambda_J"]

    segments = kwargs["segments"]
    if segments is not None:
        segments = [
            mapping.Segment.from_list(s) for s in segments
        ]

    # first determine size of alphabet;
    # default is amino acid alphabet
    if kwargs["alphabet"] is None:
        alphabet = ALPHABET_PROTEIN
        alphabet_setting = None
    else:
        alphabet = kwargs["alphabet"]

        # allow shortcuts for protein, DNA, RNA
        if alphabet in ALPHABET_MAP:
            alphabet = ALPHABET_MAP[alphabet]

        # if we have protein alphabet, do not set
        # as plmc parameter since default parameter,
        # has some implementation advantages for focus mode
        if alphabet == ALPHABET_PROTEIN:
            alphabet_setting = None
        else:
            alphabet_setting = alphabet

    # scale lambda_J to proportionally compensate
    # for higher number of J_ij compared to h_i?
    if kwargs["lambda_J_times_Lq"]:
        num_symbols = len(alphabet)

        # if we ignore gaps, there is one character less
        if kwargs["ignore_gaps"]:
            num_symbols -= 1

        # second, determine number of uppercase positions
        # that are included in the calculation
        with open(kwargs["alignment_file"]) as f:
            seq_id, seq = next(read_fasta(f))

        # gap character is by convention first char in alphabet
        gap = alphabet[0]
        uppercase = [
            c for c in seq if c == c.upper() or c == gap
        ]
        L = len(uppercase)

        # finally, scale lambda_J
        lambda_J *= (num_symbols - 1) * (L - 1)

    # run plmc... or reuse pre-exisiting results from previous run
    plm_outcfg_file = prefix + ".couplings_standard_plmc.outcfg"

    # determine if to rerun, only possible if previous results
    # were stored in ali_outcfg_file
    if kwargs["reuse_ecs"] and valid_file(plm_outcfg_file):
        plmc_result = read_config_file(plm_outcfg_file)

        # check if the EC/parameter files are there
        required_files = [outcfg["raw_ec_file"]]

        if outcfg["model_file"] is not None:
            required_files += [outcfg["model_file"]]

        verify_resources(
            "Tried to reuse ECs, but empty or "
            "does not exist",
            *required_files
        )

    else:
        # run plmc binary
        plmc_result = ct.run_plmc(
            kwargs["alignment_file"],
            outcfg["raw_ec_file"],
            outcfg["model_file"],
            focus_seq=kwargs["focus_sequence"],
            alphabet=alphabet_setting,
            theta=kwargs["theta"],
            scale=kwargs["scale_clusters"],
            ignore_gaps=kwargs["ignore_gaps"],
            iterations=kwargs["iterations"],
            lambda_h=kwargs["lambda_h"],
            lambda_J=lambda_J,
            lambda_g=kwargs["lambda_group"],
            cpu=kwargs["cpu"],
            binary=kwargs["plmc"],
        )

        # save iteration table to file
        iter_table_file = prefix + "_iteration_table.csv"
        plmc_result.iteration_table.to_csv(
            iter_table_file
        )

        # turn namedtuple into dictionary to make
        # restarting code nicer
        plmc_result = dict(plmc_result._asdict())

        # then replace table with filename so
        # we can store results in config file
        plmc_result["iteration_table"] = iter_table_file

        # save results of search for possible restart
        write_config_file(plm_outcfg_file, plmc_result)

    # store useful information about model in outcfg
    outcfg.update({
        "num_sites": plmc_result["num_valid_sites"],
        "num_valid_sequences": plmc_result["num_valid_seqs"],
        "effective_sequences": plmc_result["effective_samples"],
        "region_start": plmc_result["region_start"],
    })

    # read and sort ECs
    ecs = pairs.read_raw_ec_file(outcfg["raw_ec_file"])

    if segments is not None:
        # create index mapping
        seg_mapper = mapping.SegmentIndexMapper(
            kwargs["focus_mode"], outcfg["region_start"], *segments
        )

        # apply to EC table
        ecs = mapping.segment_map_ecs(ecs, seg_mapper)

    return outcfg, ecs, segments


def rescore_cn_score_ecs(ecs, segments, outcfg, kwargs, score="cn"):
    """
    Probabilistic rescoring of CN-score based ECS

    Parameters
    ----------
    ecs : pd.DataFrame
        EC table
    segments : list(evcouplings.couplings.mapping.Segment)
        Input segment list
    outcfg : dict
        Current output configuration state of couplings protocol
    kwargs : dict
        Input parameters of couplings protocol
    score : str, optional (default: "cn")
        Target score column to use

    Returns
    -------
    ecs : pd.DataFrame
        Enhanced EC table with probabilities and new score (if applicable)
    outcfg_update : dict
        Additional outputs for stage output configuration, need to be
        merged into outcfg in main protocol
    """
    check_required(
        kwargs,
        [
            "scoring_model", "min_sequence_distance", "theta", "frequencies_file",
        ]
    )

    # None will trigger default behaviour of add_mixture_probability
    # (which currently is "skewnormal")
    scoring_model = kwargs.get("scoring_model", "skewnormal")

    # currently we need to distinguish between full rescoring (score and
    # probability) like with logistic regression model, or just putting
    # probabilities on top of default CN score using add_mixture_probability
    if scoring_model == "logistic_regression":
        scorer = pairs.LogisticRegressionScorer()

        # load amino acid/gap frequencies and conservation info
        freqs = pd.read_csv(kwargs["frequencies_file"])

        num_sites = outcfg["num_sites"]
        min_seq_dist = kwargs["min_sequence_distance"]

        # rescore EC table
        ecs = scorer.score(
            ecs,
            freqs,
            kwargs["theta"],
            outcfg["effective_sequences"],
            num_sites,
            score=score
        )

        # currently only perform quality scoring for single segments
        if segments is None or len(segments) == 1:
            is_longrange = ((ecs.i - ecs.j).abs() >= min_seq_dist).astype(int)
            ecs_lr = ecs.assign(
                longrange_count=is_longrange.cumsum()
            )

            # compute expectation for true positives on all contacts
            expected_positives_all = ecs_lr.query(
                "longrange_count <= @num_sites"
            ).probability.sum()

            expected_positives_longrange = ecs_lr.query(
                "longrange_count <= @num_sites and abs(i - j) >= @min_seq_dist"
            ).probability.sum()

            # store in config
            outcfg_update = {
                "expected_true_ecs_all": float(expected_positives_all),
                "expected_true_ecs_longrange": float(expected_positives_longrange)
            }

    else:
        # add mixture model probability
        ecs = pairs.add_mixture_probability(
            ecs, model=scoring_model
        )

        # put CN score into default score column for more generic
        # downstream score handling
        ecs = ecs.assign(
            score=ecs[score]
        )

        # no update to output config in this case
        outcfg_update = {}

    # sort ECs
    ecs = ecs.sort_values(
        by="score", ascending=False
    )

    return ecs, outcfg_update


def standard(**kwargs):
    """
    Protocol:

    Infer ECs from alignment using plmc. Use complex protocol
    for heteromultimeric complexes instead.

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
        and infer_plmc()

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        raw_ec_file
        model_file
        num_sites
        num_sequences
        effective_sequences

        focus_mode (passed through)
        focus_sequence (passed through)
        segments (passed through)
    """
    # for additional required parameters, see infer_plmc()
    # TODO: make scoring_model mandatory eventually
    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance", "theta", "frequencies_file",
        ]
    )

    prefix = kwargs["prefix"]

    # infer ECs and load them
    outcfg, ecs, segments = infer_plmc(**kwargs)
    model = CouplingsModel(outcfg["model_file"])

    # perform EC rescoring starting from CN score output by plmc;
    # outconfig update will be merged further down in final outcfg merge
    ecs, rescorer_outcfg_update = rescore_cn_score_ecs(
        ecs, segments, outcfg, kwargs, score="cn"
    )

    # following computations are mostly specific to monomer pipeline
    is_single_segment = segments is None or len(segments) == 1
    outcfg = {
        **outcfg,
        **rescorer_outcfg_update,
        **_postprocess_inference(
            ecs, kwargs, model, outcfg, prefix,
            generate_enrichment=is_single_segment,
            generate_line_plot=is_single_segment,
            score="score"
        )
    }

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".couplings_standard.outcfg", outcfg)

    return outcfg


def complex_probability(ecs, scoring_model, use_all_ecs=False,
                        score="cn"):
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
    if use_all_ecs:
        ecs = pairs.add_mixture_probability(
            ecs, model=scoring_model
        )
    else:
        inter_ecs = ecs.query("segment_i != segment_j")
        intra_ecs = ecs.query("segment_i == segment_j")

        intra_ecs = pairs.add_mixture_probability(
            intra_ecs, model=scoring_model, score=score
        )

        inter_ecs = pairs.add_mixture_probability(
            inter_ecs, model=scoring_model, score=score
        )

        ecs = pd.concat(
            [intra_ecs, inter_ecs]
        ).sort_values(
            score, ascending=False
        )

    return ecs


def complex(**kwargs):
    """
    Protocol:

    Infer ECs for protein complexes from alignment using plmc.
    Allows user to select scoring protocol.

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
        and infer_plmc()

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        raw_ec_file
        model_file
        num_sites
        num_sequences
        effective_sequences

        focus_mode (passed through)
        focus_sequence (passed through)
        segments (passed through)
    """
    # for additional required parameters, see infer_plmc()
    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
            "scoring_model", "use_all_ecs_for_scoring",
        ]
    )

    prefix = kwargs["prefix"]

    # infer ECs and load them
    outcfg, ecs, segments = infer_plmc(**kwargs)
    model = CouplingsModel(outcfg["model_file"])

    # following computations are mostly specific to complex pipeline

    # add mixture model probability
    if kwargs["scoring_model"] in SCORING_MODELS:
        if kwargs["use_all_ecs_for_scoring"] is not None:
            use_all_ecs = kwargs["use_all_ecs_for_scoring"]
        else:
            use_all_ecs = False

        ecs = complex_probability(
            ecs, kwargs["scoring_model"], use_all_ecs
        )

    else:
        raise InvalidParameterError(
            "Invalid scoring_model parameter: " +
            "{}. Valid options are: {}".format(
                kwargs["protocol"], ", ".join(SCORING_MODELS)
            )
        )

    # also create line-drawing script (for multiple chains)
    # by convention, we map first segment to chain A,
    # second to B, a.s.f.
    chain_mapping = dict(
        zip(
            [s.segment_id for s in segments],
            string.ascii_uppercase,
        )
    )

    outcfg = {
        **outcfg,
        **_postprocess_inference(
            ecs, kwargs, model, outcfg, prefix,
            generate_line_plot=True,
            generate_enrichment=False,
            ec_filter="segment_i != segment_j or abs(i - j) >= {}",
            chain=chain_mapping
        )
    }
    
    # save just the inter protein ECs
    ## TODO: eventually have this accomplished by _postprocess_inference
    ## right now avoiding a second call with a different ec_filter
    ecs = pd.read_csv(outcfg["ec_file"])
    outcfg["inter_ec_file"] = prefix + "_CouplingScores_inter.csv"
    inter_ecs = ecs.query("segment_i != segment_j")
    inter_ecs.to_csv(outcfg["inter_ec_file"], index=False)

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".couplings_complex.outcfg", outcfg)

    # TODO: make the following complex-ready
    # EC enrichment:
    #
    # 1) think about making EC enrichment complex-ready and add
    # it back here - so far it only makes sense if all ECs are
    # on one segment
    #
    # EVzoom:
    #
    # 1) at the moment, EVzoom will use numbering before remapping
    # we should eventually get this to a point where segments + residue
    # index are displayed on EVzoom
    #
    # 2) note that this will currently use the default mixture model
    # selection for determining the EC cutoff, rather than the selection
    # used for the EC table above

    return outcfg


def mean_field(**kwargs):
    """
    Protocol:

    Infer ECs from alignment using mean field direct coupling analysis.

    For now, mean field DCA can only be run in focus mode, gaps
    included.

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required.

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * raw_ec_file
        * model_file
        * num_sites
        * num_sequences
        * effective_sequences

        * focus_mode (passed through)
        * focus_sequence (passed through)
        * segments (passed through)
    """
    check_required(
        kwargs,
        [
            "prefix", "alignment_file", "segments",
            "focus_mode", "focus_sequence", "theta",
            "pseudo_count", "alphabet",
            "min_sequence_distance", # "save_model",
            "ec_score_type",
        ]
    )

    if not kwargs["focus_mode"]:
        raise InvalidParameterError(
            "For now, mean field DCA can only be run in focus mode."
        )

    prefix = kwargs["prefix"]

    # option to save model disabled
    """
    if kwargs["save_model"]:
        model = prefix + ".model"
    else:
        model = None
    """
    model = prefix + ".model"

    outcfg = {
        "model_file": model,
        "raw_ec_file": prefix + "_ECs.txt",
        "ec_file": prefix + "_CouplingScores.csv",
        # TODO: the following are passed through stage...
        # keep this or unnecessary?
        "focus_mode": kwargs["focus_mode"],
        "focus_sequence": kwargs["focus_sequence"],
        "segments": kwargs["segments"],
    }

    # make sure input alignment exists
    alignment_file = kwargs["alignment_file"]
    verify_resources(
        "Input alignment does not exist",
        kwargs["alignment_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    segments = kwargs["segments"]
    if segments is not None:
        segments = [
            mapping.Segment.from_list(s) for s in segments
        ]

    # determine alphabet
    # default is protein
    if kwargs["alphabet"] is None:
        alphabet = ALPHABET_PROTEIN
    else:
        alphabet = kwargs["alphabet"]

        # allow shortcuts for protein, DNA, RNA
        if alphabet in ALPHABET_MAP:
            alphabet = ALPHABET_MAP[alphabet]

    # read in a2m alignment
    with open(alignment_file) as f:
        input_alignment = Alignment.from_file(
            f, alphabet=alphabet,
            format="fasta"
        )

    # init mean field direct coupling analysis
    mf_dca = MeanFieldDCA(input_alignment)

    # run mean field approximation
    model = mf_dca.fit(
        theta=kwargs["theta"],
        pseudo_count=kwargs["pseudo_count"]
    )

    # write ECs to file
    model.to_raw_ec_file(
        outcfg["raw_ec_file"]
    )

    # write model file
    if outcfg["model_file"] is not None:
        model.to_file(
            outcfg["model_file"],
            file_format="plmc_v2"
        )

    # store useful information about model in outcfg
    outcfg.update({
        "num_sites": model.L,
        "num_valid_sequences": model.N_valid,
        "effective_sequences": float(round(model.N_eff, 1)),
        "region_start": int(model.index_list[0]),
    })

    # read and sort ECs
    # Note: this now deviates from the original EC format
    # file because it has 4 score columns to accomodate
    # MI (raw), MI (APC-corrected), DI, CN;
    ecs = pd.read_csv(
        outcfg["raw_ec_file"], sep=" ",
        names=["i", "A_i", "j", "A_j", "mi_raw", "mi_apc", "di", "cn"]
    )

    # select target score;
    # by default select CN score, since it allows to compute probabilities etc.
    ec_score_type = kwargs.get("ec_score_type", "cn")
    valid_ec_type_choices = ["cn", "di", "mi_raw", "mi_apc"]

    if ec_score_type not in valid_ec_type_choices:
        raise InvalidParameterError(
            "Invalid choice for valid_ec_type: {}, valid options are: {}".format(
                ec_score_type, ", ".join(valid_ec_type_choices)
            )
        )

    # perform rescoring if CN score is selected, otherwise cannot rescore
    # since all models are based on distribution shapes generated by CN score
    if ec_score_type == "cn":
        # perform EC rescoring starting from CN score output by plmc;
        # outconfig update will be merged further down in final outcfg merge

        # returned list is already sorted
        ecs, rescorer_outcfg_update = rescore_cn_score_ecs(
            ecs, segments, outcfg, kwargs, score="cn"
        )
    else:
        # If MI or DI, cannot apply distribution-based rescoring approaches,
        # so just set score column and add dummy probability value for compatibility
        # with downstream code
        ecs = ecs.assign(
            score=ecs[ec_score_type],
            probability=np.nan
        ).sort_values(
            by="score",
            ascending=False
        )

        # no additional values to be updated in outcfg in this case
        rescorer_outcfg_update = {}

    is_single_segment = segments is None or len(segments) == 1
    outcfg = {
        **outcfg,
        **rescorer_outcfg_update,
        **_postprocess_inference(
            ecs, kwargs, model, outcfg, prefix,
            generate_enrichment=is_single_segment,
            generate_line_plot=is_single_segment,
            score="score"
        )
    }

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".couplings_meanfield.outcfg", outcfg)

    return outcfg


def _postprocess_inference(ecs, kwargs, model, outcfg, prefix, generate_line_plot=False,
                           generate_enrichment=False, ec_filter="abs(i - j) >= {}",
                           chain=None, score="cn"):
    """
    Post-process inference result of all protocols

    Parameters
    ----------
    ecs : pandas.DataFrame
        EC table with additional column "probability"
        containing confidence measure
    kwargs arguments:
        See list in protocols.
    model : CouplingsModel
        The couplings model with the inferred parameters
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * raw_ec_file
        * model_file
        * num_sites
        * num_sequences
        * effective_sequences

        * focus_mode (passed through)
        * focus_sequence (passed through)
        * segments (passed through)
    prefix : str
        file path prefix
    generate_line_plot : bool
        Determines whether a line plot pymol structure will be generated
    generate_enrichment : bool
        Determines whether an EC enrichment file and pymol structure will be generated
    ec_filter : str
        String determining the ec distance filter (default: "abs(i - j) >= {}")
    chain : dict
        Dictionary to map different segments to their chains
    score : str, optional (default: "cn")
        Score column to use for postprocessing

    Returns
    -------
    ext_outcfg : dict
        Optional output configuration of the pipeline, including
        the following fields:

        * ec_longrange_file
        * ec_lines_oml_file
        * enrichmnet_file
        * enrichment_pml_files
        * evzoom_file
    """
    ext_outcfg = {}
    # write the sorted ECs table to csv file
    ecs.to_csv(outcfg["ec_file"], index=False)

    # if maximum coupling score is 0, bail out... will crash downstream calculations
    if ecs[score].max() <= 0:
        raise BailoutException("couplings: No couplings identified")

    # also store longrange ECs as convenience output
    if kwargs["min_sequence_distance"] is not None:
        ext_outcfg["ec_longrange_file"] = prefix + "_CouplingScores_longrange.csv"
        ecs_longrange = ecs.query(
            ec_filter.format(kwargs["min_sequence_distance"])
        )
        ecs_longrange.to_csv(ext_outcfg["ec_longrange_file"], index=False)

        if generate_line_plot:
            ext_outcfg["ec_lines_pml_file"] = prefix + "_draw_ec_lines.pml"
            L = outcfg["num_sites"]
            ec_lines_pymol_script(
                ecs_longrange.iloc[:L, :],
                ext_outcfg["ec_lines_pml_file"],
                chain=chain,
                score_column=score
            )

    # compute EC enrichment (for now, for single segments
    # only since enrichment code cannot handle multiple segments)
    if generate_enrichment:
        ext_outcfg["enrichment_file"] = prefix + "_enrichment.csv"

        min_seqdist = kwargs["min_sequence_distance"]
        if min_seqdist is None:
            min_seqdist = 0

        ecs_enriched = pairs.enrichment(
            ecs, score=score, min_seqdist=min_seqdist
        )
        ecs_enriched.to_csv(ext_outcfg["enrichment_file"], index=False)

        # create corresponding enrichment pymol scripts
        ext_outcfg["enrichment_pml_files"] = []
        for sphere_view, pml_suffix in [
            (True, "_enrichment_spheres.pml"), (False, "_enrichment_sausage.pml")
        ]:
            pml_file = prefix + pml_suffix
            enrichment_pymol_script(ecs_enriched, pml_file, sphere_view=sphere_view)
            ext_outcfg["enrichment_pml_files"].append(pml_file)

    # output EVzoom JSON file if we have stored model file
    if outcfg.get("model_file", None) is not None:
        ext_outcfg["evzoom_file"] = prefix + "_evzoom.json"

        # automatically determine reordering of alphabet for EVzoom output
        # (proteins only)
        alphabet = "".join(model.alphabet)

        if alphabet == ALPHABET_PROTEIN_NOGAP:
            reorder = ALPHABET_PROTEIN_NOGAP_ORDERED
        elif alphabet == ALPHABET_PROTEIN:
            reorder = ALPHABET_PROTEIN_ORDERED
        else:
            reorder = None

        with open(ext_outcfg["evzoom_file"], "w") as f:
            # create JSON output and write to file
            # TODO: note that this will by default use CN scores as generated
            # TODO: by CouplingsModel; at the moment there is no easy way
            # TODO: around this limitation so just use CN score for now
            f.write(
                evzoom_json(model, reorder=reorder) + "\n"
            )

    return ext_outcfg

# list of available EC inference protocols

PROTOCOLS = {
    # standard plmc inference protocol
    "standard": standard,

    # runs plmc for protein complexes
    "complex": complex,

    # inference protocol using mean field approximation
    "mean_field": mean_field,
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
        Output configuration of couplings stage
        Dictionary with results in following fields:
        (in brackets: not mandatory)

         ec_file
         effective_sequences
         [enrichment_file]
         focus_mode
         focus_sequence
         model_file
         num_sequences
         num_sites
         raw_ec_file
         region_start
         segments
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
