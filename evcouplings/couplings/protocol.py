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
    read_fasta, ALPHABET_PROTEIN, ALPHABET_DNA,
    ALPHABET_RNA, Alignment
)

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
        "num_sequences": plmc_result["num_valid_seqs"],
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
    check_required(
        kwargs,
        [
            "prefix", "min_sequence_distance",
        ]
    )

    prefix = kwargs["prefix"]

    # infer ECs and load them
    outcfg, ecs, segments = infer_plmc(**kwargs)
    model = CouplingsModel(outcfg["model_file"])

    # add mixture model probability
    ecs = pairs.add_mixture_probability(ecs)

    # following computations are mostly specific to monomer pipeline
    is_single_segment = segments is None or len(segments) == 1
    outcfg = {
        **outcfg,
        **_postprocess_inference(
            ecs, kwargs, model, outcfg, prefix,
            generate_enrichment=is_single_segment,
            generate_line_plot=is_single_segment
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
        ecs = pairs.add_mixture_proability(
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
        "num_sequences": model.N_valid,
        "effective_sequences": float(round(model.N_eff, 1)),
        "region_start": int(model.index_list[0]),
    })

    # read and sort ECs
    ecs = pd.read_csv(
        outcfg["raw_ec_file"], sep=" ",
        # for now, call the last two columns
        # "fn" and "cn" to prevent compare
        # stage from crashing
        names=["i", "A_i", "j", "A_j", "fn", "cn"]
        # names=["i", "A_i", "j", "A_j", "mi", "di"]
    ).sort_values(
        by="cn",
        ascending=False
    )

    is_single_segment = segments is None or len(segments) == 1
    outcfg = {
        **outcfg,
        **_postprocess_inference(
            ecs, kwargs, model, outcfg, prefix,
            generate_enrichment=is_single_segment,
            generate_line_plot=is_single_segment
        )
    }

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".couplings_meanfield.outcfg", outcfg)

    return outcfg


def _postprocess_inference(ecs, kwargs, model, outcfg, prefix, generate_line_plot=False,
                           generate_enrichment=False, ec_filter="abs(i - j) >= {}", chain=None):
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
                score_column="cn"  # "di
            )

    # compute EC enrichment (for now, for single segments
    # only since enrichment code cannot handle multiple segments)
    if generate_enrichment:
        ext_outcfg["enrichment_file"] = prefix + "_enrichment.csv"
        ecs_enriched = pairs.enrichment(ecs, score="cn")  # "di"
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
        with open(ext_outcfg["evzoom_file"], "w") as f:
            # create JSON output and write to file
            f.write(
                evzoom_json(model) + "\n"
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
