"""
Evolutionary couplings calculation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

from evcouplings.couplings import tools as ct
from evcouplings.couplings import pairs, mapping

from evcouplings.align.alignment import (
    read_fasta, ALPHABET_PROTEIN, ALPHABET_DNA,
    ALPHABET_RNA
)

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    read_config_file, write_config_file
)

from evcouplings.utils.system import (
    create_prefix_folders, valid_file,
    verify_resources,
)

ALPHABET_MAP = {
    "aa": ALPHABET_PROTEIN,
    "dna": ALPHABET_DNA,
    "rna": ALPHABET_RNA,
}


def standard(**kwargs):
    """
    Protocol:

    Infer ECs from alignment using plmc.

    # TODO:
    (1) make EC enrichment calculation segment-ready

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
        (TODO: explain meaning of parameters in detail).

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
            "cpu", "plmc", "save_model", "reuse_ecs",
        ]
    )

    prefix = kwargs["prefix"]

    if kwargs["save_model"]:
        model = prefix + ".model"
    else:
        model = None

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
    verify_resources(
        "Input alignment does not exist",
        kwargs["alignment_file"]
    )

    # make sure output directory exists
    # TODO: Exception handling here if this fails
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

    # add mixture model probability
    ecs = pairs.add_mixture_probability(ecs)

    if segments is not None:  # and (len(segments) > 1 or not kwargs["focus_mode"]):
        # create index mapping
        seg_mapper = mapping.SegmentIndexMapper(
            kwargs["focus_mode"], outcfg["region_start"], *segments
        )

        # apply to EC table
        ecs = mapping.segment_map_ecs(ecs, seg_mapper)

    # write updated table to csv file
    ecs.to_csv(outcfg["ec_file"], index=False)

    # compute EC enrichment (for now, for single segments
    # only since enrichment code cannot handle multiple segments)
    if segments is None or len(segments) == 1:
        outcfg["enrichment_file"] = prefix + "_enrichment.csv"
        ecs_enriched = pairs.enrichment(ecs)
        ecs_enriched.to_csv(outcfg["enrichment_file"], index=False)

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".couplings_standard.outcfg", outcfg)

    return outcfg


# list of available EC inference protocols
PROTOCOLS = {
    # standard plmc inference protocol
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
