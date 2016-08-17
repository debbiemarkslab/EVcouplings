"""
Evolutionary couplings calculation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

import evcouplings.couplings.tools as ct
import evcouplings.couplings.pairs as pairs

from evcouplings.align.alignment import (
    read_fasta, ALPHABET_PROTEIN
)

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    write_config_file
)

from evcouplings.utils.system import (
    create_prefix_folders, verify_resources
)


def standard(**kwargs):
    """
    Protocol:

    Infer ECs from alignment using plmc.

    # TODO:
    (1) auto-infer alphabet from segments?
    (2) make alphabet an output of alignment stage?
    (3) remapping based on segments, e.g. for complexes

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
            "cpu", "plmc", "save_model",
        ]
    )

    prefix = kwargs["prefix"]

    if kwargs["save_model"]:
        model = prefix + ".eij"
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

    # scale lambda_J to proportionally compensate
    # for higher number of J_ij compared to h_i?
    if kwargs["lambda_J_times_Lq"]:
        # first determine size of alphabet;
        # plmc default is amino acid alphabet
        if kwargs["alphabet"] is None:
            alphabet = ALPHABET_PROTEIN
        else:
            alphabet = kwargs["alphabet"]

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

    # now run plmc binary
    plmc_result = ct.run_plmc(
        kwargs["alignment_file"],
        outcfg["raw_ec_file"],
        outcfg["model_file"],
        focus_seq=kwargs["focus_sequence"],
        alphabet=kwargs["alphabet"],
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

    # read and sort ECs, write to csv file
    ecs = pairs.read_raw_ec_file(outcfg["raw_ec_file"])
    ecs.to_csv(outcfg["ec_file"], index=False)

    segments = kwargs["segments"]
    if segments is not None and len(segments) > 1:
        # TODO: implement remapping to individual segments
        # (may differ between focusmode and non-focusmode)
        raise NotImplementedError(
            "Segment remapping not yet implemented."
        )

    # store useful information about model in outcfg or files
    plmc_result.iteration_table.to_csv(
        prefix + "_iteration_table.txt"
    )

    outcfg.update({
        "num_sites": plmc_result.num_valid_sites,
        "num_sequences": plmc_result.num_valid_seqs,
        "effective_sequences": plmc_result.effective_samples,
    })

    # dump output config to YAML file for debugging/logging
    write_config_file(prefix + ".infer_standard.outcfg", outcfg)

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
    # TODO

    Dictionary with results of stage in following fields:
    (in brackets: not returned by all protocols)
        # TODO
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
