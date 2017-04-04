"""
Protocols for predicting protein 3D structure from ECs

Authors:
  Thomas A. Hopf
"""

from evcouplings.utils.config import (
    check_required, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources
)


def cns_dgsa(**kwargs):
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

        mutation_matrix_file
        [mutation_dataset_predicted_file]
    """
    check_required(
        kwargs,
        [
            "prefix", "ec_file", "target_sequence_file",
            "segments", "config_file",
        ]
    )

    # make sure model file exists
    verify_resources(
        "Model parameter file does not exist",
        kwargs["model_file"]
    )

    prefix = kwargs["prefix"]

    # make sure output directory exists
    create_prefix_folders(prefix)

    outcfg = {}

    # TODO: allow to reuse secondary structure,
    # TODO: or have external file;
    # TODO: parallelize using multiprocessing
    # TODO: create indextable-like file here
    # TODO: implement folding protocol here
    # TODO: cut to different parts of query sequence

    return outcfg


# list of available folding protocols
PROTOCOLS = {
    # standard EVfold protocol
    "cns_dgsa": cns_dgsa,
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
