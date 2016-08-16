"""
Evolutionary couplings calculation protocols/workflows.

Authors:
  Thomas A. Hopf
"""

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    write_config_file
)

from evcouplings.utils.system import (
    create_prefix_folders
)


def standard(**kwargs):
    """
    Protocol:

    Infer ECs from alignment using plmc.

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
        (TODO: explain meaning of parameters in detail).

    If skip is given and True, the workflow will only return
    the output configuration (outcfg) and ali will be None.

    If callback is given, the function will be called at the
    end of the workflow with the kwargs arguments updated with
    the outcfg results.

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        raw_ec_file
        eij_file
        focus_mode?
        focus_sequence?
        segments?
    """
    check_required(
        kwargs,
        [
            "prefix", "alignment_file",
            "focus_mode", "focus_sequence",
        ]
    )

    prefix = kwargs["prefix"]
    outcfg = {
        "eij_file": prefix + ".eij",
        "raw_ec_file": prefix + "_ECs.txt",
        "ec_file": prefix + "_CouplingScores.csv"
    }

    # Otherwise, now run the protocol...
    # make sure output directory exists
    # TODO: Exception handling here if this fails
    create_prefix_folders(prefix)

    # TODO: implement actual protocol

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
