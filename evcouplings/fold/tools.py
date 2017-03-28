"""
Wrappers for tools for 3D structure prediction
from evolutionary couplings

Authors:
  Thomas A. Hopf
"""

from evcouplings.utils.system import (
    run, valid_file, create_prefix_folders,
    verify_resources, ResourceError
)


def run_cns(input_script, log_file=None, source_script=None, binary="cns"):
    """
    Run CNSsolve 1.3

    Note that the user is responsible for verifying the output products
    of CNS, since their paths are determined by .inp scripts and
    hard to check automatically and in a general way.

    Parameters
    ----------
    input_script : str
        Path to .inp input script
    log_file : str, optional (default: None)
        Save CNS stdout output to this file
    source_script : str, optional (default: None)
        Script to set CNS environment variables.
        This should typically point to .cns_solve_env_sh
        in the CNS installation main directory (the
        shell script itself needs to be edited to
        contain the path of the installation)
    binary : str, optional (default: "cns")
        Name of CNS binary

    Raises
    ------
    ExternalToolError
        If call to CNS fails
    """
    # usually need to source script to set up environment for CNS
    if source_script is not None:
        cmd = "source {};".format(source_script)
    else:
        cmd = ""

    cmd += binary

    # read input script, this is fed into CNS using stdin
    with open(input_script) as f:
        inp = "".join(f.readlines())

    # run and store output
    return_code, stdout, stderr = run(cmd, stdin=inp, shell=True)

    # write stdout output to log file
    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(stdout)
