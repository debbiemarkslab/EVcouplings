"""
Wrappers for tools for 3D structure prediction
from evolutionary couplings

Authors:
  Thomas A. Hopf
"""
from copy import deepcopy
import os
from os import path

from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.system import (
    run, valid_file, create_prefix_folders,
    verify_resources, ResourceError
)


def run_cns(inp_script=None, inp_file=None, log_file=None, binary="cns"):
    """
    Run CNSsolve 1.21 (without worrying about environment setup)

    Note that the user is responsible for verifying the output products
    of CNS, since their paths are determined by .inp scripts and
    hard to check automatically and in a general way.

    Either input_script or input_file has to be specified.

    Parameters
    ----------
    inp_script : str, optional (default: None)
        CNS ".inp" input script (actual commands, not file)
    inp_file : str, optional (default: None)
        Path to .inp input script file. Will override
        inp_script if also specified.
    log_file : str, optional (default: None)
        Save CNS stdout output to this file
    binary : str, optional (default: "cns")
        Absolute path of CNS binary

    Raises
    ------
    ExternalToolError
        If call to CNS fails
    InvalidParameterError
        If no input script (file or string) given
    """
    # make sure we have absolute path
    binary = path.abspath(binary)

    # extract main installation directory
    cns_main_dir = binary
    for i in range(3):
        cns_main_dir = path.dirname(cns_main_dir)

    # create environment
    env = deepcopy(os.environ)
    library_dir = path.join(cns_main_dir, "libraries")
    module_dir = path.join(cns_main_dir, "modules")
    env["CNS_LIB"] = library_dir
    env["CNS_MODULE"] = module_dir
    env["CNS_HELPLIB"] = path.join(cns_main_dir, "helplip")

    for var, subdir in [
        ("CNS_TOPPAR", "toppar"),
        ("CNS_CONFDB", "confdb"),
        ("CNS_XTALLIB", "xtal"),
        ("CNS_NMRLIB", "nmr"),
        ("CNS_XRAYLIB", "xray"),
    ]:
        env[var] = path.join(library_dir, subdir)

    for var, subdir in [
        ("CNS_XTALMODULE", "xtal"),
        ("CNS_NMRMODULE", "nmr"),
    ]:
        env[var] = path.join(module_dir, subdir)

    if inp_script is None and inp_file is None:
        raise InvalidParameterError(
            "Must specify either input_script or input_file"
        )

    # read input script, this is fed into CNS using stdin
    if inp_file is not None:
        with open(inp_file) as f:
            inp_script = "".join(f.readlines())

    # run and store output
    return_code, stdout, stderr = run(
        binary, stdin=inp_script
    )

    # write stdout output to log file
    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(stdout)


def run_cns_13(inp_script=None, inp_file=None, log_file=None,
               source_script=None, binary="cns"):
    """
    Run CNSsolve 1.3

    Note that the user is responsible for verifying the output products
    of CNS, since their paths are determined by .inp scripts and
    hard to check automatically and in a general way.

    Either input_script or input_file has to be specified.

    Parameters
    ----------
    inp_script : str, optional (default: None)
        CNS ".inp" input script (actual commands, not file)
    inp_file : str, optional (default: None)
        Path to .inp input script file. Will override
        inp_script if also specified.
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
    InvalidParameterError
        If no input script (file or string) given
    """
    # usually need to source script to set up environment for CNS
    if source_script is not None:
        cmd = "source {};".format(source_script)
    else:
        cmd = ""

    cmd += binary

    if inp_script is None and inp_file is None:
        raise InvalidParameterError(
            "Must specify either input_script or input_file"
        )

    # read input script, this is fed into CNS using stdin
    if inp_file is not None:
        with open(inp_file) as f:
            inp_script = "".join(f.readlines())

    # run and store output
    return_code, stdout, stderr = run(
        cmd, stdin=inp_script, shell=True
    )

    # write stdout output to log file
    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(stdout)
