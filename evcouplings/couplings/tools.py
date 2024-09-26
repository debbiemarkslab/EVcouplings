"""
Wrappers for tools for calculation of evolutionary
couplings from sequence alignments.

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple
import re

import pandas as pd

from evcouplings.utils.system import (
    run, valid_file, create_prefix_folders,
    verify_resources, ResourceError
)


def parse_plmc_log(log):
    """
    Parse plmc stderr text output into structured data

    Parameters
    ----------
    log : str
        stderr output from plmc

    Returns
    -------
    iter_df : pd.DataFrame
        Table with iteration statistics
    focus_index : int
        Index of focus sequence in alignment
    valid_seqs : int
        Number of valid sequences in alignment
    total_seqs : int
        Number of total sequences in alignment
    valid_sites : int
        Analyzed number of sites in alignment/focus sequence
    total_sites : int
        Total number of sites in alignment/focus sequence
    region_start : int
        Index of first position in aligment
    eff_samples : float
        Effective number of samples in alignment
    opt_status : str
        End status of iterative optimization
    """
    res = {
        "focus": re.compile("Found focus (.+) as sequence (\d+)"),
        "seqs": re.compile("(\d+) valid sequences out of (\d+)"),
        "sites": re.compile("(\d+) sites out of (\d+)"),
        "region": re.compile("Region starts at (\d+)"),
        "samples": re.compile("Effective number of samples: (\d+\.\d+)"),
        "optimization": re.compile("Gradient optimization: (.+)")
    }

    re_iter = re.compile("(\d+){}".format(
        "".join(["\s+(\d+\.\d+)"] * 6)
    ))

    matches = {}
    fields = None
    iters = []

    for line in log.split("\n"):
        for (name, pattern) in res.items():
            m = re.search(pattern, line)
            if m:
                matches[name] = m.groups()

        if line.startswith("iter"):
            fields = line.split()

        m_iter = re.search(re_iter, line)
        if m_iter:
            iters.append(m_iter.groups())

    if fields is not None and iters is not None:
        iter_df = pd.DataFrame(iters, columns=fields)
    else:
        iter_df = None

    # some output only defined in focus mode
    focus_index = None
    valid_sites, total_sites = None, None
    region_start = 1

    try:
        focus_index = int(matches["focus"][1])
        valid_sites, total_sites = map(int, matches["sites"])
        region_start = int(matches["region"][0])
    except KeyError:
        pass

    valid_seqs, total_seqs = map(int, matches["seqs"])
    eff_samples = float(matches["samples"][0])
    opt_status = matches["optimization"][0]

    return (
        iter_df,
        (
            focus_index, valid_seqs, total_seqs,
            valid_sites, total_sites, region_start,
            eff_samples, opt_status
        )
    )


# output fields for storing results of a plmc run
# (returned by run_plmc)
PlmcResult = namedtuple(
    "PlmcResult",
    [
        "couplings_file", "param_file",
        "iteration_table", "focus_seq_index",
        "num_valid_seqs", "num_total_seqs",
        "num_valid_sites", "num_total_sites",
        "region_start", "effective_samples",
        "optimization_status"
    ]
)


def run_plmc(alignment, couplings_file, param_file=None,
             focus_seq=None, alphabet=None, theta=None,
             scale=None, ignore_gaps=False, iterations=None,
             lambda_h=None, lambda_J=None, lambda_g=None,
             cpu=None, binary="plmc"):
    """
    Run plmc on sequence alignment and store
    files with model parameters and pair couplings.

    Parameters
    ----------
    alignment : str
        Path to input sequence alignment
    couplings_file : str
        Output path for file with evolutionary couplings
        (folder will be created)
    param_file : str
        Output path for binary file containing model
        parameters (folder will be created)
    focus_seq : str, optional (default: None)
        Name of focus sequence, if None, non-focus mode
        will be used
    alphabet : str, optional (default: None)
        Alphabet for model inference. If None, standard
        amino acid alphabet including gap will be used.
        First character in string corresponds to gap
        character (relevant for ignore_gaps).
    theta : float, optional (default: None)
        Sequences with pairwise identity >= theta
        will be clustered and their sequence weights
        downweighted as 1 / num_cluster_members.
        Important: Note that plmc will be parametrized using
        1 - theta. If None, default value in plmc will be used,
        which corresponds to theta=0.8 (plmc setting 0.2).
    scale : float, optional (default: None)
        Scale weights of clusters by this value.
        If None, default value in plmc (1.0) will be used
    ignore_gaps : bool, optional (default: False)
        Exclude gaps from parameter inference. Gap
        character is first character of alphabet
        parameter.
    iterations : int, optional (default: None)
        Maximum iterations for optimization.
    lambda_h : float, optional (default: None)
        l2 regularization strength on fields.
        If None, plmc default will be used.
    lambda_J : float, optional (default: None)
        l2-regularization strength on couplings.
        If None, plmc default will be used
    lambda_g : float, optional (default: None)
        group l1-regularization strength on couplings
        If None, plmc default will be used.
    cpu : Number of cores to use for running plmc.
        Note that plmc has to be compiled in openmp
        mode to runnable with multiple cores.
        Can also be set to "max".
    binary : str, optional (default: "plmc")
        Path to plmc binary

    Returns
    -------
    PlmcResult
        namedtuple containing output files and
        parsed fields from console output of plmc

    Raises
    ------
    ExternalToolError
    """
    create_prefix_folders(couplings_file)

    # Make sure input alignment exists
    verify_resources(
        "Alignment file does not exist", alignment
    )

    cmd = [
        binary,
        "-c", couplings_file,
    ]

    # store eij file if explicitly requested
    if param_file is not None:
        create_prefix_folders(param_file)
        cmd += ["-o", param_file]

    # focus sequence mode and ID
    if focus_seq is not None:
        # TODO: for now split exclude sequence
        # region from focus seq name, otherwise
        # plmc does not remap names. If this
        # behaviour changes in plmc, remove the
        # following line.
        focus_seq = focus_seq.split("/")[0]
        cmd += ["-f", focus_seq]

    # exclude gaps from calculation?
    if ignore_gaps:
        cmd += ["-g"]

    # maximum number of iterations, can also be "max"
    if iterations is not None:
        cmd += ["-m", str(iterations)]

    # set custom alphabet
    # (first character is gap by default in nogap mode)
    if alphabet is not None:
        cmd += ["-a", alphabet]

    # sequence reweighting
    if theta is not None:
        # transform into plmc convention (1-theta)
        theta = 1.0 - theta
        cmd += ["-t", str(theta)]

    # cluster weight
    if scale is not None:
        cmd += ["-s", str(scale)]

    # L2 regularization weight for fields
    if lambda_h is not None:
        cmd += ["-lh", str(lambda_h)]

    # L2 regularization weight for pair couplings
    if lambda_J is not None:
        cmd += ["-le", str(lambda_J)]

    # Group L1 regularization weight for pair couplings
    if lambda_g is not None:
        cmd += ["-lg", str(lambda_g)]

    # Number of cores to use for calculation
    if cpu is not None:
        cmd += ["-n", str(cpu)]

    # finally also add input alignment (main parameter)
    cmd += [alignment]

    # TODO: for now do not check returncode because sometimes
    # returncode == -11 (segfault) despite successful calculation
    return_code, stdout, stderr = run(cmd, check_returncode=False)

    # TODO: remove this segfault-hunting output if fixed in plmc
    if return_code != 0:
        # check if we got valid output from plmc by parsing it
        valid_plmc_output = True
        try:
            parse_plmc_log(stderr)
        except KeyError:
            valid_plmc_output = False

        # if not a segfault or invalid plmc output, still raise exception
        if return_code != -11 or not valid_plmc_output:
            from evcouplings.utils.system import ExternalToolError
            raise ExternalToolError(
                "Call failed:\ncmd={}\nreturncode={}\nstdout={}\nstderr={}".format(
                    cmd, return_code, stdout, stderr
                )
            )

    iter_df, out_fields = parse_plmc_log(stderr)

    # also check we actually calculated couplings...
    if not valid_file(couplings_file):
        raise ResourceError(
            "plmc returned no couplings: stdout={} stderr={} file={}".format(
                stdout, stderr, couplings_file
            )
        )

    # ... and parameter file, if requested
    if param_file and not valid_file(param_file):
        raise ResourceError(
            "plmc returned no parameter file: stdout={} stderr={} file={}".format(
                stdout, stderr, param_file
            )
        )

    return PlmcResult(
        couplings_file, param_file,
        iter_df, *out_fields
    )

