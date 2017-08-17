"""
Wrappers for tools for 3D structure prediction
from evolutionary couplings

Authors:
  Thomas A. Hopf
"""
from copy import deepcopy
import os
from os import path
from collections import defaultdict
import re

import pandas as pd

from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.system import (
    run, makedirs, temp, verify_resources
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

    env["CNS_SOLVE"] = cns_main_dir
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


def run_psipred(fasta_file, output_dir, binary="runpsipred"):
    """
    Run psipred secondary structure prediction

    psipred output file convention: run_psipred creates
    output files <rootname>.ss2 and <rootname2>.horiz
    in the current working directory, where <rootname>
    is extracted from the basename of the input file
    (e.g. /home/test/<rootname>.fa)

    Parameters
    ----------
    fasta_file : str
        Input sequence file in FASTA format
    output_dir : str
        Directory in which output will be saved
    binary : str, optional (default: "cns")
        Path of psipred executable (runpsipred)

    Returns
    -------
    ss2_file : str
        Absolute path to prediction output in "VFORMAT"
    horiz_file : str
        Absolute path to prediction output in "HFORMAT"

    Raises
    ------
    ExternalToolError
        If call to psipred fails
    """
    # make sure we have absolute path
    binary = path.abspath(binary)
    fasta_file = path.abspath(fasta_file)
    output_dir = path.abspath(output_dir)

    # make sure input file is valid
    verify_resources("Input FASTA file is invalid", fasta_file)

    # make sure output directory exists
    makedirs(output_dir)

    # execute psipred;
    # we have to start it from output directory so
    # result files end up there (this is hardcoded
    # in runpsipred)
    return_code, stdout, stderr = run(
        [binary, fasta_file], working_dir=output_dir,
    )

    # determine where psipred will store output based
    # on logic from runpsipred script
    rootname, _ = path.splitext(path.basename(fasta_file))
    output_prefix = path.join(
        output_dir, rootname
    )

    # construct paths to output files in vertical and horizontal formats
    ss2_file = output_prefix + ".ss2"
    horiz_file = output_prefix + ".horiz"

    # make sure we actually predicted something
    verify_resources(
        "psipred output is invalid", ss2_file, horiz_file
    )

    return ss2_file, horiz_file


def read_psipred_prediction(filename, first_index=1):
    """
    Read a psipred secondary structure prediction file
    in horizontal or vertical format (auto-detected).

    Parameters
    ----------
    filename : str
        Path to prediction output file
    first_index : int, optional (default: 1)
        Index of first position in predicted sequence

    Returns
    -------
    pred : pandas.DataFrame
        Table containing secondary structure prediction,
        with the following columns:

        * i: position

        * A_i: amino acid

        * sec_struct_3state: prediction (H, E, C)

        If reading vformat, also contains columns
        for the individual (score_coil/helix/strand)

        If reading hformat, also contains confidence
        score between 1 and 9 (sec_struct_conf)
    """
    # detect file format
    file_format = None
    with open(filename) as f:
        for line in f:
            if line.startswith("# PSIPRED HFORMAT"):
                file_format = "hformat"
            elif line.startswith("# PSIPRED VFORMAT"):
                file_format = "vformat"

    if file_format == "vformat":
        # read in output file
        pred = pd.read_csv(
            filename,
            skip_blank_lines=True, comment="#",
            delim_whitespace=True,
            names=[
                "i", "A_i", "sec_struct_3state",
                "score_coil", "score_helix", "score_strand"
            ],
        )
    elif file_format == "hformat":
        content = defaultdict(str)
        with open(filename) as f:
            # go through file and assemble Conf, Pred, and AA lines
            # into single strings
            for line in f:
                line = line.rstrip().replace(" ", "")
                if ":" in line:
                    key, _, value = line.partition(":")
                    content[key] += value

        pred = pd.DataFrame({
            "A_i": list(content["AA"]),
            "sec_struct_3state": list(content["Pred"]),
            "sec_struct_conf": list(map(int, content["Conf"])),
        })
        pred.loc[:, "i"] = list(range(1, len(pred) + 1))
    else:
        raise InvalidParameterError(
            "Input file is not a valid psipred prediciton file"
        )

    # shift indices if first_index != 1
    pred.loc[:, "i"] += (first_index - 1)

    return pred


def parse_maxcluster_comparison(comparison_output):
    """
    Parse maxcluster output into a DataFrame
    
    Parameters
    ----------
    comparison_output : str
        stdout output from maxcluster after comparison
    
    Returns
    -------
    pandas.DataFrame
        Parsed result table (columns: filename, num_pairs,
        rmsd, maxsub, tm, msi), refer to maxcluster documentation
        for explanation of the score fields.
    """
    # compile regular expression to extract output fields
    m = re.compile(
        "vs. (.+?)\s+Pairs=\s*(\d+), RMSD=\s*(\d+\.\d+), "
        "MaxSub=\s*(\d+\.\d+), TM=\s*(\d+\.\d+), MSI=\s*(\d+\.\d+)"
    )

    # extract scores for each structure (one per line)
    res = []
    for line in comparison_output.splitlines():
        match = m.search(line)
        if match:
            res.append(match.groups())

    # create dataframe of results
    df = pd.DataFrame(
        res, columns=[
            "filename", "num_pairs", "rmsd",
            "maxsub", "tm", "msi"
        ]
    )

    # convert score columns to numerical values
    for c in df.columns:
        if c != "filename":
            df.loc[:, c] = pd.to_numeric(df.loc[:, c])

    df.loc[:, "num_pairs"] = df.loc[:, "num_pairs"].astype(int)

    return df


def run_maxcluster_compare(predictions, experiment, normalization_length=None,
                           distance_cutoff=None, binary="maxcluster"):
    """
    Compare a set of predicted structures to an experimental structure
    using maxcluster.
    
    For clustering functionality, use run_maxcluster_clustering() function.
    
    For a high-level wrapper around this function that removes
    problematic atoms and compares multiple models, please look at 
    evcouplings.fold.protocol.compare_models_maxcluster().
    
    Parameters
    ----------
    predictions : list(str)
        List of PDB files that should be compared against experiment
    experiment : str
        Path of experimental structure PDB file. Note that the numbering
        and residues in this file must agree with the predicted structure,
        and that the structure may not contain duplicate atoms (multiple
        models, or alternative locations for the same atom).
    normalization_length : int, optional (default: None)
        Use this length to normalize the Template Modeling (TM)
        score (-N option of maxcluster). If None, will normalize
        by length of experiment.
    distance_cutoff : float, optional (default: None)
        Distance cutoff for MaxSub search (-d option of maxcluster).
        If None, will use maxcluster auto-calibration.
    binary : str, optional (default: "maxcluster")
        Path to maxcluster binary

    Returns
    -------
    pandas.DataFrame
        Comparison result table (see parse_maxcluster_comparison
        for more detailed explanation)
    """
    # create a list of files for input to maxcluster
    list_file = temp()
    with open(list_file, "w") as f:
        for pred_file in predictions:
            f.write(pred_file + "\n")

    cmd = [binary, "-l", list_file, "-e", experiment]

    # normalization length for TM score calculation
    if normalization_length is not None:
        cmd += ["-N", str(normalization_length)]

    # distance cutoff for MaxSub search
    if distance_cutoff is not None:
        cmd += ["-d", str(distance_cutoff)]

    return_code, stdout, stderr = run(cmd)

    return parse_maxcluster_comparison(stdout)


def parse_maxcluster_clustering(clustering_output):
    """
    Parse maxcluster clustering output into a DataFrame

    Parameters
    ----------
    clustering_output : str
        stdout output from maxcluster after clustering

    Returns
    -------
    pandas.DataFrame
        Parsed result table (columns: filename, cluster, cluster_size)
    """
    # compile regular expression to extract output fields
    m = re.compile("INFO\s*:\s*(\d+)\s*:\s*(\d+)\s+(.+)")

    # extract scores for each structure (one per line)
    res = []
    read = False
    for line in clustering_output.splitlines():
        # only parse section where cluster for each structure is output
        if "Clusters @ Threshold" in line:
            read = True

        if "Centroids" in line:
            read = False

        match = m.search(line)
        if read and match:
            res.append(match.groups())

    # turn results into table
    df = pd.DataFrame(res, columns=["item", "cluster", "filename"])

    # add column containing the size of each cluster
    cluster_sizes = df.groupby(
        "cluster"
    ).size().to_frame("cluster_size").reset_index()

    df = df.merge(cluster_sizes, on="cluster")

    return df.loc[:, ["filename", "cluster", "cluster_size"]]


def run_maxcluster_cluster(predictions, method="average", rmsd=True,
                           clustering_threshold=None, binary="maxcluster"):
    """
    Compare a set of predicted structures to an experimental structure
    using maxcluster.

    For clustering functionality, use run_maxcluster_clustering() function.

    Parameters
    ----------
    predictions : list(str)
        List of PDB files that should be compared against experiment
    method : {"single", "average", "maximum", "pairs_min", "pairs_abs"}, optional (default: "average")
        Clustering method (single / average / maximum linkage,
        or min / absolute size neighbour pairs
    clustering_threshold : float (optional, default: None)
        Initial clustering threshold (maxcluster -T option)
    rmsd : bool, optional (default: True)
        Use RMSD-based clustering (faster)
    binary : str, optional (default: "maxcluster")
        Path to maxcluster binary

    Returns
    -------
    pandas.DataFrame
        Clustering result table (see parse_maxcluster_clustering
        for more detailed explanation)
    """
    # create a list of files for input to maxcluster
    list_file = temp()
    with open(list_file, "w") as f:
        for pred_file in predictions:
            f.write(pred_file + "\n")

    method_map = {
        "single": 1,
        "average": 2,
        "maximum": 3,
        "pairs_min": 4,
        "pairs_abs": 5,
    }

    if method not in method_map:
        raise InvalidParameterError(
            "Method must be one of the following: " +
            ", ".join(method_map.keys())
        )

    cmd = [binary, "-l", list_file, "-C", str(method_map[method])]

    if rmsd:
        cmd += ["-rmsd"]

    if clustering_threshold is not None:
        cmd += ["-T", str(clustering_threshold)]

    return_code, stdout, stderr = run(cmd)

    return parse_maxcluster_clustering(stdout)
