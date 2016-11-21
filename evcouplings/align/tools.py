"""
Wrappers for running external sequence alignment tools

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple
from evcouplings.utils.system import (
    run, create_prefix_folders, verify_resources
)

# output fields for storing results of a jackhmmer run
# (returned by run_jackhmmer)
JackhmmerResult = namedtuple(
    "JackhmmerResult",
    ["prefix", "alignment", "output", "tblout", "domtblout"]
)


def run_jackhmmer(query, database, prefix,
                  use_bitscores, domain_threshold, seq_threshold,
                  iterations=5, nobias=False, cpu=None,
                  stdout_redirect=None, checkpoints_hmm=False,
                  checkpoints_ali=False, binary="jackhmmer"):
    """
    Run jackhmmer sequence search against target database.
    Refer to HMMER Userguide for explanation of these parameters.

    Parameters
    ----------
    query : str
        File containing query sequence
    database : str
        File containing sequence database
    prefix : str
        Prefix path for output files. Folder structure in
        the prefix will be created if not existing.
    use_bitscores : bool
        Use bitscore inclusion thresholds rather than E-values.
    domain_threshold : int or float or str
        Inclusion threshold applied on the domain level
        (e.g. "1E-03" or 0.001 or 50)
    seq_threshold : int or float or str
        Inclusion threshold applied on the sequence level
        (e.g. "1E-03" or 0.001 or 50)
    iterations : int
        number of jackhmmer search iterations
    nobias : bool, optional (default: False)
        Turn of bias correction
    cpu : int, optional (default: None)
        Number of CPUs to use for search. Uses all if None.
    stdout_redirect : str, optional (default: None)
        Redirect bulky stdout instead of storing
        with rest of results (use "/dev/null" to dispose)
    checkpoints_hmm : bool, optional (default: False)
        Store checkpoint HMMs to prefix.<iter>.hmm
    checkpoints_ali : bool, optional (default: False)
        Store checkpoint alignments to prefix.<iter>.sto
    bin : str (default: "jackhmmer")
        Path to jackhmmer binary (put in PATH for
        default to work)

    Returns
    -------
    JackhmmerResult
        namedtuple with fields corresponding to the different
        output files (prefix, alignment, output, tblout, domtblout)

    Raises
    ------
    ExternalToolError, ResourceError
    """
    verify_resources(
        "Input file does not exist or is empty",
        query, database
    )

    create_prefix_folders(prefix)

    # store filenames of all individual results;
    # these will be returned as result of the
    # function.
    result = JackhmmerResult(
        prefix,
        prefix + ".sto",
        prefix + ".output" if stdout_redirect is None else stdout_redirect,
        prefix + ".tblout",
        prefix + ".domtblout"
    )

    cmd = [
        binary,
        "-N", str(iterations),
        "-o", result.output,
        "-A", result.alignment,
        "--tblout", result.tblout,
        "--domtblout", result.domtblout,
        "--noali",
        "--notextw"
    ]

    # reporting thresholds are set accordingly to
    # inclusion threshold to reduce memory footprit
    if use_bitscores:
        cmd += [
            "-T", str(seq_threshold),
            "--domT", str(domain_threshold),
            "--incT", str(seq_threshold),
            "--incdomT", str(domain_threshold)
        ]
    else:
        cmd += [
            "-E", str(seq_threshold),
            "--domE", str(domain_threshold),
            "--incE", str(seq_threshold),
            "--incdomE", str(domain_threshold)
        ]

    # number of CPUs
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]

    # bias correction filter
    if nobias:
        cmd += ["--nobias"]

    # save checkpoints for alignments and HMMs?
    if checkpoints_ali:
        cmd += ["--chkali", prefix]
    if checkpoints_hmm:
        cmd += ["--chkhmm", prefix]

    cmd += [query, database]

    return_code, stdout, stderr = run(cmd)

    # also check we actually created some sort of alignment
    verify_resources(
        "jackhmmer returned empty alignment: "
        "stdout={} stderr={} file={}".format(
            stdout, stderr, result.alignment
        ),
        result.alignment
    )

    return result


HmmscanResult = namedtuple(
    "HmmscanResult",
    ["prefix", "output", "tblout", "domtblout", "pfamtblout"]
)


def run_hmmscan(query, database, prefix,
                use_model_threshold=True, threshold_type="cut_ga",
                use_bitscores=True, domain_threshold=None, seq_threshold=None,
                nobias=False, cpu=None, stdout_redirect=None, binary="hmmscan"):
    """
    Run hmmscan of HMMs in database against sequences in query
    to identify matches of these HMMs.
    Refer to HMMER Userguide for explanation of these parameters.

    Parameters
    ----------
    query : str
        File containing query sequence
    database : str
        File containing HMM database (prepared with hmmpress)
    prefix : str
        Prefix path for output files. Folder structure in
        the prefix will be created if not existing.
    use_model_threshold: bool (default: True)
        Use model-specific inclusion thresholds from
        HMM database rather than global bitscore/E-value
        thresholds (use_bitscores, domain_threshold and
        seq_threshold are overriden by this flag).
    threshold-type: {"cut_ga", "cut_nc", "cut_tc"} (default: "cut_ga")
        Use gathering (default), noise or trusted cutoff
        to define scan hits. Please refer to HMMER manual for
        details.
    use_bitscores : bool
        Use bitscore inclusion thresholds rather than E-values.
        Overriden by use_model_threshold flag.
    domain_threshold : int or float or str
        Inclusion threshold applied on the domain level
        (e.g. "1E-03" or 0.001 or 50)
    seq_threshold : int or float or str
        Inclusion threshold applied on the sequence level
        (e.g. "1E-03" or 0.001 or 50)
    nobias : bool, optional (default: False)
        Turn of bias correction
    cpu : int, optional (default: None)
        Number of CPUs to use for search. Uses all if None.
    stdout_redirect : str, optional (default: None)
        Redirect bulky stdout instead of storing
        with rest of results (use "/dev/null" to dispose)
    bin : str (default: "hmmscan")
        Path to jackhmmer binary (put in PATH for
        default to work)

    Returns
    -------
    HmmscanResult
        namedtuple with fields corresponding to the different
        output files (prefix, output, tblout, domtblout, pfamtblout)

    Raises
    ------
    ExternalToolError, ResourceError
    """
    verify_resources(
        "Input file does not exist or is empty",
        query, database
    )

    create_prefix_folders(prefix)

    result = HmmscanResult(
        prefix,
        prefix + ".output" if stdout_redirect is None else stdout_redirect,
        prefix + ".tblout",
        prefix + ".domtblout",
        prefix + ".pfamtblout"
    )

    cmd = [
        binary,
        "-o", result.output,
        "--tblout", result.tblout,
        "--domtblout", result.domtblout,
        "--pfamtblout", result.pfamtblout,
        "--notextw",
        "--acc",
    ]

    # number of CPUs
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]

    # bias correction filter
    if nobias:
        cmd += ["--nobias"]

    # either use model-specific threshold, or custom
    # bitscore/E-value thresholds
    if use_model_threshold:
        THRESHOLD_CHOICES = ["cut_ga", "cut_nc", "cut_tc"]
        if threshold_type not in THRESHOLD_CHOICES:
            raise ValueError(
                "Invalid model threshold, valid choices are: " +
                ", ".join(THRESHOLD_CHOICES)
            )

        cmd += ["--" + threshold_type]
    else:
        if seq_threshold is None or domain_threshold is None:
            raise ValueError(
                "Must define sequence- and domain-level reporting"
                "thresholds, or use gathering threshold instead."
            )

        if use_bitscores:
            cmd += [
                "-T", str(seq_threshold),
                "--domT", str(domain_threshold),
            ]
        else:
            cmd += [
                "-E", str(seq_threshold),
                "--domE", str(domain_threshold),
            ]

    cmd += [database, query]

    return_code, stdout, stderr = run(cmd)

    # also check we actually created a table with hits
    verify_resources(
        "hmmscan did not return results: "
        "stdout={} stderr={} file={}".format(
            stdout, stderr, result.domtblout
        ),
        result.domtblout
    )

    return result


def run_hhfilter(input_file, output_file, threshold=95,
                 columns="a2m", binary="hhfilter"):
    """
    Redundancy-reduce a sequence alignment using hhfilter
    from the HHsuite alignment suite.

    Parameters
    ----------
    input_file : str
        Path to input alignment in A2M/FASTA format
    output_file : str
        Path to output alignment (will be in A3M format)
    threshold : int, optional (default: 95)
        Sequence identity threshold for maximum pairwise
        identity (between 0 and 100)
    columns : {"first", "a2m"}, optional (default: "a2m")
        Definition of match columns (based on first sequence
        or upper-case columns (a2m))
    binary : str
        Path to hhfilter binary

    Returns
    -------
    str
        output_file

    Raises
    ------
    ResourceError
        If output alignment is non-existent/empty
    ValueError
        Upon invalid value of columns parameter
    """
    if columns not in ["first", "a2m"]:
        raise ValueError(
            "Invalid column selection: {}".format(columns)
        )

    verify_resources(
        "Alignment file does not exist or is empty",
        input_file
    )

    create_prefix_folders(output_file)

    cmd = [
        binary,
        "-i", input_file,
        "-o", output_file,
        "-id", str(threshold),
        "-M", columns,
        "-v", str(2)
    ]

    return_code, stdout, stderr = run(cmd)

    verify_resources(
        "hhfilter returned empty alignment: "
        "stdout={} stderr={} file={}".format(
            stdout, stderr, output_file
        ),
        output_file
    )

    return output_file
