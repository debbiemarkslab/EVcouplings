"""
Wrappers for running external sequence alignment tools

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple
from evcouplings.utils.system import (
    run, ExternalToolError, file_not_empty, create_prefix_folders
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
    ExternalToolError
    """
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
        cmd += ["cpu", str(cpu)]

    # bias correction filter
    if nobias:
        cmd += ["--nobias"]

    # save checkpoints for alignments and HMMs?
    if checkpoints_ali:
        cmd += ["--chkali", prefix]
    if checkpoints_hmm:
        print("checkpointing")
        cmd += ["--chkhmm", prefix]

    cmd += [query, database]

    return_code, stdout, stderr = run(cmd)

    # make sure return code is okay
    if return_code != 0:
        raise ExternalToolError(
            "jackhmmer run failed: returncode={} stdout={} stderr={}".format(
                return_code, stdout, stderr
            )
        )

    # also check we actually created some sort of alignment
    if not file_not_empty(result.alignment):
        raise ExternalToolError(
            "jackhmmer returned empty alignment: stdout={} stderr={} file={}".format(
                stdout, stderr, result.alignment
            )
        )

    return result
