from evcouplings.align import read_hmmer_domtbl
import pandas as pd
from collections import namedtuple
from evcouplings.utils.system import (
    run, create_prefix_folders, verify_resources, temp
)

from os import path


# output fields for storing results of a hmmbuild run
# (returned by run_jackhmmer)
HmmbuildResult = namedtuple(
    "HmmbuildResult",
    ["prefix", "hmmfile", "output"]
)


def run_hmmbuild(msafile, prefix,
                  cpu=None,
                  stdout_redirect=None,
                  binary="hmmbuild"):
    """
    Profile HMM construction from multiple sequence alignments
    Refer to HMMER documentation for details.

    http://eddylab.org/software/hmmer3/3.1b2/Userguide.pdf

    Parameters
    ----------
    msafile : str
        File containing the multiple sequence alignment
    prefix : str
        Prefix path for output files. Folder structure in
        the prefix will be created if not existing.
    hmmfile_out : str
        File containing the profile HMM
    cpu : int, optional (default: None)
        Number of CPUs to use for search. Uses all if None.
    stdout_redirect : str, optional (default: None)
        Redirect bulky stdout instead of storing
        with rest of results (use "/dev/null" to dispose)
    binary : str (default: "hmmbuild")
        Path to jackhmmer binary (put in PATH for
        default to work)

    Returns
    -------
    HmmbuildResult
        namedtuple with fields corresponding to the different
        output files (prefix, alignment, output, tblout, domtblout)

    Raises
    ------
    ExternalToolError, ResourceError
    """
    verify_resources(
        "Input file does not exist or is empty",
        msafile
    )

    create_prefix_folders(prefix)

    # store filenames of all individual results;
    # these will be returned as result of the
    # function.
    result = HmmbuildResult(
        prefix,
        prefix + ".hmm",
        prefix + ".output" if stdout_redirect is None else stdout_redirect,
    )

    cmd = [
        binary,
        "-o", result.output,
    ]

    # number of CPUs
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]

    cmd += [result.hmmfile, msafile]

    return_code, stdout, stderr = run(cmd)

    # also check we actually created some sort of alignment
    verify_resources(
        "hmmbuild returned empty HMM profile: "
        "stdout={} stderr={} file={}".format(
            stdout, stderr, result.hmmfile
        ),
        result.hmmfile
    )

    return result


# output fields for storing results of a hmmsearch run
# (returned by run_hmmsearch)
HmmsearchResult = namedtuple(
    "HmmsearchResult",
    ["prefix", "alignment", "output", "tblout", "domtblout"]
)

def run_hmmsearch(hmmfile, database, prefix,
                  use_bitscores, domain_threshold, seq_threshold,
                  nobias=False, cpu=None,
                  stdout_redirect=None, binary="hmmsearch"):
    """
    Search profile(s) against a sequence database.
    Refer to HMMER documentation for details.

    http://eddylab.org/software/hmmer3/3.1b2/Userguide.pdf

    Parameters
    ----------
    hmmfile : str
        File containing the profile(s)
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
    nobias : bool, optional (default: False)
        Turn of bias correction
    cpu : int, optional (default: None)
        Number of CPUs to use for search. Uses all if None.
    stdout_redirect : str, optional (default: None)
        Redirect bulky stdout instead of storing
        with rest of results (use "/dev/null" to dispose)
    binary : str (default: "hmmsearch")
        Path to jackhmmer binary (put in PATH for
        default to work)

    Returns
    -------
    HmmsearchResult
        namedtuple with fields corresponding to the different
        output files (prefix, alignment, output, tblout, domtblout)

    Raises
    ------
    ExternalToolError, ResourceError
    """
    verify_resources(
        "Input file does not exist or is empty",
        hmmfile, database
    )

    create_prefix_folders(prefix)

    # store filenames of all individual results;
    # these will be returned as result of the
    # function.
    result = HmmsearchResult(
        prefix,
        prefix + ".sto",
        prefix + ".output" if stdout_redirect is None else stdout_redirect,
        prefix + ".tblout",
        prefix + ".domtblout"
    )

    cmd = [
        binary,
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

    cmd += [hmmfile, database]

    return_code, stdout, stderr = run(cmd)

    # also check we actually created some sort of alignment
    verify_resources(
        "hmmsearch returned empty alignment: "
        "stdout={} stderr={} file={}".format(
            stdout, stderr, result.alignment
        ),
        result.alignment
    )

    return result



def hmmbuild_and_search(database, prefix,
                  use_bitscores, domain_threshold, seq_threshold,
                  nobias=False, cpu=None,
                  stdout_redirect=None,
                  binary=["hmmbuild", "hmmsearch"]):
    """
    Search profile(s) against a sequence database.
    Refer to HMMER documentation for details.

    http://eddylab.org/software/hmmer3/3.1b2/Userguide.pdf

    Parameters
    ----------
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
    nobias : bool, optional (default: False)
        Turn of bias correction
    cpu : int, optional (default: None)
        Number of CPUs to use for search. Uses all if None.
    stdout_redirect : str, optional (default: None)
        Redirect bulky stdout instead of storing
        with rest of results (use "/dev/null" to dispose)
    binary : str (default: ["hmmbuild", "hmmsearch"])
        Path to jackhmmer binary (put in PATH for
        default to work)

    Returns
    -------
    HmmsearchResult
        namedtuple with fields corresponding to the different
        output files (prefix, alignment, output, tblout, domtblout)

    Raises
    ------
    ExternalToolError, ResourceError
    """
    verify_resources(
        "Input file does not exist or is empty",
        database
    )

    # name of the protein (prefix)
    protein_name = path.basename(prefix)

    # path to the main directory
    main_dir = path.normpath(path.dirname(prefix) + '/../..')

    # path to the compare directory
    compare_dir = path.join(main_dir, 'compare')
    
    # path to the align directory
    align_dir = path.join(main_dir, 'align')    


    # path to the directory to save the hmmfile in
    hmmfile = path.join(compare_dir, protein_name + ".hmm")

    # path to the msa file
    msafile = path.join(align_dir, protein_name + ".a2m")

    # create a hmm file from a2m file if
    # hmm file doesn't exist
    if not path.isfile(hmmfile):
        run_hmmbuild(msafile, path.join(compare_dir, protein_name), cpu,
                stdout_redirect, binary = binary[0])

    # running hmmsearch binary with the hmmfile
    result = run_hmmsearch(hmmfile, database, prefix,
                use_bitscores, domain_threshold,
                seq_threshold, nobias, cpu, stdout_redirect,
                binary = binary[1])

    return result


