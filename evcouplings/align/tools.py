"""
Wrappers for running external sequence alignment tools

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple
import pandas as pd
from evcouplings.utils.system import (
    run, create_prefix_folders, verify_resources, temp
)



# output fields for storing results of a hmmbuild run
# (returned by run_hmmbuild)
HmmbuildResult = namedtuple(
    "HmmbuildResult",
    ["prefix", "hmmfile", "output"]
)


def run_hmmbuild(msafile, prefix, cpu=None,
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



def run_hmmbuild_and_search(database, prefix,
                  use_bitscores, domain_threshold, seq_threshold,
                  nobias=False, cpu=None,
                  stdout_redirect=None,
                  hmmbuild="hmmbuild", hmmsearch="hmmsearch"):
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
    hmmbuild : str (default: "hmmbuild")
        Path to hmmbuild binary
    hmmsearch : str (default: "hmmsearch")
        Path to hmmsearch binary

    Returns
    -------
    outcfg : dict
        Output configuration of the protocol, including
        the following fields:

        * target_sequence_file
        * sequence_file
        * raw_alignment_file
        * hittable_file
        * focus_mode
        * focus_sequence
        * segments

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
        run_hmmbuild(
            msafile, path.join(compare_dir, protein_name), cpu,
            stdout_redirect, binary=hmmbuild
        )

    # running hmmsearch binary with the hmmfile
    result = run_hmmsearch(
                hmmfile, database, prefix,
                use_bitscores, domain_threshold,
                seq_threshold, nobias, cpu, stdout_redirect,
                binary=hmmsearch
            )

    return result


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
    binary : str (default: "jackhmmer")
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
        File containing query sequence(s)
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
    binary : str (default: "hmmscan")
        Path to hmmscan binary (put in PATH for
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


def _read_hmmer_table(filename, column_names):
    """
    Parse a HMMER file in (dom)tbl format into
    a pandas DataFrame.

    (Why this is necessary: cannot easily split on
    whitespace with pandas because of last column
    that contains whitespace both in header and rows)

    Parameters
    ----------
    filename : str
        Path of (dom)tbl file
    column_names : list of str
        Columns in the respective format
        (different for tbl and domtbl)

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed (dom)tbl
    """
    res = []
    num_splits = len(column_names) - 1

    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.rstrip().split(maxsplit=num_splits)
            res.append(fields)

    # at the moment, all fields in dataframe are strings, even
    # if numeric. To convert to numbers, cheap trick is to store
    # to csv file and let pandas guess the types, rather than
    # going through convert_objects (deprecated) or to_numeric
    # (more effort)
    tempfile = temp()
    pd.DataFrame(
        res, columns=column_names
    ).to_csv(tempfile, index=False)

    return pd.read_csv(tempfile)


def read_hmmer_tbl(filename):
    """
    Read a HMMER tbl file into DataFrame.

    Parameters
    ----------
    filename : str
        Path of tbl file

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed tbl
    """
    column_names = [
        "target_name", "target_accession",
        "query_name", "query_accession",
        "full_Evalue", "full_score", "full_bias",
        "best_domain_Evalue", "best_domain_score",
        "best_domain_bias",
        "domain_exp", "domain_reg", "domain_clu",
        "domain_ov", "domain_env", "domain_dom",
        "domain_rep", "domain_inc",
        "description"
    ]

    return _read_hmmer_table(filename, column_names)


def read_hmmer_domtbl(filename):
    """
    Read a HMMER domtbl file into DataFrame.

    Parameters
    ----------
    filename : str
        Path of domtbl file

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed domtbl
    """
    column_names = [
        "target_name", "target_accession", "target_len",
        "query_name", "query_accession", "query_len",
        "full_Evalue", "full_score", "full_bias",
        "hit_number", "total_hit_number",
        "domain_c_Evalue", "domain_i_Evalue",
        "domain_score", "domain_bias",
        "hmm_from", "hmm_to",
        "ali_from", "ali_to",
        "env_from", "env_to",
        "acc", "description"
    ]

    return _read_hmmer_table(filename, column_names)


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
