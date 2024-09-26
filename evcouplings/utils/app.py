"""
evcouplings command-line app

Authors:
  Thomas A. Hopf

.. todo::

    Once there are different pipelines to run, there should
    be individual commands for these, so will need to define additional
    entry points for applications (e.g. evcomplex in addition to evcouplings).
"""

import re
from copy import deepcopy
from os import path, environ
from collections.abc import Mapping

import billiard
import click

from evcouplings import utils
from evcouplings.utils import pipeline
from evcouplings.utils.tracker import (
    get_result_tracker, EStatus
)

from evcouplings.utils.system import (
    create_prefix_folders, ResourceError, valid_file
)

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    read_config_file, write_config_file
)

# store individual config files in files with this name
CONFIG_NAME = "{}_config.txt"


def substitute_config(**kwargs):
    """
    Substitute command line arguments into config file

    Parameters
    ----------
    **kwargs
        Command line parameters to be substituted
        into configuration file

    Returns
    -------
    dict
        Updated configuration
    """
    # mapping of command line parameters to config file entries
    CONFIG_MAP = {
        "prefix": ("global", "prefix"),
        "protein": ("global", "sequence_id"),
        "seqfile": ("global", "sequence_file"),
        "alignment": ("align", "input_alignment"),
        "iterations": ("align", "iterations"),
        "id": ("align", "seqid_filter"),
        "seqcov": ("align", "minimum_sequence_coverage"),
        "colcov": ("align", "minimum_column_coverage"),
        "theta": ("global", "theta"),
        "plmiter": ("couplings", "iterations"),
        "queue": ("environment", "queue"),
        "time": ("environment", "time"),
        "cores": ("environment", "cores"),
        "memory": ("environment", "memory"),
    }

    # try to read in configuration
    config_file = kwargs["config"]
    if not valid_file(config_file):
        raise ResourceError(
            "Config file does not exist or is empty: {}".format(
                config_file
            )
        )

    config = read_config_file(config_file, preserve_order=True)

    # substitute command-line parameters into configuration
    # (if straightforward substitution)
    for param, value in kwargs.items():
        if param in CONFIG_MAP and value is not None:
            outer, inner = CONFIG_MAP[param]
            config[outer][inner] = value

    # make sure that number of CPUs requested by
    # programs within pipeline does not exceed
    # number of cores requested in environment
    if config["environment"]["cores"] is not None:
        config["global"]["cpu"] = config["environment"]["cores"]

    # handle the more complicated parameters

    # If alignment is given, run "existing" protocol
    if kwargs.get("alignment", None) is not None:
        # TODO: think about what to do if sequence_file is given
        # (will not be used)
        config["align"]["protocol"] = "existing"

    # subregion of protein
    if kwargs.get("region", None) is not None:
        region = kwargs["region"]
        m = re.search("(\d+)-(\d+)", region)
        if m:
            start, end = map(int, m.groups())
            config["global"]["region"] = [start, end]
        else:
            raise InvalidParameterError(
                "Region string does not have format "
                "start-end (e.g. 5-123):".format(
                    region
                )
            )

    # pipeline stages to run
    if kwargs.get("stages", None) is not None:
        config["stages"] = kwargs["stages"].replace(
            " ", ""
        ).split(",")

    # sequence alignment input database
    if kwargs.get("database", None) is not None:
        db = kwargs["database"]
        # check if we have a predefined sequence database
        # if so, use it; otherwise, interpret as file path
        if db in config["databases"]:
            config["align"]["database"] = db
        else:
            config["align"]["database"] = "custom"
            config["databases"]["custom"] = db

    # make sure bitscore and E-value thresholds are exclusively set
    if kwargs.get("bitscores", None) is not None and kwargs.get("evalues", None) is not None:
        raise InvalidParameterError(
            "Can not specify bitscore and E-value threshold at the same time."
        )

    if kwargs.get("bitscores", None) is not None:
        thresholds = kwargs["bitscores"]
        bitscore = True
    elif kwargs.get("evalues", None) is not None:
        thresholds = kwargs["evalues"]
        bitscore = False
    else:
        thresholds = None

    if thresholds is not None:
        T = thresholds.replace(" ", "").split(",")
        try:
            x_cast = [
                (float(t) if "." in t else int(t)) for t in T
            ]
        except ValueError:
            raise InvalidParameterError(
                "Bitscore/E-value threshold(s) must be numeric: "
                "{}".format(thresholds)
            )

        config["align"]["use_bitscores"] = bitscore

        # check if we have a single threshold (single job)
        # or if we need to create an array of jobs
        if len(x_cast) == 1:
            config["align"]["domain_threshold"] = x_cast[0]
            config["align"]["sequence_threshold"] = x_cast[0]
        else:
            config["batch"] = {}
            for t in x_cast:
                sub_prefix = ("_b" if bitscore else "_e") + str(t)
                config["batch"][sub_prefix] = {
                    "align": {
                        "domain_threshold": t,
                        "sequence_threshold": t,
                    }
                }

    return config


def unroll_config(config):
    """
    Create individual job configs from master config file
    (e.g. containing batch section)

    Parameters
    ----------
    config : dict
        Global run dictionary that will be split
        up into individual pipeline jobs

    Returns
    -------
    configs : dict
        Dictionary of prefix to individual configurations
        created by substitution from input configuration.
        If no batch section is present, there will be only
        one entry in the dictionary that corresponds to
        the master run specified by the input configuration.
    """
    # get global prefix of run
    prefix = config["global"]["prefix"]

    # store unrolled configurations here
    configs = {}

    # check if we have a single job or need to unroll
    # into multiple jobs
    if config.get("batch", None) is None:
        configs[prefix] = config
    else:
        # go through all specified runs
        for sub_id, delta_config in config["batch"].items():
            # create copy of config and update for current subjob
            sub_config = deepcopy(config)

            # create prefix of subjob (may contain / to route
            # subjob output to subfolder)
            sub_prefix = prefix + sub_id

            # these are not batch jobs anymore, so deactivate section
            sub_config["batch"] = None

            # create full prefix for subjob
            sub_config["global"]["prefix"] = sub_prefix

            # apply subconfig delta
            # (assuming parameters are nested in two layers)
            for section in delta_config:
                # if dictionary, substitute all items on second level
                if isinstance(delta_config[section], Mapping):
                    for param, value in delta_config[section].items():
                        sub_config[section][param] = value
                else:
                    # substitute entire section (this only affects pipeline stages)
                    sub_config[section] = delta_config[section]

            configs[sub_prefix] = sub_config

    return configs


def run_jobs(configs, global_config, overwrite=False, workdir=None, abort_on_error=True, environment=None):
    """
    Submit config to pipeline

    Parameters
    ----------
    configs : dict
        Configurations for individual subjobs
    global_config : dict
        Master configuration (if only one job,
        the contents of this dictionary will be
        equal to the single element of config_files)
    overwrite : bool, optional (default: False)
        If True, allows overwriting previous run of the same
        config, otherwise will fail if results from previous
        execution are present
    workdir : str, optional (default: None)
        Workdir in which to run job (will combine
        workdir and prefix in joint path)
    abort_on_error : bool, optional (default: True)
        Abort entire job submission if error occurs for
        one of the jobs by propagating RuntimeError
    environment : str, optional (default: None)
        Allow to pass value for environment parameter
        of submitter, will override environment.configuration
        from global_config (e.g., for setting environment
        variables like passwords)

    Returns
    -------
    job_ids : dict
        Mapping from subjob prefix (keys in configs parameter)
        to identifier returned by submitter for each of the jobs
        that was *successfully* submitted (i.e. missing keys from
        configs param indicate these jobs could not be submitted).

    Raises
    ------
    RuntimeError
        If error encountered during submission and abort_on_error
        is True
    """
    cmd_base = environ.get("EVCOUPLINGS_RUNCFG_APP") or "evcouplings_runcfg"
    summ_base = environ.get("EVCOUPLINGS_SUMMARIZE_APP") or "evcouplings_summarize"

    # determine output directory for config files
    prefix = global_config["global"]["prefix"]

    # integrate working directory into output prefix
    # if it is given; if prefix contains an absolute path,
    # this will override the workdir according to
    # implementation of path.join()
    if workdir is not None:
        out_prefix = path.join(workdir, prefix)
    else:
        out_prefix = prefix

    # save configuration file, make sure we do not overwrite previous run
    # if overwrite protection is activated
    # (but only if it is a valid configuration file with contents)
    cfg_filename = CONFIG_NAME.format(out_prefix)

    if not overwrite and valid_file(cfg_filename):
        raise InvalidParameterError(
            "Existing configuration file {} ".format(cfg_filename) +
            "indicates current prefix {} ".format(prefix) +
            "would overwrite existing results. Use --yolo " +
            "flag to deactivate overwrite protection (e.g. for "
            "restarting a job or running a different stage)."
        )

    # make sure working directory exists
    create_prefix_folders(cfg_filename)

    # write global config file
    write_config_file(cfg_filename, global_config)

    # also write individual subjob configuration files
    # (we have to write these before submitting, since
    # the job summarizer needs the paths to all files)
    for subjob_prefix, subjob_cfg in configs.items():
        # determine working dir for each subjob, since subjob
        # prefix may contain slashes leading to subfolder creation
        if workdir is not None:
            subjob_out_prefix = path.join(workdir, subjob_prefix)
        else:
            subjob_out_prefix = subjob_prefix

        subcfg_filename = CONFIG_NAME.format(subjob_out_prefix)

        # make sure output subfolder exists
        create_prefix_folders(subcfg_filename)

        # write subjob configuration file
        write_config_file(subcfg_filename, subjob_cfg)

    # now create list of subjob config files relative to working
    # directory (above, we allow to run submitted in arbitrary directory)
    config_files = [
        CONFIG_NAME.format(subjob_prefix) for subjob_prefix in configs
    ]

    # create command for summarizer (needs to know all subjob config files)
    summ_cmd = "{} {} {} {}".format(
        summ_base,
        global_config["pipeline"],
        global_config["global"]["prefix"],
        " ".join(config_files)
    )

    # create submitter from global (pre-unrolling) configuration
    submitter_cfg = global_config["environment"]
    submitter_engine = submitter_cfg["engine"]
    submitter_cores = submitter_cfg.get("cores")

    # special treatment for local submitter - allow to choose
    # how many jobs to run in parallel (normally defined by grid engine)
    submitter_kws = {}

    # requires that number of cores per job is defined or external tools might
    # request all CPUs for each subjob
    if submitter_engine == "local" and submitter_cores is not None:
        # check which value was set in config for number of parallel workers, default to None
        max_parallel_workers = submitter_cfg.get("parallel_workers")

        # if not defined, calculate
        if max_parallel_workers is None:
            max_cores = billiard.cpu_count()
            max_parallel_workers = int(max_cores / submitter_cores)

        # do not request more workers than needed for number of subjobs
        num_workers = min(len(configs), max_parallel_workers)
        submitter_kws = {"ncpu": num_workers}

    submitter = utils.SubmitterFactory(
        submitter_engine,
        db_path=out_prefix + "_job_database.txt",
        **submitter_kws
    )

    # collect individual submitted jobs here
    commands = []

    # record subjob IDs returned by submitter for each job
    job_ids = {}

    # prepare individual jobs for submission
    for job, job_cfg in configs.items():
        job_prefix = job_cfg["global"]["prefix"]
        job_cfg_file = CONFIG_NAME.format(job)

        # create submission command
        env = job_cfg["environment"]
        cmd = utils.Command(
            [
                "{} {}".format(cmd_base, job_cfg_file),
                summ_cmd
            ],
            name=job_prefix,
            environment=environment or env["configuration"],
            workdir=workdir,
            resources={
                utils.EResource.queue: env["queue"],
                utils.EResource.time: env["time"],
                utils.EResource.mem: env["memory"],
                utils.EResource.nodes: env["cores"],
                utils.EResource.out: job_prefix + "_stdout.log",
                utils.EResource.error: job_prefix + "_stderr.log",
            }
        )

        # store job for later dependency creation
        commands.append(cmd)

        tracker = get_result_tracker(job_cfg)

        try:
            # finally, submit job
            current_job_id = submitter.submit(cmd)

            # store run identifier returned by submitter
            # TODO: consider storing current_job_id using tracker right away
            job_ids[job] = current_job_id

            # set job status in database to pending
            tracker.update(status=EStatus.PEND)

        except RuntimeError as e:
            # set job as failed in database
            tracker.update(status=EStatus.FAIL, message=str(e))

            # fail entire job submission if requested
            if abort_on_error:
                raise

    # submit final summarizer
    # (hold for now - summarizer is run after each subjob finishes)

    # wait for all runs to finish (but only if blocking)
    submitter.join()

    # return job identifiers
    return job_ids


def run(**kwargs):
    """
    Exposes command line interface as a Python function.
    
    Parameters
    ----------
    kwargs
        See click.option decorators for app() function 
    """
    # substitute commmand line options in config file
    config = substitute_config(**kwargs)

    # check minimal set of parameters is present in config
    check_required(
        config,
        ["pipeline", "stages", "global"]
    )

    # verify that global prefix makes sense
    pipeline.verify_prefix(verify_subdir=False, **config)

    # for convenience, turn on N_eff computation if we run alignment,
    # but not the couplings stage
    if "align" in config["stages"] and "couplings" not in config["stages"]:
        config["align"]["compute_num_effective_seqs"] = True

    # unroll batch jobs into individual pipeline jobs
    sub_configs = unroll_config(config)

    # run pipeline computation for each individual (unrolled) config
    run_jobs(
        sub_configs, config, kwargs.get("yolo", False),
        kwargs.get("workdir", None)
    )


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
# run settings
@click.argument('config')
@click.option("-P", "--prefix", default=None, help="Job prefix")
@click.option("-S", "--stages", default=None, help="Stages of pipeline to run (comma-separated)")
@click.option("-p", "--protein", default=None, help="Sequence identifier of query protein")
@click.option("-s", "--seqfile", default=None, help="FASTA file with query sequence")
@click.option(
    "-a", "--alignment", default=None,
    help="Existing sequence alignment to start from (aligned FASTA/Stockholm). Use -p to select target sequence."
)
@click.option("-r", "--region", default=None, help="Region of query sequence(e.g 25-341)")
@click.option(
    "-b", "--bitscores", default=None,
    help="List of alignment bitscores (comma-separated, length-normalized "
         "(float) or absolute score (int))"
)
@click.option(
    "-e", "--evalues", default=None,
    help="List of alignment E-values (negative exponent, comma-separated)"
)
@click.option(
    "-n", "--iterations", default=None, help="Number of alignment iterations", type=int
)
@click.option("-d", "--database", default=None, help="Path or name of sequence database")
@click.option(
    "-i", "--id", default=None, help="Filter alignment at x% sequence identity", type=int
)
@click.option(
    "-f", "--seqcov", default=None, help="Minimum % aligned positions per sequence", type=int
)
@click.option(
    "-m", "--colcov", default=None, help="Minimum % aligned positions per column", type=int
)
@click.option(
    "-t", "--theta", default=None,
    help="Downweight sequences above this identity cutoff"
         " during inference (e.g. 0.8 for 80% identity cutoff)",
    type=float
)
@click.option(
    "--plmiter", default=None, help="Maximum number of iterations during inference",
    type=int
)
# environment configuration
@click.option("-Q", "--queue", default=None, help="Grid queue to run job(s)")
@click.option(
    "-T", "--time", default=None, help="Time requirement (hours) for batch jobs", type=int
)
@click.option("-N", "--cores", default=None, help="Number of cores for batch jobs", type=int)
@click.option(
    "-M", "--memory", default=None, help="Memory requirement for batch jobs (MB or 'auto')"
)
@click.option(
    "-y", "--yolo", default=False, is_flag=True, help="Disable overwrite protection"
)
def app(**kwargs):
    """
    EVcouplings command line interface

    Any command line option specified in addition to the config file
    will overwrite the corresponding setting in the config file.

    Specifying a list of bitscores or E-values will result in the creation
    of multiple jobs that only vary in this parameter, with all other parameters
    constant.
    """
    run(**kwargs)


if __name__ == '__main__':
    app()
