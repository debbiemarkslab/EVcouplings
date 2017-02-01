"""
evcouplings command-line app

Authors:
  Thomas A. Hopf

# TODO: Once there are different pipelines to run, will
        need to use subcommands to differentiate
"""

import re
from copy import deepcopy

import click

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
        "theta": ("couplings", "theta"),
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

    # handle the more complicated parameters

    # If alignment is given, run "existing" protocol
    if kwargs["alignment"] is not None:
        # TODO: think about what to do if sequence_file is given
        # (will not be used)
        config["align"]["protocol"] = "existing"

    # subregion of protein
    if kwargs["region"] is not None:
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
    if kwargs["stages"] is not None:
        config["stages"] = kwargs["stages"].replace(
            " ", ""
        ).split(",")

    # sequence alignment input database
    if kwargs["database"] is not None:
        db = kwargs["database"]
        # check if we have a predefined sequence database
        # if so, use it; otherwise, interpret as file path
        if db in config["databases"]:
            config["align"]["database"] = db
        else:
            config["align"]["database"] = "custom"
            config["databases"]["custom"] = db

    # make sure bitscore and E-value thresholds are exclusively set
    if kwargs["bitscores"] is not None and kwargs["evalues"] is not None:
        raise InvalidParameterError(
            "Can not specify bitscore and E-value threshold at the same time."
        )

    if kwargs["bitscores"] is not None:
        thresholds = kwargs["bitscores"]
        bitscore = True
    elif kwargs["evalues"] is not None:
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
    jobs : list
        List of paths to individual config files
        that can be fed into pipeline one by one
    """
    # get global prefix of run and create folder
    prefix = config["global"]["prefix"]
    create_prefix_folders(prefix)

    # check if we have a single job or need to unroll
    # into multiple jobs
    if config["batch"] is None:
        cfg_filename = CONFIG_NAME.format(prefix)
        write_config_file(cfg_filename, config)
        jobs = [cfg_filename]
    else:
        jobs = []
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

            # in case subprefices might be folder, create it
            create_prefix_folders(sub_prefix)

            # apply subconfig delta
            # (assuming parameters are nested in two layers)
            for section in delta_config:
                for param, value in delta_config[section].items():
                    sub_config[section][param] = value

            # save config for individual job
            subcfg_filename = CONFIG_NAME.format(sub_prefix)
            write_config_file(subcfg_filename, sub_config)

            jobs.append(subcfg_filename)

    return jobs


def run_jobs(config_files):
    """
    Submit config to pipeline
    """
    print(config_files)
    return


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
    help="Existing sequence alignment to start from (aligned FASTA/Stockholm)"
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
def app(**kwargs):
    """
    EVcouplings command line interface

    Any command line option specified in addition to the config file
    will overwrite the corresponding setting in the config file.

    Specifying a list of bitscores or E-values will result in the creation
    of multiple jobs that only vary in this parameter, with all other parameters
    constant.
    """
    # substitute commmand line options in config file
    config = substitute_config(**kwargs)

    # check minimal set of parameters is present in config
    check_required(
        config,
        ["pipeline", "stages", "global"]
    )

    # for convenience, turn on N_eff computation if we run alignment,
    # but not the couplings stage
    if "align" in config["stages"] and "couplings" not in config["stages"]:
        config["align"]["compute_num_effective_seqs"] = True

    # unroll batch jobs into individual pipeline jobs
    jobs = unroll_config(config)

    # run pipeline computation for each individual (unrolled) config
    run_jobs(jobs)


if __name__ == '__main__':
    app()
