"""
Pipelining of different stages of method

Authors:
  Thomas A. Hopf
"""

# chose backend for command-line usage
import matplotlib
matplotlib.use("Agg")

import signal
import os
from os import path
import sys
import traceback
import tarfile

import click

from evcouplings.utils.config import (
    read_config_file, check_required, write_config_file,
    InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, insert_dir, verify_resources,
    valid_file
)
from evcouplings.utils.database import (
    update_job_status, EStatus
)

import evcouplings.align.protocol as ap
import evcouplings.couplings.protocol as cp
import evcouplings.compare.protocol as cm
import evcouplings.mutate.protocol as mt
import evcouplings.fold.protocol as fd
import evcouplings.complex.protocol as pp

# supported pipelines
#
# stages are defined by:
# 1) name of stage
# 2) function to execute for stage
# 3) key prefix (to avoid name collisions
#    of output fields if same stage is run
#    multiple times, e.g. 2 alignments for
#    complexes)
PIPELINES = {
    "protein_monomer": [
        ("align", ap.run, None),
        ("couplings", cp.run, None),
        ("compare", cm.run, None),
        ("mutate", mt.run, None),
        ("fold", fd.run, None),
    ],
    "protein_complex": [
        ("align_1", ap.run, "first_"),
        ("align_2", ap.run, "second_"),
        ("concatenate", pp.run, None),
        ("couplings", cp.run, None),
        ("compare", cm.run, None),
    ]
}

# suffix of file containing final output configuration of pipeline
FINAL_CONFIG_SUFFIX = "_final.outcfg"

# suffix of file that will be generated if execution
# is terminated externally (SIGINT, SIGTERM, ...)
EXTENSION_TERMINATED = ".terminated"

# suffix of file that will be generated if execution
# fails internally (i.e. exception is raised)
EXTENSION_FAILED = ".failed"

# suffix of file that will be generated if execution
# runs through sucessfully
EXTENSION_DONE = ".done"


def execute(**config):
    """
    Execute a pipeline configuration

    Parameters
    ----------
    **config
        Input configuration for pipeline
        (see pipeline config files for
        example of how this should look like)

    Returns
    -------
    global_state : dict
        Global output state of pipeline
    """
    check_required(
        config,
        ["pipeline", "stages", "global"]
    )

    # check if valid pipeline was selected
    if config["pipeline"] not in PIPELINES:
        raise InvalidParameterError(
            "Not a valid pipeline selection. "
            "Valid choices are:\n{}".format(
                ", ".join(PIPELINES.keys())
            )
        )

    stages = config["stages"]
    if stages is None:
        raise InvalidParameterError(
            "No stages defined, need at least one."
        )

    # get definition of selected pipeline
    pipeline = PIPELINES[config["pipeline"]]
    prefix = config["global"]["prefix"]

    # make sure output directory exists
    create_prefix_folders(prefix)

    # this is the global state of results as
    # we move through different stages of
    # the pipeline
    global_state = config["global"]

    # keep track of how many stages are still
    # to be run, so we can leave out stages at
    # the end of workflow below
    num_stages_to_run = len(stages)

    # set job status to running
    update_job_status(config, status=EStatus.RUN)

    # iterate through individual stages
    for (stage, runner, key_prefix) in pipeline:
        # check if anything else is left to
        # run, otherwise skip
        if num_stages_to_run == 0:
            break

        # check if config for stage is there
        check_required(config, [stage])

        # output files for stage into an individual folder
        stage_prefix = insert_dir(prefix, stage)
        create_prefix_folders(stage_prefix)

        # config files for input and output of stage
        stage_incfg = "{}_{}.incfg".format(stage_prefix, stage)
        stage_outcfg = "{}_{}.outcfg".format(stage_prefix, stage)

        # update current stage of job
        update_job_status(config, stage=stage)

        # check if stage should be executed
        if stage in stages:
            # global state inserted at end, overrides any
            # stage-specific settings (except for custom prefix)
            incfg = {
                **config["tools"],
                **config["databases"],
                **config[stage],
                **global_state,
                "prefix": stage_prefix
            }
            # save input of stage in config file
            write_config_file(stage_incfg, incfg)

            # run stage
            outcfg = runner(**incfg)

            # prefix output keys if this parameter is
            # given in stage configuration, to avoid
            # name clashes if same protocol run multiple times
            if key_prefix is not None:
                outcfg = {
                    key_prefix + k: v for k, v in outcfg.items()
                }

            # save output of stage in config file
            write_config_file(stage_outcfg, outcfg)

            # one less stage to put through after we ran this...
            num_stages_to_run -= 1
        else:
            # skip state by injecting state from previous run
            verify_resources(
                "Trying to skip, but output configuration "
                "for stage '{}' does not exist. Has it already "
                "been run?".format(stage, stage),
                stage_outcfg
            )

            # read output configuration
            outcfg = read_config_file(stage_outcfg)

            # verify all the output files are there
            outfiles = [
                filepath for f, filepath in outcfg.items()
                if f.endswith("_file") and filepath is not None
            ]

            verify_resources(
                "Output files from stage '{}' "
                "missing".format(stage),
                *outfiles
            )

        # update global state with outputs of stage
        global_state = {**global_state, **outcfg}

    # create results archive
    archive_file = prefix + ".tar.gz"
    create_archive(config, global_state, archive_file)
    global_state["archive_file"] = archive_file

    # delete selected output files if requested
    global_state = delete_outputs(config, global_state)

    # write final global state of pipeline
    write_config_file(
        prefix + FINAL_CONFIG_SUFFIX, global_state
    )

    # set job status to done
    update_job_status(config, status=EStatus.DONE)

    return global_state


def create_archive(config, outcfg, output_file):
    """
    Create archive of files generated by pipeline

    Parameters
    ----------
    config : dict-like
        Input configuration of job. Uses 
        config["management"]["archive"] (list of key 
        used to index outcfg) to determine
        which files should be added to archive
    outcfg : dict-like
        Output configuration of job
    output_file : str
        Store archive file to this path
    """
    # determine keys (corresponding to files) in
    # outcfg that should be stored
    outkeys = config.get("management", {}).get("archive", None)

    # if no output keys are requested, nothing to do
    if outkeys is None or len(outkeys) == 0:
        return

    # create archive
    with tarfile.open(output_file, "w:gz") as tar:
        # add files based on keys one by one
        for k in outkeys:
            # skip missing keys or ones not defined
            if k not in outcfg or outcfg[k] is None:
                continue

            # distinguish between files and lists of files
            if k.endswith("files"):
                for f in outcfg[k]:
                    if valid_file(f):
                        tar.add(f)
            else:
                if valid_file(outcfg[k]):
                    tar.add(outcfg[k])


def delete_outputs(config, outcfg):
    """
    Remove pipeline outputs to save memory
    after running the job

    Parameters
    ----------
    config : dict-like
        Input configuration of job. Uses 
        config["management"]["delete"] (list of key 
        used to index outcfg) to determine
        which files should be added to archive
    outcfg : dict-like
        Output configuration of job
    
    Returns
    -------
    outcfg_cleaned : dict-like
        Output configuration with selected
        output keys removed.
    """
    # determine keys (corresponding to files) in
    # outcfg that should be stored
    outkeys = config.get("management", {}).get("delete", None)

    # if no output keys are requested, nothing to do
    if outkeys is None:
        return outcfg

    # go through all flagged files and delete if existing
    for k in outkeys:
        # skip missing keys or ones not defined
        if k not in outcfg or k is None:
            continue

        # delete list of files
        if k.endswith("files"):
            for f in outcfg[k]:
                try:
                    os.remove(f)
                except OSError:
                    pass
            del outcfg[k]

        # delete individual file
        else:
            try:
                os.remove(outcfg[k])
                del outcfg[k]
            except OSError:
                pass

    return outcfg


def verify_prefix(verify_subdir=True, **config):
    """
    Check if configuration contains a prefix,
    and that prefix is a valid directory we
    can write to on the filesystem
    
    Parameters
    ----------
    verify_subdir : bool, optional (default: True)
        Check if we can create subdirectory containing
        full prefix. Set this to False for outer evcouplings
        app loop.
    **config
        Input configuration for pipeline
        
    Returns
    -------
    prefix : str
        Verified prefix
    """
    # check we have a prefix entry, otherwise all hope is lost...
    try:
        prefix = config["global"]["prefix"]
    except KeyError:
        raise InvalidParameterError(
            "Configuration does not include 'prefix' setting in "
            "'global' section"
        )

    # make sure prefix is also specified
    if prefix is None:
        raise InvalidParameterError(
            "'prefix' must be specified and cannot be None"
        )

    # verify that prefix is workable in terms
    # of filesystem
    try:
        # make prefix folder
        create_prefix_folders(prefix)

        # try if we can write in the folder
        with open(prefix + ".test__", "w") as f:
            pass

        # get rid of the file again
        os.remove(prefix + ".test__")

        if verify_subdir:
            # make sure we can create a subdirectory
            sub_prefix = insert_dir(prefix, "test__")
            create_prefix_folders(sub_prefix)

            # remove again
            os.rmdir(path.dirname(sub_prefix))

    except OSError as e:
        raise InvalidParameterError(
            "Not a valid prefix: {}".format(prefix)
        ) from e

    return prefix


def execute_wrapped(**config):
    """
    Execute a pipeline configuration in "wrapped"
    mode that handles external interruptions and
    exceptions and documents these using files
    (.finished, .terminated, .failed), as well
    as documenting failure in a job database

    Parameters
    ----------
    **config
        Input configuration for pipeline
        (see pipeline config files for
        example of how this should look like)

    Returns
    -------
    outcfg : dict
        Global output state of pipeline
    """
    # make sure the prefix in configuration is valid
    try:
        prefix = verify_prefix(**config)
    except Exception:
        update_job_status(config, status=EStatus.TERM)
        raise

    # delete terminated/failed flags from previous
    # executions of pipeline
    for ext in [
        EXTENSION_FAILED, EXTENSION_TERMINATED, EXTENSION_DONE,
    ]:
        try:
            os.remove(prefix + ext)
        except OSError:
            pass

    # handler for external interruptions
    # (needs config for database access)
    def _handler(signal_, frame):
        # set job status to terminated in database
        update_job_status(config, status=EStatus.TERM)

        # create file flag that job was terminated
        with open(prefix + ".terminated", "w") as f:
            f.write("SIGNAL: {}\n".format(signal_))

        # terminate program
        sys.exit(1)

    # set up handlers for job termination
    # (note that this list may not be complete and may
    # need extension for other computing environments)
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2]:
        signal.signal(sig, _handler)

    try:
        # execute configuration
        outcfg = execute(**config)

        # if we made it here, job was sucessfully run to completing
        # create file flag that job was terminated
        with open(prefix + ".finished", "w") as f:
            f.write(repr(outcfg))

        return outcfg

    except Exception as e:
        # set status in database to failed
        update_job_status(config, status=EStatus.FAIL)

        # create failed file flag
        with open(prefix + ".failed", "w") as f:
            f.write(traceback.format_exc())

        # raise exception again after we updated status
        raise


def run(**kwargs):
    """
    EVcouplings pipeline execution from a
    configuration file (single thread, no
    batch or environment configuration)
    
    Parameters
    ----------
    kwargs
        See click.option decorators for app()
    """
    config_file = kwargs["config"]
    verify_resources(
        "Config file does not exist or is empty.",
        config_file
    )

    # read configuration and execute
    config = read_config_file(config_file)

    # execute configuration in "wrapped" mode
    # that handles exceptions and internal interrupts
    return execute_wrapped(**config)


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config')
def app(**kwargs):
    """
    Command line app entry point
    """
    # execute configuration file
    outcfg = run(**kwargs)

    # print final result configuration to stdout
    print(outcfg)

if __name__ == '__main__':
    app()
