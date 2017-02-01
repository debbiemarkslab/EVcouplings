"""
Pipelining of different stages of method

Authors:
  Thomas A. Hopf
"""

from sys import argv, exit, stderr

import click

from evcouplings.utils.config import (
    read_config_file, check_required, write_config_file
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources
)

import evcouplings.align.protocol as ap
import evcouplings.couplings.protocol as cp

# supported pipelines
PIPELINES = {
    "protein_monomer": [
        ("align", ap.run),
        ("couplings", cp.run),
    ]
}


def execute(**kwargs):
    """
    Execute a pipeline configuration

    Parameters
    ----------
    **kwargs
        Input configuration for pipeline
        (see pipeline config files for
        example of how this should look like)

    Returns
    -------
    global_state : dict
        Global output state of pipeline
    """
    check_required(
        kwargs,
        ["pipeline", "stages", "global"]
    )

    # check if valid pipeline was selected
    if kwargs["pipeline"] not in PIPELINES:
        raise InvalidParameterError(
            "Not a valid pipeline selection. "
            "Valid choices are:\n{}".format(
                ", ".join(PIPELINES.keys())
            )
        )

    stages = kwargs["stages"]
    if stages is None:
        raise InvalidParameterError(
            "No stages defined, need at least one."
        )

    # get definition of selected pipeline
    pipeline = PIPELINES[kwargs["pipeline"]]
    prefix = kwargs["global"]["prefix"]

    # make sure output directory exists
    # TODO: Exception handling here if this fails
    create_prefix_folders(prefix)

    # this is the global state of results as
    # we move through different stages of
    # the pipeline
    global_state = kwargs["global"]

    # keep track of how many stages are still
    # to be run, so we can leave out stages at
    # the end of workflow below
    num_stages_to_run = len(stages)

    # iterate through individual stages
    for (stage, runner) in pipeline:
        # check if anything else is left to
        # run, otherwise skip
        if num_stages_to_run == 0:
            break

        # define custom prefix for stage and create folder
        # stage_prefix = path.join(prefix, stage, "")
        stage_prefix = prefix
        create_prefix_folders(stage_prefix)

        # config files for input and output of stage
        stage_incfg = "{}_{}.incfg".format(stage_prefix, stage)
        stage_outcfg = "{}_{}.outcfg".format(stage_prefix, stage)

        # check if stage should be executed
        if stage in stages:
            # global state inserted at end, overrides any
            # stage-specific settings (except for custom prefix)
            incfg = {
                **kwargs["tools"],
                **kwargs["databases"],
                **kwargs[stage],
                **global_state,
                "prefix": stage_prefix
            }
            # save input of stage in config file
            write_config_file(stage_incfg, incfg)

            # run stage
            outcfg = runner(**incfg)

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
                if f.endswith("_file")
            ]

            verify_resources(
                "Output files from stage '{}' "
                "missing".format(stage),
                *outfiles
            )

        # update global state with outputs of stage
        global_state = {**global_state, **outcfg}

    # write final global state of pipeline
    write_config_file(
        prefix + "_final_global_state.outcfg", global_state
    )

    return global_state


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('config')
def run(**kwargs):
    """
    EVcouplings pipeline execution
    """
    config_file = kwargs["config"]
    verify_resources(
        "Config file does not exist or is empty.",
        config_file
    )

    # read configuration and execute
    config = read_config_file(config_file)
    outcfg = execute(**config)

    # print final configuration (end result)
    print(outcfg)


if __name__ == '__main__':
    run()
