"""
Pipelining of different stages of method

TODO:
- subfolders / prefix?
- proper argument parsing and overriding
  of options in config file
- callbacks after stages

Authors:
  Thomas A. Hopf
"""

from sys import argv, exit, stderr
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


def run(**kwargs):
    """
    Execute a configuration

    Parameters
    ----------
    # TODO

    Returns
    -------
    # TODO
    """
    check_required(
        kwargs,
        ["pipeline", "stages", "prefix", "global"]
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
    prefix = kwargs["prefix"]

    # make sure output directory exists
    # TODO: Exception handling here if this fails
    create_prefix_folders(prefix)

    # this is the global state of results as
    # we move through different stages of
    # the pipeline
    global_state = kwargs["global"]

    # iterate through individual stages
    for (stage, runner) in pipeline:
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


if __name__ == "__main__":
    # TODO: replace this with proper arg parsing
    if len(argv) == 2:
        config_file = argv[1]
        verify_resources(
            "Config file does not exist or is empty.",
            config_file
        )
        config = read_config_file(config_file)
        outcfg = run(**config)

        # TODO: save outcfg?
        print(outcfg)
    else:
        print(
            "Usage: {} <config file>".format(argv[0]),
            file=stderr
        )
        exit(1)
