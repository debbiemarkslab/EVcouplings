"""
Alignment creation protocols/workflows.

# TODO: think about how to easily plug in alternative workflows
        without having to change module

Authors:
  Thomas A. Hopf
"""


def fetch_sequence(config):
    """
    Gets sequence data based on defined logic
    """
    return


def search_jackhmmer(config):
    """
    Get alignment by homology search from query
    """
    return


def modify_alignment(config):
    """
    Prepare alignment to be ready for EC calculation
    """
    return


def standard(config, realign=False):
    """
    Standard buildali workflow

    config: list all items read by this protocol

    TODO: think about how to handle config parameters best.
    TODO: The workflow has to take care of logging
    """
    # get the sequence (or move this outside?)

    # run jackhmmer

    # parse to a2m

    # apply id filter, gap threshold

    # set correct headers (make ready for plmc)

    # generate specieslist

    # visualize distributions?

    if realign:
        realign(config)

    # TODO: how to get alignment statistics and plots?
    # (modularize this into an independent function too)

    # in the end, return both alignment object (if in memory)
    # and path to final alignment file
    return


def realign(config):
    """
    Realign sequences in multiple sequence alignment.
    Fetch fulll sequences, etc, etc.
    """
    raise NotImplementedError("Realignment not yet implemented")
