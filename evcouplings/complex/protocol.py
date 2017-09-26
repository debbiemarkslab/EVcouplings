"""
Protocols for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
  Thomas A. Hopf
  Anna G. Green
"""

from evcouplings.couplings.mapping import Segment
from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    read_config_file, write_config_file
)
from evcouplings.utils.system import (
    create_prefix_folders, valid_file,
    verify_resources,
)


def genome_distance(**kwargs):
    """
    Protocol:

    Concatenate alignments based on genomic distance

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

        .. todo::

            Explain meaning of parameters in detail.

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        .. todo::

            this is the full list normally returned by alignment protocol, decide which ones to keep.
            Mandatory:

            * alignment_file
            * focus_sequence
            * focus_mode
            * segments

        * alignment_file
        * [raw_alignment_file]
        * statistics_file
        * target_sequence_file
        * sequence_file
        * [annotation_file]
        * frequencies_file
        * identities_file
        * [hittable_file]
        * focus_mode
        * focus_sequence
        * segments
    """
    check_required(
        kwargs,
        [
            "prefix",
            "first_raw_focus_alignment_file", "second_raw_focus_alignment_file",
            "first_focus_sequence", "second_focus_sequence",
            "first_focus_mode", "second_focus_mode",
            "first_segments", "second_segments",
        ]
    )

    prefix = kwargs["prefix"]

    # make sure input alignments
    verify_resources(
        "Input alignment does not exist",
        kwargs["first_alignment_file"], kwargs["second_alignment_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # -------------------------------------------------
    # TODO: implement concatenation functionality and
    # postprocessing functionality here
    # -------------------------------------------------

    def _modify_segments(seg_list, seg_prefix):
        # extract segments from list representation into objects
        segs = [
            Segment.from_list(s) for s in seg_list
        ]
        # update segment IDs
        for i, s in enumerate(segs, start=1):
            s.segment_id = "{}_{}".format(seg_prefix, i)

        return segs

    # merge segments - this allows to have more than one segment per
    # "monomer" alignment
    segments_1 = _modify_segments(kwargs["first_segments"], "A")
    segments_2 = _modify_segments(kwargs["second_segments"], "B")
    segments_complex = segments_1 + segments_2

    # make sure we return all the necessary information:
    # * alignment_file: final concatenated alignment that will go into plmc
    # * focus_sequence: this is the identifier of the concatenated target
    #   sequence which will be passed into plmc with -f

    outcfg = {
        "alignment_file": None,  # TODO: specify
        "focus_mode": True,
        "focus_sequence": None,  # TODO: specify
        "segments": [s.to_list() for s in segments_complex],
        # optional but good to have:
        "num_sites": None,
        "num_sequences": None,
        # "effective_sequences": n_eff # TODO: could compute this like in align stage
        # TODO: there are more outputs that we could add here (not mandatory),
        # e.g. single column frequencies in concatenated alignment
    }

    return outcfg


# list of available EC inference protocols
PROTOCOLS = {
    # concatenate based on genomic distance ("operon-based")
    "genome_distance": genome_distance,
}


def run(**kwargs):
    """
    Run alignment concatenation protocol

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: concatenation protocol to run
        prefix: Output prefix for all generated files

    Returns
    -------
    outcfg : dict
        Output configuration of concatenation stage
        Dictionary with results in following fields (in brackets: not mandatory):

        .. todo::

            to be finalized after implementing protocols

        * alignment_file
        * focus_mode
        * focus_sequence
        * segments
        * num_sites
        * num_sequences
    """
    check_required(kwargs, ["protocol"])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)
