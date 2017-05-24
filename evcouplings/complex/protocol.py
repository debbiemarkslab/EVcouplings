"""
Protocols for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
  Thomas A. Hopf
  Anna G. Green
"""

from evcouplings.couplings.mapping import Segment
from evcouplings.align.alignment import read_fasta

from evcouplings.utils.config import (
    check_required, InvalidParameterError,
    read_config_file, write_config_file
)
from evcouplings.utils.system import (
    create_prefix_folders, valid_file,
    verify_resources,
)
from evcouplings.align.protocol import (
    modify_alignment
)
from evcouplings.align.alignment import (
    retrieve_sequence_ids,
    Alignment
)
from evcouplings.align.ena import (
    load_uniprot_to_embl,
    load_embl_to_annotation
)
from evcouplings.complex.alignment import (
    write_concatenated_alignment,
)
from evcouplings.complex.distance import (
    find_possible_partners,
    best_reciprocal_matching,
    filter_ids_by_distance
)
import re

def genome_distance(**kwargs):
    """
    Protocol:

    Concatenate alignments based on genomic distance

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
        (TODO: explain meaning of parameters in detail).

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        # TODO: this is the full list normally returned
        # by alignment protocol, decide which ones
        # to keep. Mandatory:
        # alignment_file, focus_sequence, focus_mode, segments

        alignment_file
        raw_alignment_file


        statistics_file
        target_sequence_file
        sequence_file
        [annotation_file]
        frequencies_file
        identities_file
        [hittable_file]
        focus_mode
        focus_sequence
        segments
    """
    check_required(
        kwargs,
        [
            "prefix",
            "first_alignment_file", "second_alignment_file",
            "first_focus_sequence", "second_focus_sequence",
            "first_focus_mode", "second_focus_mode",
            "first_segments", "second_segments",
            "genome_distance_threshold"
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

    def _load_monomer_information(alignment,uniprot_to_embl_filename,genome_location_filename):

        seq_ids_ali,id_to_header = retrieve_sequence_ids(open(alignment))
        
        uniprot_to_embl = load_uniprot_to_embl(uniprot_to_embl_filename)

        embl_to_annotation = load_embl_to_annotation(genome_location_filename)

        return seq_ids_ali,id_to_header,uniprot_to_embl,embl_to_annotation

    #load the information for each monomer alignment
    alignment_1 = kwargs["first_alignment_file"]
    alignment_2 = kwargs["second_alignment_file"]

    uniprot_to_embl_filename_1=kwargs['first_embl_mapping_file']
    uniprot_to_embl_filename_2=kwargs['second_embl_mapping_file']

    genome_location_filename_1=kwargs["first_genome_location_file"]
    genome_location_filename_2=kwargs["second_genome_location_file"]

    seq_ids_ali_1,id_to_header_1,uniprot_to_embl_1,embl_to_annotation_1 = \
        _load_monomer_information(
            alignment_1,
            uniprot_to_embl_filename_1,
            genome_location_filename_1
        )

    seq_ids_ali_2,id_to_header_2,uniprot_to_embl_2,embl_to_annotation_2 = \
        _load_monomer_information(
            alignment_2,
            uniprot_to_embl_filename_2,
            genome_location_filename_2
        )

    uniprot_to_embl = {**uniprot_to_embl_1,**uniprot_to_embl_2}
    embl_to_annotation = {**embl_to_annotation_1,**embl_to_annotation_2}

    #find all possible matches
    possible_partners = find_possible_partners(seq_ids_ali_1, 
                                            seq_ids_ali_2, 
                                            uniprot_to_embl, 
                                            embl_to_annotation) 

    #find the best reciprocal matches
    id_pairing_unfiltered,id_pair_to_distance = best_reciprocal_matching(possible_partners)
    
    #filter best reciprocal matches by genome distance threshold
    id_pairing = filter_ids_by_distance(
        id_pairing_unfiltered,
        id_pair_to_distance,
        kwargs['genome_distance_threshold']
    )

    raw_alignment_file = prefix + '_raw.fasta'

    #write concatenated alignment with distance filtering
    target_seq_id, target_seq_index = write_concatenated_alignment(
        id_pairing,
        id_to_header_1,
        id_to_header_2,
        alignment_1,
        alignment_2,
        kwargs['first_focus_sequence'],
        kwargs['second_focus_sequence'],
        raw_alignment_file
    )

    #filter the alignment
    raw_ali = Alignment.from_file(open(raw_alignment_file))
    aln_outcfg,_ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs['first_region_start'],
        **kwargs
    )

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
    outcfg=aln_outcfg
    outcfg["segments"] = [s.to_list() for s in segments_complex]
    outcfg["focus_sequence"] = target_seq_id
    return outcfg

def best_hit(**kwargs):
    """
    Protocol:

    Concatenate alignments based on the best hit 
    to the focus sequence in each species

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required
        (TODO: explain meaning of parameters in detail).

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        # TODO: this is the full list normally returned
        # by alignment protocol, decide which ones
        # to keep. Mandatory:
        # alignment_file, focus_sequence, focus_mode, segments

        alignment_file
        raw_alignment_file
        statistics_file
        target_sequence_file
        sequence_file
        [annotation_file]
        frequencies_file
        identities_file
        [hittable_file]
        focus_mode
        focus_sequence
        segments
    """
    check_required(
        kwargs,
        [
            "prefix",
            "first_alignment_file", "second_alignment_file",
            "first_focus_sequence", "second_focus_sequence",
            "first_focus_mode", "second_focus_mode",
            "first_segments", "second_segments",
            "first_identities_file","second_identities_file"
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


    #load the information for each monomer alignment

    #find all possible matches

    #find the best matches

    #write concatenated alignment with distance filtering
    raw_alignment_file = prefix + '_raw.fasta'
    target_seq_id, target_seq_index = write_concatenated_alignment(id_pairing,
                             id_to_header_1,
                             id_to_header_2,
                             alignment_1,
                             alignment_2,
                             kwargs['first_focus_sequence'],
                             kwargs['second_focus_sequence'],
                             raw_alignment_file
                             )

    #filter the alignment
    raw_ali = Alignment.from_file(open(raw_alignment_file))
    aln_outcfg,_ = modify_alignment(raw_ali,
                                target_seq_index,
                                target_seq_id,
                                kwargs['first_region_start'],
                                **kwargs

        )


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
    outcfg=aln_outcfg
    outcfg["segments"] = [s.to_list() for s in segments_complex]
    outcfg["focus_sequence"] = target_seq_id
    return outcfg


# list of available EC inference protocols
PROTOCOLS = {
    # concatenate based on genomic distance ("operon-based")
    "genome_distance": genome_distance,

    # concatenate based on best hit per genome ("species")
    "best_hit": best_hit

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
        Dictionary with results in following fields:
        (in brackets: not mandatory)

        # TODO: to be finalized after implementing protocols
        alignment_file
        focus_mode
        focus_sequence
        segments
        num_sites
        num_sequences
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
