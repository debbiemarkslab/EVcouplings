"""
Protocols for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
  Thomas A. Hopf
  Anna G. Green
"""

import numpy as np
import pandas as pd
from collections import Counter

from evcouplings.couplings.mapping import Segment

from evcouplings.utils.config import (
    check_required, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources
)
from evcouplings.align.protocol import modify_alignment
from evcouplings.align.alignment import Alignment
from evcouplings.align.ids import retrieve_sequence_ids

from evcouplings.complex.alignment import (
    write_concatenated_alignment
)
from evcouplings.complex.distance import (
    find_possible_partners, best_reciprocal_matching,
    plot_distance_distribution
)
from evcouplings.complex.similarity import (
    read_identity_file,
    read_annotation_file,
    most_similar_by_organism,
    filter_best_reciprocal
)


def describe_concatenation(annotation_file_1, annotation_file_2,
                           genome_location_filename_1, genome_location_filename_2,
                           outfile):
    """
    Describes properties of concatenated alignment. 

    Writes a csv with the following columns

    num_seqs_1 : number of sequences in the first monomer alignment
    num_seqs_2 : number of sequences in the second monomer alignment
    num_nonred_species_1 : number of unique species annotations in the 
        first monomer alignment
    num_nonred_species_2 : number of unique species annotations in the 
        second monomer alignment
    num_species_overlap: number of unique species found in both alignments
    median_num_per_species_1 : median number of paralogs per species in the 
        first monomer alignmment
    median_num_per_species_2 : median number of paralogs per species in 
        the second monomer alignment
    num_with_embl_cds_1 : number of IDs for which we found an EMBL CDS in the 
        first monomer alignment (relevant to distance concatention only)
    num_with_embl_cds_2 : number of IDs for which we found an EMBL CDS in the 
        first monomer alignment (relevant to distance concatention only)
    
    Parameters
    ----------
    annotation_file_1 : str
        Path to annotation.csv file for first monomer alignment
    annotation_file_2 : str
        Path to annotation.csv file for second monomer alignment
    genome_location_filename_1 : str
        Path to Uniprot to CDS genome location mapping file for
        first alignment
    genome_location_filename_2 : str
        Path to Uniprot to CDS genome location mapping file for
        second alignment
    outfile: str
        Path to output file
    """

    # load the annotations for each alignment
    annotations_1 = read_annotation_file(
        annotation_file_1
    )

    annotations_2 = read_annotation_file(
        annotation_file_2
    )
    
    #calculate the number of sequences found in each alignment
    num_seqs_1 = len(annotations_1)
    num_seqs_2 = len(annotations_2)
    
    #calculate the number of species found in each alignment
    #where a species is defined as a unique OS or Tax annotation field
    nonredundant_annotations_1 = len(
        list(set(annotations_1.values()))
    )

    nonredundant_annotations_2 = len(
        list(set(annotations_2.values()))
    )

    #calculate the number of overlapping species
    species_overlap = list(
        set(annotations_1.values()).intersection(
            set(annotations_2.values())
        )
    )
    n_species_overlap = len(species_overlap)
    
    #calculate the median number of paralogs per species
    n_paralogs_1 = float(
        np.median(list(Counter(annotations_1.values()).values()))
    )

    n_paralogs_2 = float(
        np.median(list(Counter(annotations_2.values()).values()))
    )
    
    # If the user provided genome location files, calculate the number
    # of ids for which we found an embl CDS
    if genome_location_filename_1 is not None and \
        genome_location_filename_2 is not None:

        genome_location_table_1 = pd.read_csv(genome_location_filename_1)
        genome_location_table_2 = pd.read_csv(genome_location_filename_2)

        #Number uniprot IDs with EMBL CDS that is not NA
        genome_location_table_1 = genome_location_table_1.dropna(inplace=True)
        genome_location_table_2 = genome_location_table_2.dropna(inplace=True)
        embl_cds1 = len(list(set(genome_location_filename_1.uniprot_ac)))
        embl_cds2 = len(list(set(genome_location_filename_2.uniprot_ac)))

    else:
        embl_cds1 = None
        embl_cds2 = None

    concatenation_data = [
        num_seqs_1,
        num_seqs_2,
        nonredundant_annotations_1,
        nonredundant_annotations_2,
        n_species_overlap,
        n_paralogs_1,
        n_paralogs_2,
        embl_cds1,
        embl_cds2,
    ]
    
    cols = [
        "num_seqs_1",
        "num_seqs_2",
        "num_nonred_species_1",
        "num_nonred_species_2",
        "num_species_overlap",
        "median_num_per_species_1",
        "median_num_per_species_2",
        "num_with_embl_cds_1",
        "num_with_embl_cds_2",
    ]

    # create dataframe and store
    data_df = pd.DataFrame(
        [concatenation_data], columns=cols
    )

    data_df.to_csv(outfile)


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

        * alignment_file
        * raw_alignment_file
        * focus_mode
        * focus_sequence
        * segments
        * frequencies_file
        * identities_file
        * num_sequences
        * num_sites
        * raw_focus_alignment_file
        * statistics_file
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

    # make sure input alignments exist
    verify_resources(
        "Input alignment does not exist",
        kwargs["first_alignment_file"], kwargs["second_alignment_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # load the information for each monomer alignment
    alignment_1 = kwargs["first_alignment_file"]
    alignment_2 = kwargs["second_alignment_file"]

    genome_location_filename_1 = kwargs["first_genome_location_file"]
    genome_location_filename_2 = kwargs["second_genome_location_file"]

    def _load_monomer_information(alignment, genome_location_filename):
        # load the ids and full headers from the sequence alignment
        seq_ids_ali, id_to_header = retrieve_sequence_ids(open(alignment))

        # load the previously computed table of CDS locations
        gene_location_table = pd.read_csv(genome_location_filename,header=0)

        return seq_ids_ali, id_to_header, gene_location_table

    seq_ids_ali_1, id_to_header_1, gene_location_table_1 = \
        _load_monomer_information(
            alignment_1,
            genome_location_filename_1
        )

    seq_ids_ali_2, id_to_header_2, gene_location_table_2 = \
        _load_monomer_information(
            alignment_2,
            genome_location_filename_2
        )
    print(gene_location_table_1.head())
    id_to_header_1[kwargs["first_focus_sequence"]] = [kwargs["first_focus_sequence"]]
    id_to_header_2[kwargs["second_focus_sequence"]] = [kwargs["second_focus_sequence"]]

    # find all possible matches
    possible_partners = find_possible_partners(
        seq_ids_ali_1,
        seq_ids_ali_2,
        gene_location_table_1,
        gene_location_table_2
    )

    # find the best reciprocal matches
    id_pairing_unfiltered = best_reciprocal_matching(possible_partners)

    # filter best reciprocal matches by genome distance threshold
    if kwargs["genome_distance_threshold"]:
        distance_threshold = kwargs['genome_distance_threshold']
        id_pairing = id_pairing_unfiltered.query("distance < @distance_threshold")
    else:
        id_pairing = id_pairing_unfiltered

    raw_alignment_file = prefix + "_raw.fasta"

    # write concatenated alignment with distance filtering
    id_pair_tuples = zip(id_pairing["uniprot_id_1"],id_pairing["uniprot_id_2"])
    target_seq_id, target_seq_index = write_concatenated_alignment(
        id_pairing,
        id_to_header_1,
        id_to_header_2,
        alignment_1,
        alignment_2,
        kwargs["first_focus_sequence"],
        kwargs["second_focus_sequence"],
        raw_alignment_file
    )

    # filter the alignment
    raw_ali = Alignment.from_file(open(raw_alignment_file))
    aln_outcfg, _ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs["first_region_start"],
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
    outcfg = aln_outcfg

    outcfg["segments"] = [s.to_list() for s in segments_complex]
    outcfg["focus_sequence"] = target_seq_id

    # plot the genome distance distribution
    outcfg["distance_plot_file"]=prefix + "_distplot.pdf"
    plot_distance_distribution(id_pairing_unfiltered, outcfg["distance_plot_file"])

    # calculate some basic statistics on the concatenated alignment
    outcfg["concatentation_statistics_file"]=prefix+"_concatenation_statistics.csv"
    describe_concatenation(kwargs["first_annotation_file"],kwargs["second_annotation_file"],
                      kwargs["first_genome_location_file"],kwargs["second_genome_location_file"],
                      outcfg["concatentation_statistics_file"])

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

        alignment_file
        raw_alignment_file
        focus_mode
        focus_sequence
        segments
        frequencies_file
        identities_file
        num_sequences
        num_sites
        raw_focus_alignment_file
        statistics_file
    """
    check_required(
        kwargs,
        [
            "prefix",
            "first_alignment_file", "second_alignment_file",
            "first_focus_sequence", "second_focus_sequence",
            "first_focus_mode", "second_focus_mode",
            "first_segments", "second_segments",
            "first_identities_file", "second_identities_file",
            "first_annotation_file", "second_annotation_file"
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

    def _load_monomer_info(annotations_file,
                           identities_file,
                           target_sequence,
                           alignment_file):

        id_to_organism = read_annotation_file(annotations_file)
        # This dictionary maps identifiers used to calculate pairing
        # to identifiers that will be used
        id_to_header = {x: [x] for x in id_to_organism.keys()}

        # TODO: fix this so that we don"t assume target sequence is the first sequence
        id_to_header[target_sequence] = [Alignment.from_file(open(alignment_file)).ids[0]]

        similarities = read_identity_file(identities_file)
        species_to_most_similar = most_similar_by_organism(similarities, id_to_organism)

        return species_to_most_similar, id_to_header

    # load the information about each monomer alignment
    species_to_most_similar_1, id_to_header_1 = _load_monomer_info(
        kwargs["first_annotation_file"],
        kwargs["first_identities_file"],
        kwargs["first_focus_sequence"],
        kwargs["first_alignment_file"]
    )

    species_to_most_similar_2, id_to_header_2 = _load_monomer_info(
        kwargs["second_annotation_file"],
        kwargs["second_identities_file"],
        kwargs["second_focus_sequence"],
        kwargs["second_alignment_file"]
    )


    # determine the species intersection
    species_intersection = [x for x in species_to_most_similar_1.keys() if x in species_to_most_similar_2.keys()]
    
    # pair the sequence identifiers
    sequence_pairing = [(species_to_most_similar_1[x][1], species_to_most_similar_2[x][1]) for x in
                        species_intersection]

    raw_alignment_file = prefix + "_raw.fasta"

    target_seq_id, target_seq_index = write_concatenated_alignment(
        sequence_pairing,
        id_to_header_1, id_to_header_2,
        kwargs["first_alignment_file"],
        kwargs["second_alignment_file"],
        kwargs["first_focus_sequence"],
        kwargs["second_focus_sequence"],
        raw_alignment_file
    )

    # filter the alignment
    raw_ali = Alignment.from_file(open(raw_alignment_file))
    aln_outcfg, _ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs["first_region_start"],
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
    outcfg = aln_outcfg
    outcfg["segments"] = [s.to_list() for s in segments_complex]
    outcfg["focus_sequence"] = target_seq_id

    outcfg["concatentation_statistics_file"]=prefix+"_concatenation_statistics.csv"
    describe_concatenation(kwargs["first_annotation_file"],kwargs["second_annotation_file"],
                      kwargs["first_genome_location_file"],kwargs["second_genome_location_file"],
                      outcfg["concatentation_statistics_file"])

    return outcfg


def best_reciprocal_hit(**kwargs):
    """
    Protocol:

    Concatenate alignments based on the best reciprocal
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

        alignment_file
        raw_alignment_file
        focus_mode
        focus_sequence
        segments
        frequencies_file
        identities_file
        num_sequences
        num_sites
        raw_focus_alignment_file
        statistics_file

    """
    check_required(
        kwargs,
        [
            "prefix",
            "first_alignment_file", "second_alignment_file",
            "first_focus_sequence", "second_focus_sequence",
            "first_focus_mode", "second_focus_mode",
            "first_segments", "second_segments",
            "first_identities_file", "second_identities_file",
            "first_annotation_file", "second_annotation_file",
            "paralog_identity_threshold"
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


    def _find_paralogs(query_id, annotation_file, identity_file,
                       identity_threshold):
        '''
        Parameters
        ----------
        identity_threshold: float
            sequences above this identity to the query are not considered paralogs

        Returns
        -------
        filtered_paralogs: list of str
            full sequence identities of paralogs found in the same genome as
            the query id
        '''


        annotations = pd.read_csv(annotation_file,na_values=None)
        base_query = query_id.split("/")[0]

        identities = read_identity_file(identity_file)

        # if it's uniprot, extract based on having the same species annotation
        if annotations.RepID.isnull().all():
            self_hit_row = annotations[[True if base_query in x else False
                                        for x in list(annotations.id)]
            ]
            self_hit_annotation = self_hit_row.iloc[-1, :]["OS"]
            paralogs = list(annotations[annotations.OS == self_hit_annotation].id)

        # if it's uniref, extract based on having the same RepID
        else:
            self_hit_annotation = base_query.split("_")[1]
            paralogs = [annotations.iloc[0, :].id]
            annots = [""]+list(annotations.RepID)[1::]
            paralog_rows = annotations[[True if self_hit_annotation in x else False
                                        for x in annots]
            ]
            paralogs += list(paralog_rows.id)

        # confirm that paralogs are below the similarity threshold
        # ie, are diverged in sequence space from the query
        filtered_paralogs = [paralogs[0]]
        for paralog in paralogs[1::]:  # first entry is query
            if not paralog in identities:
                continue
            if identities[paralog] < identity_threshold:
                filtered_paralogs.append(paralog)
        print(len(filtered_paralogs))
        return filtered_paralogs

    def _load_monomer_info(annotations_file,
                           identities_file,
                           target_sequence,
                           alignment_file,
                           identity_threshold):
        id_to_organism = read_annotation_file(annotations_file)
        id_to_header = {x: [x] for x in id_to_organism.keys()}

        # TODO: fix this so that we don"t assume target sequence is the first sequence
        id_to_header[target_sequence] = [Alignment.from_file(open(alignment_file)).ids[0]]

        similarities = read_identity_file(identities_file)
        species_to_most_similar = most_similar_by_organism(similarities, id_to_organism)

        paralogs = _find_paralogs(target_sequence, annotations_file, identities_file,
                                  identity_threshold)

        species_to_bestreciprocal,_ = filter_best_reciprocal(alignment_file,paralogs,species_to_most_similar)

        return species_to_bestreciprocal, id_to_header

    # load the information about each monomer alignment
    species_to_most_similar_1, id_to_header_1 = _load_monomer_info(
        kwargs["first_annotation_file"],
        kwargs["first_identities_file"],
        kwargs["first_focus_sequence"],
        kwargs["first_alignment_file"],
        kwargs["paralog_identity_threshold"]
    )

    species_to_most_similar_2, id_to_header_2 = _load_monomer_info(
        kwargs["second_annotation_file"],
        kwargs["second_identities_file"],
        kwargs["second_focus_sequence"],
        kwargs["second_alignment_file"],
        kwargs["paralog_identity_threshold"]
    )

    # determine the species intersection
    species_intersection = [x for x in species_to_most_similar_1.keys() if x in species_to_most_similar_2.keys()]

    # pair the sequence identifiers
    sequence_pairing = [(species_to_most_similar_1[x][1], species_to_most_similar_2[x][1]) for x in
                        species_intersection]

    raw_alignment_file = prefix + "_raw.fasta"

    target_seq_id, target_seq_index = write_concatenated_alignment(
        sequence_pairing,
        id_to_header_1, id_to_header_2,
        kwargs["first_alignment_file"],
        kwargs["second_alignment_file"],
        kwargs["first_focus_sequence"],
        kwargs["second_focus_sequence"],
        raw_alignment_file
    )

    # filter the alignment
    raw_ali = Alignment.from_file(open(raw_alignment_file))
    aln_outcfg, _ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs["first_region_start"],
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
    outcfg = aln_outcfg
    outcfg["segments"] = [s.to_list() for s in segments_complex]
    outcfg["focus_sequence"] = target_seq_id

    outcfg["concatentation_statistics_file"] = prefix + "_concatenation_statistics.csv"

    describe_concatenation(
        kwargs["first_annotation_file"], kwargs["second_annotation_file"],
        kwargs["first_genome_location_file"], kwargs["second_genome_location_file"],
        outcfg["concatentation_statistics_file"]
    )

    return outcfg

# list of available EC inference protocols
PROTOCOLS = {
    # concatenate based on genomic distance ("operon-based")
    "genome_distance": genome_distance,

    # concatenate based on best hit per genome ("species")
    "best_hit": best_hit,

    # concatenate based on best hit per genome ("species")
    "best_reciprocal": best_reciprocal_hit

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
        raw_alignment_file
        focus_mode
        focus_sequence
        segments
        frequencies_file
        identities_file
        num_sequences
        num_sites
        raw_focus_alignment_file
        statistics_file

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
