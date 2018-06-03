"""
Protocols for matching putatively interacting sequences
in protein complexes to create a concatenated sequence
alignment

Authors:
  Anna G. Green
  Thomas A. Hopf
"""

from collections import Counter
import numpy as np
import pandas as pd

from evcouplings.couplings.mapping import Segment

from evcouplings.utils.config import (
    check_required, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources
)
from evcouplings.align.protocol import modify_alignment

from evcouplings.complex.alignment import (
    write_concatenated_alignment
)
from evcouplings.complex.distance import (
    find_possible_partners, best_reciprocal_matching,
    plot_distance_distribution
)
from evcouplings.complex.similarity import (
    read_species_annotation_table,
    most_similar_by_organism,
    filter_best_reciprocal,
    find_paralogs
)

def modify_complex_segments(outcfg, **kwargs):
    """
    Modifies the output configuration so
    that the segments are correct for a
    concatenated alignment

    Parameters
    ----------
    outcfg : dict
        The output configuration

    Returns
    -------
    outcfg: dict
        The output configuration, with
        a new field called "segments"

    """
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
    outcfg["segments"] = [s.to_list() for s in segments_complex]

    return outcfg

def _run_describe_concatenation(outcfg, **kwargs):
    """
    calculate some basic statistics on the concatenated alignment
    """
    prefix = kwargs["prefix"]
    outcfg["concatentation_statistics_file"] = prefix + "_concatenation_statistics.csv"
    describe_concatenation(
        kwargs["first_annotation_file"],
        kwargs["second_annotation_file"],
        kwargs["first_genome_location_file"],
        kwargs["second_genome_location_file"],
        outcfg["concatentation_statistics_file"]
    )
    return outcfg


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
        Path to genome location mapping file for first alignment
    genome_location_filename_2 : str
        Path to genome location mapping file for second alignment
    outfile: str
        Path to output file
    """

    # load the annotations for each alignment
    # as a pd.DataFrame
    annotations_1 = read_species_annotation_table(
        annotation_file_1
    )
    species_1 = annotations_1.species.values

    annotations_2 = read_species_annotation_table(
        annotation_file_2
    )
    species_2 = annotations_2.species.values
    
    # calculate the number of sequences found in each alignment
    num_seqs_1 = len(annotations_1)
    num_seqs_2 = len(annotations_2)
    
    # calculate the number of species found in each alignment
    # where a species is defined as a unique OS or Tax annotation field
    nonredundant_annotations_1 = len(set(species_1))
    nonredundant_annotations_2 = len(set(species_2))

    # calculate the number of overlapping species
    species_overlap = list(
        set(species_1).intersection(set(species_2))
    )
    n_species_overlap = len(species_overlap)
    
    # calculate the median number of paralogs per species
    n_paralogs_1 = float(
        # counts the number of times each species occurs in the list
        # then takes the median
        np.median(list(Counter(species_1).values()))
    )

    n_paralogs_2 = float(
        np.median(list(Counter(species_2).values()))
    )
    
    # If the user provided genome location files, calculate the number
    # of ids for which we found an embl CDS. Default value is np.nan
    embl_cds1 = np.nan
    embl_cds2 = np.nan

    if (genome_location_filename_1 is not None and
        genome_location_filename_2 is not None):

        genome_location_table_1 = pd.read_csv(genome_location_filename_1)
        genome_location_table_2 = pd.read_csv(genome_location_filename_2)

        # Number uniprot IDs with EMBL CDS that is not NA
        if "uniprot_ac" in genome_location_table_1.columns:
            embl_cds1 = len(list(set(genome_location_table_1.uniprot_ac)))
        if "uniprot_ac" in genome_location_table_2.columns:
            embl_cds2 = len(list(set(genome_location_table_2.uniprot_ac)))

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
            "first_region_start", "second_region_start",
            "first_segments", "second_segments",
            "genome_distance_threshold",
            "first_genome_location_file",
            "second_genome_location_file",
            "first_annotation_file",
            "second_annotation_file"
        ]
    )

    prefix = kwargs["prefix"]

    # make sure input alignments exist
    verify_resources(
        "Input alignment does not exist",
        kwargs["first_alignment_file"], kwargs["second_alignment_file"]
    )

    verify_resources(
        "Genome location file does not exist",
        kwargs["first_genome_location_file"],
        kwargs["second_genome_location_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # load the information for each monomer alignment
    alignment_1 = kwargs["first_alignment_file"]
    alignment_2 = kwargs["second_alignment_file"]

    genome_location_filename_1 = kwargs["first_genome_location_file"]
    genome_location_filename_2 = kwargs["second_genome_location_file"]

    gene_location_table_1 = pd.read_csv(genome_location_filename_1, header=0)
    gene_location_table_2 = pd.read_csv(genome_location_filename_2, header=0)

    # find all possible matches
    possible_partners = find_possible_partners(
        gene_location_table_1, gene_location_table_2
    )

    # find the best reciprocal matches
    id_pairing_unfiltered = best_reciprocal_matching(possible_partners)

    # filter best reciprocal matches by genome distance threshold
    if kwargs["genome_distance_threshold"]:
        distance_threshold = kwargs["genome_distance_threshold"]
        id_pairing = id_pairing_unfiltered.query("distance < @distance_threshold")
    else:
        id_pairing = id_pairing_unfiltered

    id_pairing.loc[:, "id_1"] = id_pairing.loc[:, "uniprot_id_1"]
    id_pairing.loc[:, "id_2"] = id_pairing.loc[:, "uniprot_id_2"]

    # write concatenated alignment with distance filtering
    # TODO: save monomer alignments?
    target_seq_id, target_seq_index, raw_ali, mon_ali_1, mon_ali_2 = \
        write_concatenated_alignment(
            id_pairing,
            alignment_1,
            alignment_2,
            kwargs["first_focus_sequence"],
            kwargs["second_focus_sequence"]
        )

    # save the alignment files
    raw_alignment_file = prefix + "_raw.fasta"
    with open(raw_alignment_file, "w") as of:
        raw_ali.write(of)

    mon_alignment_file_1 = prefix + "_monomer_1.fasta"
    with open(mon_alignment_file_1, "w") as of:
        mon_ali_1.write(of)   

    mon_alignment_file_2 = prefix + "_monomer_2.fasta"
    with open(mon_alignment_file_2, "w") as of:
        mon_ali_2.write(of)   

    # filter the alignment
    aln_outcfg, _ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs["first_region_start"],
        **kwargs
    )

    # make sure we return all the necessary information:
    # * alignment_file: final concatenated alignment that will go into plmc
    # * focus_sequence: this is the identifier of the concatenated target
    #   sequence which will be passed into plmc with -f
    outcfg = aln_outcfg
    outcfg["raw_alignment_file"] = raw_alignment_file
    outcfg["first_concatenated_monomer_alignment_file"] = mon_alignment_file_1
    outcfg["second_concatenated_monomer_alignment_file"] = mon_alignment_file_2
    outcfg["focus_sequence"] = target_seq_id

    # Update the segments
    outcfg = modify_complex_segments(outcfg, **kwargs)

    # Describe the statistics of the concatenation
    outcfg = _run_describe_concatenation(outcfg, **kwargs)

    # plot the genome distance distribution
    outcfg["distance_plot_file"] = prefix + "_distplot.pdf"
    plot_distance_distribution(id_pairing_unfiltered, outcfg["distance_plot_file"])

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
            "use_best_reciprocal", "paralog_identity_threshold"
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

    def _load_monomer_info(annotations_file, identities_file,
                           target_sequence, alignment_file,
                           use_best_reciprocal, identity_threshold):

        # read in annotation to a file and rename the appropriate column
        annotation_table = read_species_annotation_table(annotations_file)

        # read identity file
        similarities = pd.read_csv(identities_file)

        # create a pd.DataFrame containing the best hit in each organism
        most_similar_in_species = most_similar_by_organism(similarities, annotation_table)

        if use_best_reciprocal:
            paralogs = find_paralogs(
                target_sequence, annotation_table, similarities,
                identity_threshold
            )

            most_similar_in_species = filter_best_reciprocal(
                alignment_file, paralogs, most_similar_in_species
            )

        return most_similar_in_species

    # load the information about each monomer alignment
    most_similar_in_species_1 = _load_monomer_info(
        kwargs["first_annotation_file"],
        kwargs["first_identities_file"],
        kwargs["first_focus_sequence"],
        kwargs["first_alignment_file"],
        kwargs["use_best_reciprocal"],
        kwargs["paralog_identity_threshold"]
    )

    most_similar_in_species_2 = _load_monomer_info(
        kwargs["second_annotation_file"],
        kwargs["second_identities_file"],
        kwargs["second_focus_sequence"],
        kwargs["second_alignment_file"],
        kwargs["use_best_reciprocal"],
        kwargs["paralog_identity_threshold"]
    )

    # merge the two dataframes to get all species found in 
    # both alignments
    species_intersection = most_similar_in_species_1.merge(
        most_similar_in_species_2,
        how="inner",  # takes the intersection
        on="species",  # merges on species identifiers
        suffixes=("_1", "_2")
    )

    # write concatenated alignment with distance filtering
    # TODO: save monomer alignments?
    target_seq_id, target_seq_index, raw_ali, mon_ali_1, mon_ali_2 = \
        write_concatenated_alignment(
            species_intersection,
            kwargs["first_alignment_file"],
            kwargs["second_alignment_file"],
            kwargs["first_focus_sequence"],
            kwargs["second_focus_sequence"]
        )

    # save the alignment files
    raw_alignment_file = prefix + "_raw.fasta"
    with open(raw_alignment_file, "w") as of:
        raw_ali.write(of)

    mon_alignment_file_1 = prefix + "_monomer_1.fasta"
    with open(mon_alignment_file_1, "w") as of:
        mon_ali_1.write(of)

    mon_alignment_file_2 = prefix + "_monomer_2.fasta"
    with open(mon_alignment_file_2, "w") as of:
        mon_ali_2.write(of)

    aln_outcfg, _ = modify_alignment(
        raw_ali,
        target_seq_index,
        target_seq_id,
        kwargs["first_region_start"],
        **kwargs
    )

    # make sure we return all the necessary information:
    # * alignment_file: final concatenated alignment that will go into plmc
    # * focus_sequence: this is the identifier of the concatenated target
    #   sequence which will be passed into plmc with -f
    outcfg = aln_outcfg
    outcfg["raw_alignment_file"] = raw_alignment_file
    outcfg["first_concatenated_monomer_alignment_file"] = mon_alignment_file_1
    outcfg["second_concatenated_monomer_alignment_file"] = mon_alignment_file_2
    outcfg["focus_sequence"] = target_seq_id

    # Update the segments
    outcfg = modify_complex_segments(outcfg, **kwargs)

    # Describe the statistics of the concatenation
    outcfg = _run_describe_concatenation(outcfg, **kwargs)

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
