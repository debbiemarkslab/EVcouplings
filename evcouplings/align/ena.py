"""
Protocols for mapping Uniprot sequences to EMBL/ENA
database for extracting genomic location, so that genomic
distance between putatively interacting pairs can be
calculated.

Authors:
  Anna G. Green
  Thomas A. Hopf
  Charlotta P.I. Schärfe
"""
import os
from operator import itemgetter
from collections import defaultdict
from evcouplings.align.ids import retrieve_sequence_ids
import pandas as pd


def extract_cds_ids(alignment_file,
                    uniprot_to_embl_table):
    """
    Extracts mapping from set of Uniprot IDs to EMBL 
    Coding DNA sequence (CDS) from precomputed ID mapping table. 
    Will only include CDSs that can be mapped unambiguously
    to one EMBL genome.
    
    Parameters
    ----------
    alignment_file : str
        Path to alignment with sequences for which IDs
         should be retrieved
    uniprot_to_embl_table : str
        Path to uniprot to embl mapping database

    Returns
    -------
    list of str
        A list CDS ids that correspond to 
        the uniprot IDs in the alignment file. 
        Note these are not ordered as the alignment IDs.

    """

    def _split_annotation_string(annotation_string):
        # reformat the ena string as a list of tuples

        full_annotation = [
            tuple(x.split(":")) for x in
            annotation_string.split(",")
        ]  # list of lists in format [read,cds]

        return full_annotation

    def _remove_redundant_genomes(genome_and_cds):

        """
        Removes CDSs that have hits to multiple genomes

        """
        filtered_genome_and_cds = []
        for full_annotation in genome_and_cds:

            count_reads = defaultdict(list)

            for genome, cds in full_annotation:
                count_reads[cds].append(genome)

            # check how many genomes are associated with a particular CDS,
            # only keep CDSs that can be matched to *one* genome
            for cds, genomes in count_reads.items():
                if len(genomes) == 1:
                    filtered_genome_and_cds.append((genomes[0], cds))

        return filtered_genome_and_cds

    # extract identifiers from sequence alignment
    with open(alignment_file) as f:
        sequence_id_list, _ = retrieve_sequence_ids(f)

    # store IDs in set for faster membership checking
    target_ids = set(sequence_id_list)

    # initialize list of list of (genome,CDS) tuples
    # example: [[(genome1,cds1),(genome2,cds2)],[(genome1,cds2)]]
    genome_and_cds = []

    # read through the data table line-by-line to improve speed
    with open(uniprot_to_embl_table) as f:
        for line in f:
            #ENA data is formatted as 'genome1:cds1,genome2:cds2'
            uniprot_ac, _, ena_data = line.rstrip().split(" ")

            if uniprot_ac in target_ids:
                genome_and_cds.append(_split_annotation_string(ena_data))

    # clean the uniprot to embl hits
    filtered_genome_and_cds = _remove_redundant_genomes(genome_and_cds)

    # extract just the CDS
    cds = [x for _,x in filtered_genome_and_cds]

    # store mapping information for alignment to file
    return cds

def extract_embl_annotation(cds_target_ids,
                            ena_genome_location_table,
                            genome_location_filename):

    """
    Reads genomic location information
    for all input EMBL coding DNA sequences (CDS); writes that
    information to a csv file with the following columns:

    cds_id, genome_id, uniprot_ac, gene_start, gene_end

    Each row is a unique CDS. Uniprot ACs may be repeated if one
    Uniprot AC hits multiple CDS.
    
    Parameters
    ----------
    cds_target_ids : list of str
        A list of CDS ids for which to extract genome location
    ena_genome_location_table : str
        Path to ENA genome location database table, which is a 
        a tsv file with the following columns:
        cds_id, genome_id, uniprot_ac, genome_start, genome_end
    genome_location_filename : str
        File to write containing CDS location info for
        target sequences

    """

    # initialize list of list
    # will contain [[CDS,genome_id,uniprot_ac,gene_start,gene_end]]
    embl_cds_annotation = [] 

    # extract the annotation
    with open(ena_genome_location_table) as inf:
        for line in inf:
            cds_id, genome_id, uniprot_id, start, end = (
                line.rstrip().split("\t")
            )

            if cds_id in cds_target_ids:
                embl_cds_annotation.append([
                    cds_id, genome_id, uniprot_id, start, end
                ])

    genome_location_table = pd.DataFrame(embl_cds_annotation,columns=[
        "cds","genome_id","uniprot_ac","gene_start","gene_end"
    ])

    # write the annotation
    genome_location_table.to_csv(genome_location_filename)
