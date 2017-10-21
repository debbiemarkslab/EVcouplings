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
from copy import copy
import pandas as pd
from collections import defaultdict
from evcouplings.align.ids import retrieve_sequence_ids

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
    list of (str, str)
        A list of Uniprot ac, CDS id pairs
        the CDS id(s) corresponding to each
        Uniprot AC. Uniprot ACs may be repeated
        if there were multiple CDS hits.
    """

    def _split_annotation_string(annotation_string):
        """
        reformats the ENA annotation string as a list of
        [(read,cds)] tuples

        """
        full_annotation = [
            tuple(x.split(":")) for x in
            annotation_string.split(",")
        ]  # list of lists in format [read,cds]

        return full_annotation

    def _remove_redundant_cds(uniprot_and_genome_cds):
        """
        Removes CDSs that have hits to multiple genomes

        Returns a list of tuples (Uniprot_AC, CDS)

        """

        filtered_uniprot_and_cds = []
        for uniprot_ac, genome_and_cds in uniprot_and_genome_cds:

            count_reads = defaultdict(list)

            for genome, cds in genome_and_cds:
                count_reads[cds].append(genome)

            # check how many genomes are associated with a particular CDS,
            # only keep CDSs that can be matched to one genome
            for cds, genomes in count_reads.items():
                if len(genomes) == 1:
                    filtered_uniprot_and_cds.append((uniprot_ac, cds))

        return filtered_uniprot_and_cds

    # extract identifiers from sequence alignment
    with open(alignment_file) as f:
        sequence_id_list, _ = retrieve_sequence_ids(f)

    # store IDs in set for faster membership checking
    target_ids = set(sequence_id_list)

    # initialize list of list of (uniprot,[(genome,CDS)]) tuples
    # eg [(uniprot1,[(genome1,cds1),(genome2,cds2)])]
    # TODO: I know this is ugly but I need to save the uniprot id
    # for later use, and I need to save the genome so that I can remove
    # cds's that are hit to multiple genomes
    genome_and_cds = []

    # read through the data table line-by-line to improve speed
    with open(uniprot_to_embl_table) as f:
        for line in f:
            # ena_data is formatted as 'genome1:cds1,genome2:cds2'
            uniprot_ac, _, ena_data = line.rstrip().split(" ")

            if uniprot_ac in target_ids:
                genome_and_cds.append((
                    uniprot_ac, _split_annotation_string(ena_data)
                ))

    # clean the uniprot to embl hits
    # returns a list of uniprot_ac, cds_id tuples
    filtered_uniprot_and_cds = _remove_redundant_cds(genome_and_cds)

    return filtered_uniprot_and_cds


def extract_embl_annotation(uniprot_and_cds,
                            ena_genome_location_table,
                            genome_location_filename):
    """
    Reads genomic location information
    for all input EMBL coding DNA sequences (CDS) corresponding
    to the amino cid sequences in the alignment; creates
    a pd.DataFrame with the following columns:

    cds_id, genome_id, uniprot_ac, gene_start, gene_end

    Each row is a unique CDS. Uniprot ACs may be repeated if one
    Uniprot AC hits multiple CDS.
    
    Parameters
    ----------
    uniprot_and_cds : list of (str, str)
        A list of uniprot ac, CDS id pairs for which to extract
        genome location
    ena_genome_location_table : str
        Path to ENA genome location database table, which is a 
        a tsv file with the following columns:
        cds_id, genome_id, uniprot_ac, genome_start, genome_end
    genome_location_filename : str
        File to write containing CDS location info for
        target sequences

    Returns
    -------
    pd.DataFrame
        Columns: cds, genome_id, uniprot_ac, gene_start, gene_end
        Each row is a unique CDS. Uniprot ACs may be repeated if one
        Uniprot AC hits multiple CDS.
    """

    # initialize list of list
    # will contain [[CDS,genome_id,uniprot_ac,gene_start,gene_end]]
    embl_cds_annotation = [] 

    # convert cds_target_ids to set to speed up membership checking
    cds_target_ids = [x for _, x in uniprot_and_cds]
    cds_target_set = set(cds_target_ids)

    # dictionary of uniprot acs to save for each cds
    cds_to_uniprot = {y: x for x, y in uniprot_and_cds}

    # extract the annotation
    # note: we don't use the Uniprot AC from this file
    # as the mapping is sometimes ambiguous
    # ie, a CDS can map to multiple Uniprot ACs
    with open(ena_genome_location_table) as inf:
        for line in inf:

            cds_id, genome_id, _, start, end = (
                line.rstrip().split("\t")
            )

            # if this row of the table contains a CDS id in our set
            # save the information
            if cds_id in cds_target_set:
                embl_cds_annotation.append([
                    cds_id, genome_id, cds_to_uniprot[cds_id], start, end
                ])

    genome_location_table = pd.DataFrame(embl_cds_annotation, columns=[
        "cds", "genome_id", "uniprot_ac", "gene_start", "gene_end"
    ])

    # write the annotation
    return genome_location_table

def add_full_header(table, alignment_file):
    """
    Add column called full_id
    with the header in the sequence alignment
    that corresponds to the uniprot_AC in the
    genome location table

    Parameters
    ----------
    table : pd.DataFrame
        Columns: cds, genome_id, uniprot_ac, gene_start, gene_end
        Each row is a unique CDS. Uniprot ACs may be repeated if one
        Uniprot AC hits multiple CDS.
    alignment_file : str
        Path to sequence alignment

    Returns
    -------
    pd.DataFrame
        Same as above but with a "full_id"
        column
    """
    
    with open(alignment_file) as inf:
        _, id_to_header = retrieve_sequence_ids(inf)

    new_df = pd.DataFrame()

    for _, row in table.iterrows():
        row_copy = copy(row).to_frame().transpose() 

        # for each full_id that corresponds to that uniprot AC
        for full_id in id_to_header[row_copy.uniprot_ac.values[0]]:
            # create a new row and save that full_id 
            row_copy = row_copy.assign(full_id = full_id)
            new_df = pd.concat([new_df,row_copy])
    return new_df
