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


def extract_uniprot_to_embl(alignment_file,
                            uniprot_to_embl_table,
                            uniprot_to_embl_filename):
    """
    Extracts mapping from set of Uniprot IDs to EMBL IDs
    from precomputed ID mapping table. The file generated
    by this function is subsequently loaded by
    load_uniprot_to_embl().
    
    Parameters
    ----------
    alignment_file : str
        Path to alignment with sequences for which IDs
         should be retrieved
    uniprot_to_embl_table : str
        Path to uniprot to embl mapping database
    uniprot_to_embl_filename : str
        Write extracted mapping table to this file
    """
    # extract identifiers from sequence alignment
    with open(alignment_file) as f:
        sequence_id_list, _ = retrieve_sequence_ids(f)

    # store IDs in set for faster membership checking
    target_ids = set(sequence_id_list)

    # load matching entries from ID mapping table
    uniprot_to_embl = {}

    with open(uniprot_to_embl_table) as f:
        for line in f:
            uniprot_ac, _, ena_data = line.rstrip().split(" ")

            if uniprot_ac in target_ids:
                uniprot_to_embl[uniprot_ac] = ena_data
   
    # store mapping information for alignment to file
    with open(uniprot_to_embl_filename, "w") as of:
        for key, value in uniprot_to_embl.items():
            value_to_write = value.replace(",", ";")
            of.write(
                "{},{}\n".format(key, value_to_write)
            )


def load_uniprot_to_embl(uniprot_to_embl_filename):
    """
    Load a previously extracted Uniprot to EMBL ID
    mapping file and exclude inconsistent mappings
    
    Parameters
    ----------
    uniprot_to_embl_filename : str
        Path to Uniprot to EMBL mapping file

    Returns
    -------
    uniprot_to_embl : dict of tuple
        Mapping from Uniprot AC to ENA genomes
        and CDSs {uniprot_ac: [(ena_genome, ena_cds)]}
    """ 

    # reads the csv file uniprot_to_embl_filename
    # info a dictionary in the format uniprot_ac: annotation_string
    uniprot_to_embl_str = {}

    with open(uniprot_to_embl_filename) as inf:
        for line in inf:
            ac, annotation = line.rstrip().split(",")
            uniprot_to_embl_str[ac] = annotation

    # Clean up uniprot to embl dictionary
    uniprot_to_embl = {}
    
    for uniprot_ac, embl_mapping in uniprot_to_embl_str.items():
        # reformat the ena string as a list of tuples
        full_annotation = [
            x.split(":") for x in
            uniprot_to_embl_str[uniprot_ac].split(";")
        ]
        
        count_reads = defaultdict(list)
        
        for read, cds in full_annotation:
            count_reads[cds].append(read)

        uniprot_to_embl[uniprot_ac] = []

        # check how many reads are associated with a particular CDS, 
        # only keep CDSs that can be matched to *one* read
        for cds, reads in count_reads.items():
            if len(reads) == 1:
                uniprot_to_embl[uniprot_ac].append((reads[0], cds))

        # if none of the ena hits passed quality control, delete the entry
        if len(uniprot_to_embl[uniprot_ac]) == 0:
            del uniprot_to_embl[uniprot_ac]

    return uniprot_to_embl


def extract_embl_annotation(uniprot_to_embl_filename,
                            ena_genome_location_table,
                            genome_location_filename):
    """
    Reads CDS genomic location information for all entries mapped
    from Uniprot to EMBL; writes that information to a csv file 
    with the following columns:
    cds_id, genome_id, uniprot_ac, genome_start, genome_end
    
    Parameters
    ----------
    uniprot_to_embl_filename : str
        Path to Uniprot to EMBL mapping file
    ena_genome_location_table : str
        Path to ENA genome location database table which is a 
        a tsv file with the following columns:
        cds_id, genome_id, uniprot_ac, genome_start, genome_end
    genome_location_filename : str
        File to write containing CDS location info for
        target sequences
    
    Returns
    -------
    embl_cds_to_annotation: dict of tuple (str,str,int,int)
        {cds_id:(genome_id,uniprot_ac,genome_start,genome_end)}
    """
    uniprot_to_embl = load_uniprot_to_embl(
        uniprot_to_embl_filename
    )

    # initialize values
    cds_target_ids = set(
        sum(uniprot_to_embl.values(), [])
    )
    embl_cds_to_annotation = {}

    # extract the annotation
    with open(ena_genome_location_table) as inf:
        for line in inf:
            cds_id, read_id, uniprot_id, start, end = (
                line.rstrip().split("\t")
            )

            if cds_id in cds_target_ids:
                embl_cds_to_annotation[cds_id] = (
                    read_id, uniprot_id, start, end
                )

    # write the annotation
    with open(genome_location_filename, "w") as of:
        for key, value in embl_cds_to_annotation.items():
            of.write(
                key + "," + ",".join(value) + "\n"
            )


def load_embl_to_annotation(embl_cds_filename):
    """
    Read genomic location information for a set
    of EMBL CDSs

    Parameters
    ----------
    embl_cds_filename : str
        Path to file containing target CDS information 
        n tab-separated format (columns: cds_id, 
        genome_id, uniprot_ac, genome_start, genome_end)
    
    Returns
    -------
    embl_cds_to_annotation : dict of tuple(str, str ,int, int)
        Mapping dictionary of the form 
        {cds_id: (genome_id, uniprot_ac, genome_start, genome_end)}

    """

    embl_cds_to_annotation = {}

    with open(embl_cds_filename) as inf:
        for line in inf:
            cds_id, read_id, uniprot_id, start, end = (
                line.rstrip().split(",")
            )

            embl_cds_to_annotation[cds_id] = (
                read_id, uniprot_id, int(start), int(end)
            )
            
    return embl_cds_to_annotation
