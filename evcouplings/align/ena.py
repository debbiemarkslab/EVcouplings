"""
Protocols for mapping uniprot identifiers to EMBL/ENA
database and extracting genomic distances

Authors:
  Anna G. Green
  Thomas A. Hopf
  Charlotta P.I. Scharfe
"""
import os
from operator import itemgetter
from collections import defaultdict
from evcouplings.align.alignment import retrieve_sequence_ids

def extract_uniprot_to_embl(alignment_file, 
                            uniprot_to_embl_table,
                            uniprot_to_embl_filename):
    """
    
    extracts the uniprot:embl mapping data from the database table
    
    Parameters
    ----------
    sequence_id_list: list of str
        uniprot ACs 
    uniprot_to_embl_table: str
        path to uniprot to embl database
    uniprot_to_embl_filename: str
        path to file to write 
        

    """

    sequence_id_list,_ = retrieve_sequence_ids(open(alignment_file,'r'))

    #load the info
    uniprot_to_embl = dict((id, "") for id in sequence_id_list)

    for line in open(uniprot_to_embl_table,'r'):
        uniprot_ac,_,ena_data = line.rstrip().split(' ')

        if uniprot_ac in uniprot_to_embl:
            uniprot_to_embl[uniprot_ac] = ena_data
   
    #write the information
    of = open(uniprot_to_embl_filename,'w')

    for key,value in uniprot_to_embl.items():
        #if no mapping, delete entry
        if value == "" or len(value)==0: 
            continue
        value_to_write = value.replace(',',';')
        of.write('{},{}\n'.format(key,value_to_write))

    of.close() 


def load_uniprot_to_embl(uniprot_to_embl_filename):
    '''
    Parameters
    ----------

    reads the uniprot:embl mapping from a file and cleans up the 
    data
    
    Parameters
    ----------

    uniprot_to_embl_filename: str
        path to unxiprot_embl mapping file

    sequence_id_list: list of str
        uniprot ACs 
    mapping_filename: str
        path to uniprot_embl mapping file      
    Returns
    -------
    uniprot_to_embl: dict of tuple
        {uniprot_ac:[(ena_genome,ena_cds)]}
    
    ''' 

    def _read_uniprot_to_embl(uniprot_to_embl_filename) :
        #reads the csv file uniprot_to_embl_filename
        #returns a dictionary in the format Uniprot_ac:annotation_string          
        uniprot_to_embl_str = {}
        for line in open(uniprot_to_embl_filename,'r'):
            ac,annotation = line.rstrip().split(',')
            uniprot_to_embl_str[ac]=annotation
        return uniprot_to_embl_str

    uniprot_to_embl_str = _read_uniprot_to_embl(uniprot_to_embl_filename)

    # Clean up uniprot to embl dictionary

    uniprot_to_embl = {}
    
    for uniprot_ac, embl_mapping in uniprot_to_embl_str.items():

        #reformat the ena string as a list of tuples
        full_annotation = [x.split(':') for x in uniprot_to_embl_str[uniprot_ac].split(";")]
        
        count_reads = defaultdict(list)
        
        for read, cds in full_annotation:
            count_reads[cds].append(read)

        uniprot_to_embl[uniprot_ac] = []

        # check how many reads are associated with a particular CDS, 
        # only keep CDS's that can be matched to *one* read
        for cds, reads in count_reads.items():
            if len(reads) == 1:
                uniprot_to_embl[uniprot_ac].append((reads[0], cds))

        #if none of the ena hits passed quality control, delete the entry
        if len(uniprot_to_embl[uniprot_ac]) == 0:
            del uniprot_to_embl[uniprot_ac]

    return uniprot_to_embl

def extract_embl_annotation(uniprot_to_embl_filename,
                            ena_genome_location_table,
                            genome_location_filename):
    """
    reads CDS location information for all entries mapped from Uniprot to EMBL
    write that information to a csv file with entries:
    cds_id,genome_id,uniprot_ac,genome_start,genome_end
    
    Parameters
    ----------
    uniprot_to_embl_filename: str
        path to uniprot_embl mapping file
    
    ena_genome_location_table: str
        path to ena genome location database table.
        A tsv file in format:
        cds_id,genome_id,uniprot_ac,genome_start,genome_end

    genome_location_filename: str
        file to write containing CDS location info

    
    Returns
    -------
    embl_cds_to_annotation: dict of tuple (str,str,int,int)
        {cds_id:(genome_id,uniprot_ac,genome_start,genome_end)}
    
            
    """
    uniprot_to_embl = load_uniprot_to_embl(uniprot_to_embl_filename)

    #set up 
    embl_id_list = sum(uniprot_to_embl.values(), [])
    cds_id_hash = dict((cds, True) for (read, cds) in embl_id_list)
    embl_cds_to_annotation = {}

    #extract the annotation
    for line in open(ena_genome_location_table,'r'):
        cds_id, read_id, uniprot_id, start, end = line.rstrip().split('\t')
        if cds_id in cds_id_hash: 
            embl_cds_to_annotation[cds_id] = (read_id, uniprot_id, start, end)

    #swrite the annotation
    of = open(genome_location_filename,'w')
    for key,value in embl_cds_to_annotation.items():
        of.write(key+','+','.join(value)+'\n')
    of.close()

def load_embl_to_annotation(embl_cds_filename):
    """

    Parameters
    ----------
    embl_cds_filename: str
        path to file containing cds information in tab-separated format
        cds_id,genome_id,uniprot_ac,genome_start,genome_end
    
    Returns
    -------
    embl_cds_to_annotation: dict of tuple (str,str,int,int)
        {cds_id:(genome_id,uniprot_ac,genome_start,genome_end)}
    
            
    """

    embl_cds_to_annotation = {}
    for line in open(embl_cds_filename,'r'):
        cds_id, read_id, uniprot_id, start, end = line.rstrip().split(',')
        embl_cds_to_annotation[cds_id] = (read_id, uniprot_id, int(start), int(end))
            
    return embl_cds_to_annotation
