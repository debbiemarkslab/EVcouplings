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

def extract_uniprot_to_embl(sequence_id_list, 
                         mapping_filename):
    """
    
    extracts the uniprot:embl mapping data from a given file
    only extracts information for uniprot ACs included
    in the given list to save memory
    
    Parameters
    ----------
    sequence_id_list: list of str
        uniprot ACs 
    mapping_filename: str
        path to uniprot_embl mapping file
        
    Returns
    -------
    uniprot_to_embl: dict of str
        points uniprot acs to embl annotation strings in the format 
            ena_genome1:ena_cds1,ena_genome2:ena_cds2

    """

    uniprot_to_embl = dict((id, "") for id in sequence_id_list)

    for line in open(mapping_filename,'r'):
        uniprot_ac,_,ena_data = line.rstrip().split(' ')
        if uniprot_ac in uniprot_to_embl:
            uniprot_to_embl[uniprot_ac] = ena_data
    return uniprot_to_embl

def write_uniprot_to_embl_str(uniprot_to_embl, 
                         mapping_filename):
    """
    
    writes the uniprot to embl mapping in space-separated value format
    
    Parameters
    ----------
    uniprot_to_embl_str: dict of str
        points uniprot acs to embl annotation strings in the format 
            ena_genome1:ena_cds1,ena_genome2:ena_cds2
    mapping_filename: str
        path to uniprot_embl mapping file

    """

    of = open(mapping_filename,'w')
    for key,value in uniprot_to_embl.iter():
        of.write('{} - {}\n'.format(key,value))
    of.close()

def load_uniprot_to_embl(sequence_id_list,
                          uniprot_embl_mapping_filename):
    
    """
    reads the uniprot:embl mapping from a file and cleans up the 
    data
    
    Parameters
    ----------
    sequence_id_list: list of str
        uniprot ACs 
    mapping_filename: str
        path to uniprot_embl mapping file
        
    Returns
    -------
    uniprot_to_embl: dict of tuple
        {uniprot_ac:[(ena_genome,ena_cds)]}
    

    """             
    
    #try to open then given file
    if os.path.exists(uniprot_embl_mapping_filename) and \
        os.path.getsize(uniprot_embl_mapping_filename)>0:
        uniprot_to_embl_str = extract_uniprot_to_embl(sequence_id_list,uniprot_embl_mapping_filename)
    
    #if file is empty, load annotation from database and write file
    else:
        uniprot_to_embl_str = extract_uniprot_to_embl(sequence_id_list,UNIPROT_EMBL_MAPPING_FILE)
        write_uniprot_to_embl_str(uniprot_to_embl_str,mapping_filename)
        
        
    # Clean up uniprot to embl dictionary
    
    uniprot_to_embl = {}
    
    for uniprot_ac, embl_mapping in uniprot_to_embl_str.items():

        #if no mapping, delete entry
        if embl_mapping == "" or len(embl_mapping)==0: 
            continue

        #else, reformat the ena string as a list of tuples
        else:
            full_annotation = [x.split(':') for x in uniprot_to_embl_str[uniprot_ac].split(",")]
            
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

def extract_embl_annotation(uniprot_to_embl,
                            embl_cds_filename):
    """
    reads CDS location information for all entries mapped from Uniprot to EMBL
    only extracts information for EMBL entries found in the given list
    
    Parameters
    ----------
    uniprot_to_embl: dict of tuple
        {uniprot_ac:[(ena_genome,ena_cds)]} 
    
    embl_cds_filename: str
        path to file containing cds information in tab-separated format
        cds_id,genome_id,uniprot_ac,genome_start,genome_end
    
    Returns
    -------
    embl_cds_to_annotation: dict of tuple (str,str,int,int)
        {cds_id:(genome_id,uniprot_ac,genome_start,genome_end)}
    
            
    """
    embl_id_list = sum(uniprot_to_embl.values(), [])
    cds_id_hash = dict((cds, True) for (read, cds) in embl_id_list)

    embl_cds_to_annotation = {}
    for line in open(embl_cds_filename,'r'):
        cds_id, read_id, uniprot_id, start, end = line.rstrip().split('\t')
        if cds_id in cds_id_hash: 
            embl_cds_to_annotation[cds_id] = (read_id, uniprot_id, int(start), int(end))
    return embl_cds_to_annotation

def write_embl_cds_to_annotation(embl_cds_to_annotation, 
                             filename):
    """
    
    writes the embl cds mapping in tab-separated value format
    
    Parameters
    ----------
    embl_cds_to_annotation: dict of tuple (str,str,int,int)
        {cds_id:(genome_id,uniprot_ac,genome_start,genome_end)}
    filename: str
        path to output file
    """

    of = open(filename,'w')
    for key,value in embl_cds_to_annotation.iter():
        of.write(key+'\t'+'\t'.join(value)+'\n')
    of.close()


def load_embl_to_annotation(uniprot_to_embl,
                         embl_cds_filename):
    """
    
    
    Parameters
    ----------
    uniprot_to_embl: dict of tuple
        {uniprot_ac:[(ena_genome,ena_cds)]} 
    
    embl_cds_filename: str
        path to file containing cds information in tab-separated format
        cds_id,genome_id,uniprot_ac,genome_start,genome_end
    
    Returns
    -------
    embl_cds_to_annotation: dict of tuple (str,str,int,int)
        {cds_id:(genome_id,uniprot_ac,genome_start,genome_end)}
    
            
    """
    #try to open the given file
    if os.path.exists(embl_cds_filename) and \
        os.path.getsize(embl_cds_filename)>0:
        embl_cds_to_annotation = extract_embl_annotation(uniprot_to_embl,embl_cds_filename)
    
    #if file is empty, load annotation from database and write file
    else:
        embl_cds_to_annotation = extract_embl_annotation(uniprot_to_embl,EMBL_CDS_DATABASE)
        write_embl_cds_to_annotation(embl_cds_to_annotation,embl_cds_filename)
            
    return embl_cds_to_annotation

def get_genome_to_cds_list(sequence_id_list, 
                           uniprot_to_embl):
    """
    Parameters
    ----------
    sequence_id_list: list of str
        uniprot ACs 
        
    uniprot_to_embl: dict of tuple
        {uniprot_ac:[(ena_genome,ena_cds)]} 
        
    Returns
    -------
    genome_to_cds:dict of list
        {genome:[cds1,cds2]}
    cds_to_uniprot:dict of str
        {cds:uniprot_ac}
    """
    genome_to_cds = defaultdict(list)
    cds_to_uniprot = {}

    for id in sequence_id_list:
        if id in uniprot_to_embl:
            for embl_genome, embl_cds in uniprot_to_embl[id]:
                genome_to_cds[embl_genome].append(embl_cds)
                cds_to_uniprot[embl_cds] = id

    return genome_to_cds, cds_to_uniprot

def get_distance(annotation1, annotation2):
    """
    Paramters
    ---------
    annotation1,annotation2: tuple of (str,str,int,int)
        EMBL annotation in the format
        (genome_id,uniprot_ac,genome_start,genome_end)
        
    returns: int
        distance between gene 1 and gene 2 in the Ena genome
    """
    #extract the two locations from the annotation
    #sort each so that smaller genome position is first
    location_1 = sorted((annotation1[2],annotation1[3]))
    location_2 = sorted((annotation2[2],annotation2[3]))
    
    #sort the two annotations so that the one with an earlier start site is first
    x,y = sorted((location_1,location_2))
    
    if x[0]<=x[1] and x[1]<y[0]: #if not overlapping, calculate the distance
        return y[0] - x[1]
    
    #if overlapping, return 0
    return 0
