"""
Functions for handling sequence
identifiers in alignments

Authors:
  Anna G. Green
"""
import re
from collections import defaultdict
from evcouplings.align import read_fasta

ID_EXTRACTION_REGEX = [
            # example: >UniRef100_H6SNJ6/11-331
            "^Uni\w+\_(\w+).*/",

            # example: >tr|Q1NYN0|Q1NYN0_9FLAO
            "^\w+\|(\w+)\|\w+\/",

            # example: >NQO8_THET8/1-365
            "^(\w+).*/.*$",

            # example: >Q60019|NQO8_THET8/1-365
            "^\w+\|\w+\|(\w+)",
]

def retrieve_sequence_ids(fileobj, regex=None):
    """
    Returns all identifiers in a FASTA alignment; 
    extracts ids based on the given regular expressions
    
    Note: if multiple regular expressions match the
    FASTA file header, will return the string extracted
    by the FIRST match
    
    Parameters
    ----------
    fileobj : file-like object
        FASTA alignment file
    regex : list of str, optional (default: None)
        Regular expression strings to extract sequence ids;
        if None uses list of default regular expressions
        to extract Uniprot and UniRef identifiers

    Returns
    -------
    list of str
        Sequence ids 
    dict
        Map from sequence id to list of str containing
        the full sequence headers corresponding to that
        sequence id
    """
    if regex is None:
        regex = ID_EXTRACTION_REGEX

    sequence_ids = []
    id_to_full_header = defaultdict(list)

    for current_id, _ in read_fasta(fileobj):
        for pattern in regex:
            m = re.match(pattern, current_id)
            # require a non-None match and at least one extracted pattern
            if m and len(m.groups()) > 0:
                # this extracts the parenthesized match
                sequence_ids.append(m.group(1))
                id_to_full_header[m.group(1)].append(current_id)
                break

    return sequence_ids, id_to_full_header
