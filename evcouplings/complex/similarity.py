"""
Functions for concatenating based on best hit or
best reciprocal hit to target

Authors:
  Anna G. Green
"""
import pandas as pd
import numpy as np
from evcouplings.align.alignment import (
    Alignment
)


def read_identity_file(identity_file):
    """
    Parameters
    ----------
    identity_file:str
        path to identity file

    Returns
    -------
    dict of str:str
        sequence identifier to species annotation
        
    """
    data = pd.read_csv(identity_file)
    id_to_identity = {}
    for id, ident in zip(data.id, data.identity_to_query):
        id_to_identity[id] = ident
    return id_to_identity


def read_annotation_file(annotation_file, column_1="OS", column_2 = "Tax"):
    """
    Parameters
    ----------
    annotation_file: str
        path to annotation file
    column: str, optional (default="OS")
        the column to use for species information

    Returns
    -------
    dict of str:str
        sequence identifier to species annotation
        
    """
    data = pd.read_csv(annotation_file, dtype=str)
    data = data.fillna(value="None")
    id_to_species = {}
    for id, species1, species2 in zip(data.id, data[column_1],data[column_2]):

        if not species1 is "None":

            if "TaxID=" in species1:
                species1 = species1.split(" TaxID=")[0]
            id_to_species[id] = species1
        else:

            if "TaxID=" in species2:
                species2 = species2.split(" TaxID=")[0]
            id_to_species[id] = species2

    return id_to_species


def most_similar_by_organism(similarities, id_to_organism):
    """
    for each species in the alignment, finds the sequence identifier
    from that species that is most similar to the target sequence

    Parameters:
    similarities: dict of str:int
        sequence identifier to identity
    id_to_organism:  dict of str:str
        sequence identifier to species annotation
        
    Returns
    -------
    species_to_most_similar: dict of str: (float,str)
        species identifier to tuple of (percent_identity,
        sequence identifier)
    
    """
    species_to_most_similar = {}
    for full_id, value in similarities.items():
        organism = id_to_organism[full_id]

        # if the current similarity is higher than the already saved similarity, save
        # the new value
        if organism in species_to_most_similar and \
                        value > species_to_most_similar[organism][0]:
            species_to_most_similar[organism] = (value, full_id)
        else:
            species_to_most_similar[organism] = (value, full_id)

    return species_to_most_similar


def filter_best_reciprocal(alignment, paralogs, species_to_most_similar, allowed_error=0.02):
    """
    Takes in a dictionary of the best hit to each genome
    Removes sequences that are not the best reciprocal hit to the query sequence

    Parameters
    ----------
    alignment: str
        path to sequence alignment file
    paralogs: list of str
        identifiers of paralogs to the query sequence
    species_to_most_similar: dict of str:(float,str)
        dictionary of species name pointing to the percent identity and
        sequence identifier that represents the most similar gene in that species
        to the target sequence
    allowed_error: float
        in order for a sequence to be filtered out of the alignment, it must be more identitcal to a
        paralog sequence than the target sequence by at least this amount

    Returns
    -------
    dict of str:(float,str)
        same as species_to_most_similar, but with sequences that are not best reciprocal hits filtered out
    int
        number of sequences filtered
    """
    ali = Alignment.from_file(open(alignment))

    # get the list of sequence identifiers
    ids = [x[1] for x in list(species_to_most_similar.values())]

    # Create an n_paralogs x n_sequences ndarray with the % identity to each paralog
    # in each row
    # note the identity here will be different than for the unfiltered alignment

    identity_array = np.zeros((len(list(paralogs)), len(ali.ids)), dtype=float)
    for idx, x in enumerate(paralogs):
        identities = ali.identities_to(ali[ali.id_to_index[x]])
        identity_array[idx, :] = identities

    num_discarded = 0
    new_species_to_most_similar = {}

    for key, value in species_to_most_similar.items():
        alignment_index = ali.id_to_index[value[1]]

        # if there are any identities higher than the identity to the query sequence
        if np.any(identity_array[:, alignment_index] > identity_array[0, alignment_index] + allowed_error):
            num_discarded += 1
        else:
            new_species_to_most_similar[key] = value

    return new_species_to_most_similar, num_discarded
