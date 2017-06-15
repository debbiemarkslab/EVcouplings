"""
Functions for concatenating based on best hit to target

Authors:
  Anna G. Green
"""
import pandas as pd


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


def read_annotation_file(annotation_file, column="OS"):
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
    data = pd.read_csv(annotation_file)
    id_to_species = {}
    for id, species in zip(data.id, data[column]):
        id_to_species[id] = species
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
