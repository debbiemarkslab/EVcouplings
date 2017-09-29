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

SPECIES_ANNOTATION_COLUMNS = ["OS", "Tax"]


def read_species_annotation_table(annotation_file):
    """
    Reads in the annotation.csv file and decides which column
    contains the species information - this differs for uniref 
    and uniprot alignments. Adds a new column called "species"
    with the correct annotations. 

    Note: Uniprot and Uniref databases can have different 
    annotations even for the same sequence. 


    Parameters
    ----------
    annotation_file : str
        path to annotation file

    Returns
    -------
    pd.DataFrame
        the annotation dataframe with an additional column
        called species. 

    """
    data = pd.read_csv(annotation_file, dtype=str)

    # initialize the column to extract the species information from
    annotation_column = None

    # Determine whether to extract based on the "OS" field
    # or the "Tax" field. Generally, OS is for Uniprot
    # and "Tax" is for Uniref
    for column in SPECIES_ANNOTATION_COLUMNS:
        # if this column contains non-null values
        if not data[column].isnull().all():
            # use that column to extract data
            annotation_column = column
            break

    # creates a new column called species with the species annotations
    data["species"] = data[annotation_column]

    return data


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
   	pd.DataFrame
        with columns id, species, identity_to_query
        where each row is the sequence in a particular species
        that was the most similar to the target sequence.
    
    """
    species_to_most_similar = []

    # merge the two data frames
    data = similarities.merge(id_to_organism, on="id")

    # group by species
    species_groups = data.groupby("species")
    for species in species_groups.groups.keys():

        species_subset = species_groups.get_group(species)

        # if there is only one sequence in that species,
        # keep it
        if len(species_subset) == 1:
            id = species_subset.id.tolist()[0]
            identity_to_query = species_subset.identity_to_query.tolist()[0]

        # if there are multiple sequences, get the
        # most similar sequence
        else:
            max_identity_index = species_subset.identity_to_query.idxmax()
            id = species_subset.loc[max_identity_index, :].id
            identity_to_query = species_subset.loc[max_identity_index, :].identity_to_query

        species_to_most_similar.append([
            id,
            species,
            identity_to_query
        ])

    most_similar_in_species = pd.DataFrame(
        species_to_most_similar,
        columns=["id", "species", "identity_to_query"]
    )

    return most_similar_in_species


def find_paralogs(query_id, annotation_data, identity_data,
                  identity_threshold):
    """
    Finds all the sequences in the alignment that originate
    from the same species as the query_id, if those sequences
    are below the identity threshold (ie, are diverged from
    the query)

    Parameters
    ----------
    query_id : str
        full identifier of the query sequence
    annotation_data : pd.DataFrame
        the contents of the annotation.csv file with an additional column
        called species, which contains the species annotation to use
    identity_data : pd.DataFrame
        the contents of the identities.csv file
    identity_threshold : float
        sequences above this identity to the query are not considered paralogs

    Returns
    -------
    pd.DataFrame
        with columns created by merging annotation_data and identity_data
        entries are paralogs found in the same genome as the query id
    """

    base_query = query_id.split("/")[0]

    # merge the annotation and identity dataframes
    data = annotation_data.merge(identity_data, on="id")

    # get all the rows that have an id that contains the
    # query id. This includes the focus sequence and its hit to
    # itself in the database.
    query_hits = data[
        [base_query in str(x) for x in list(data.id)]
    ]

    # get the species annotation for the query sequence
    query_species = list(query_hits.species.dropna())

    # get all rows that are from the query species
    paralogs = data.query("species == @query_species")

    # confirm that paralogs are below the similarity threshold
    # ie, are diverged in sequence space from the query
    paralogs = paralogs[paralogs.identity_to_query < identity_threshold]
    return paralogs


def filter_best_reciprocal(alignment, paralogs, most_similar_in_species, allowed_error=0.02):

    """
    Takes in a dictionary of the best hit to each genome
    Removes sequences that are not the best reciprocal hit to the query sequence

    Parameters
    ----------
    alignment: str
        path to sequence alignment file
    paralogs: pd.DataFrame
        rows correspond to paralogs to the query sequence
        Created by find_paralogs() function
    most_similar_in_species: pd.DataFrame
        contains the id, species name, and percent identity to query
        for each sequence that was the best hit to the query in its respective species
    allowed_error: float
        in order for a sequence to be filtered out of the alignment, it must be more identitical to a
        paralog sequence than the target sequence by at least this amount

    Returns
    -------
      pd.DataFrame
        contains the id, species name, and percenty identity to query
        for each sequence that was the best reciprocal hit to the query sequence
    """
    ali = Alignment.from_file(open(alignment))

    # Create an n_paralogs x n_sequences ndarray
    # where entry i,j is percent identity of paralog i to sequence j
    # note the identity here will be different than for the unfiltered alignment

    # initialize the matrix
    identity_mat = np.zeros((len(paralogs), len(ali.ids)), dtype=float)

    for idx, paralog_id in enumerate(paralogs.id):
        # calculate the % identity of every seq in the alignment to current paralog
        identities = ali.identities_to(ali[ali.id_to_index[paralog_id]])

        # save the results
        identity_mat[idx, :] = identities

    indices_to_keep = []
    # for every sequence in the alignment that is the most similar to the query
    # in its respective species
    for index, row in most_similar_in_species.iterrows():

        # get the index of that sequence in the alignment
        alignment_index = ali.id_to_index[row.id]

        # keep sequences if they are the best reciprocal hit -
        # ie, that sequence is not more similar to any paralog
        # than it is to the query sequence
        if np.all(identity_mat[:, alignment_index] < \
           row.identity_to_query + allowed_error):

            indices_to_keep.append(index)

    return most_similar_in_species.loc[indices_to_keep,:]





