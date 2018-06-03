"""
Functions for concatenating based on best hit or
best reciprocal hit to target

Authors:
  Anna G. Green
"""
import pandas as pd
import numpy as np
from evcouplings.align.alignment import (
    Alignment, parse_header
)
from evcouplings.utils import InvalidParameterError

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
        an annotation dataframe with columns id, species, and annotation
    """
    data = pd.read_csv(annotation_file, dtype=str)

    # initialize the column to extract the species information from
    annotation_column = None
    current_num_annotations = 0

    # Determine whether to extract based on the "OS" field
    # or the "Tax" field. Generally, OS is for Uniprot
    for column in SPECIES_ANNOTATION_COLUMNS:
        # if this column contains more non-null values
        if column not in data:
            continue

        num_annotations = sum(data[column].notnull())
        if num_annotations > current_num_annotations:
            # use that column to extract data
            annotation_column = column
            current_num_annotations = num_annotations

    # if we did not find an annotation column, return an error
    if annotation_column is None:
        raise InvalidParameterError(
            "provided annotation file {} has no annotation information".format(
                annotation_file
            )
        )

    # creates a new column called species with the species annotations
    data.loc[:, "species"] = data.loc[:, annotation_column]

    return data[["id", "name", "species"]]

def most_similar_by_organism(similarities, id_to_organism):
    """
    For each species in the alignment, finds the sequence 
    from that species that is most similar to the target sequence

    Parameters
    ----------
    similarities : pd.DataFrame
        The contents of identities.csv
    id_to_organism :  pd.DataFrame
        The contents of annotation.csv
        
    Returns
    -------
    pd.DataFrame
        With columns id, species, identity_to_query.
        Where each row is the sequence in a particular species
        that was the most similar to the target sequence.
    
    """
    species_to_most_similar = []

    # merge the two data frames
    data = similarities.merge(id_to_organism, on="id")

    # find the most similar in every organism
    most_similar_in_species = data.sort_values(by="identity_to_query").groupby("species").last()
    most_similar_in_species["species"] = most_similar_in_species.index
    most_similar_in_species = most_similar_in_species.reset_index(drop=True)

    return most_similar_in_species


def find_paralogs(target_id, id_to_organism, similarities, identity_threshold):
    """
    Finds all the sequences in the alignment that originate
    from the same species as the target_id, if those sequences
    are below the identity threshold (ie, are diverged from
    the query)

    Parameters
    ----------
    target_id : str
        Full identifier of the query sequence
    similarities : pd.DataFrame
        The contents of identities.csv
    id_to_organism :  pd.DataFrame
        The contents of annotation.csv
    identity_threshold : float
        Sequences above this identity to the query are not considered paralogs

    Returns
    -------
    pd.DataFrame
        with columns id, species, identity_to_query
        Entries are paralogs found in the same genome as the query id
    """

    # output of parse_header is (ID, region_start, region_end)
    base_query_id, _, _ = parse_header(target_id)

    # get all the rows that have an id that contains the
    # query id. This includes the focus sequence and its hit to
    # itself in the database.
    annotation_data = similarities.merge(id_to_organism, on="id")
    contains_annotation = [base_query_id in x for x in annotation_data.id]
    query_hits = annotation_data.loc[contains_annotation , :]
    # get the species annotation for the query sequence
    query_species = list(query_hits.species.dropna())

    # get all rows that are from the query species
    paralogs = annotation_data.query("species == @query_species")

    # confirm that paralogs are below the similarity threshold
    # ie, are diverged in sequence space from the query
    paralogs = paralogs.query("identity_to_query < @identity_threshold")
    return paralogs


def filter_best_reciprocal(alignment, paralogs, most_similar_in_species, allowed_error=0.02):
    """
    Takes in a dictionary of the best hit to each genome
    Removes sequences that are not the best reciprocal hit to the query sequence

    Parameters
    ----------
    alignment : str
        Path to sequence alignment file
    paralogs : pd.DataFrame
        Rows correspond to paralogs to the query sequence
        Created by find_paralogs() function
    most_similar_in_species : pd.DataFrame
        Contains the id, species name, and percent identity to query
        for each sequence that was the best hit to the query in its 
        respective species
    allowed_error : float
        In order for a sequence to be filtered out of the alignment, 
        it must be more identitical to a paralog sequence than the 
        target sequence by at least this amount

    Returns
    -------
    pd.DataFrame
        Contains the id, species name, and percenty identity to query
        for each sequence that was the best reciprocal hit to the query sequence
    """
    with open(alignment, "r") as inf:
        ali = Alignment.from_file(inf)

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
    # in its respective species...
    for index, row in most_similar_in_species.iterrows():

        # get the index of that sequence in the alignment.
        alignment_index = ali.id_to_index[row.id]

        # Keep sequences if they are the best reciprocal hit -
        # i.e., that sequence is not more similar to any paralog
        # than it is to the query sequence
        if np.all(identity_mat[:, alignment_index] < row.identity_to_query + allowed_error):

            indices_to_keep.append(index)

    return most_similar_in_species.loc[indices_to_keep, :]
