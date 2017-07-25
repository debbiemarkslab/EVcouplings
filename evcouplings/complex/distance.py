"""
Protocols for concatenating sequences based
on distance in genome

Authors:
  Anna G. Green
  Thomas A. Hopf
  Charlotta P.I. Schärfe
"""
from collections import defaultdict
from operator import itemgetter
# TODO: was setting backend here necessary? Library itself shouldn't do this except for the standalone components like app.py
import matplotlib.pyplot as plt


def get_genome_to_cds_list(sequence_id_list, uniprot_to_embl):
    """
    # TODO: this needs better documentation
    
    Parameters
    ----------
    sequence_id_list: list of str
        Retrieve CDS information for these
        Uniprot ACs
    uniprot_to_embl: dict of tuple
        Database of mappings from Uniprot AC to
        ENA genomes/CDSs 
        {uniprot_ac: [(ena_genome,ena_cds)]} 
        
    Returns
    -------
    genome_to_cds : dict of list
        {genome: [cds1, cds2]}
    cds_to_uniprot : dict of str
        {cds:uniprot_ac}
    """
    genome_to_cds = defaultdict(list)
    cds_to_uniprot = {}

    for id_ in sequence_id_list:
        if id_ in uniprot_to_embl:
            for embl_genome, embl_cds in uniprot_to_embl[id_]:
                genome_to_cds[embl_genome].append(embl_cds)
                cds_to_uniprot[embl_cds] = id_

    return genome_to_cds, cds_to_uniprot


def get_distance(annotation1, annotation2):
    """
    Compute distance between two CDS locations on
    the same genome
    
    Parameters
    ----------
    annotation1 : tuple of (str, str, int, int)
        EMBL annotation for first CDS in the format
        (genome_id, uniprot_ac, genome_start, genome_end)
    annotation2 : tuple of (str, str, int, int)
        EMBL annotation for second CDS
        
    Returns
    -------
    int
        Distance between gene 1 and gene 2 on the
        ENA genome
    """
    # extract the two locations from the annotation
    # sort each so that smaller genome position is first
    location_1 = sorted(
        (annotation1[2], annotation1[3])
    )

    location_2 = sorted(
        (annotation2[2], annotation2[3])
    )

    # sort the two annotations so that the one with
    # an earlier start site is first
    x, y = sorted((location_1, location_2))

    # if not overlapping, calculate the distance
    if x[0] <= x[1] < y[0]:
        return y[0] - x[1]

    # if overlapping, return 0
    return 0


def best_reciprocal_matching(possible_partners):
    """
    Amongst all possible pairings of CDSs, finds
    those where both sequences are closest neighbors
    to each other
    
    Parameters
    ----------
    possible_partners: dict of dict of int
        uniprot_id_1: {
            uniprot_id_2_A: genome_distance_A,
            uniprot_id_2_B: genome_distance_B
        }
    
    Returns
    -------
    id_pairing: list of tuple
        matched pairs of ids
    id_pair_to_distance: dict of tuple:int
        genome distance between pairs
    
    """

    # create a reverse matching of sequences (from 2nd alignment sequences to 1st alignment sequences)
    reverse_partners = defaultdict(list)
    for sid1 in possible_partners:
        for sid2 in possible_partners[sid1]:
            reverse_partners[sid2].append((sid1, possible_partners[sid1][sid2]))

    id_pairing = []
    id_pair_to_distance = {}
    # look at all sequences in first alignment, and check their possible partners
    for sid1 in possible_partners:
        # what is the closest sequence in second alignment wrt to genome distance?
        closest_sid2 = sorted(possible_partners[sid1].items(), key=itemgetter(1))[0][0]

        # find the closest sequence in first alignment to the above sequence in second alignment
        (closest_sid1_reverse, closest_sid1_reverse_dist) = sorted(reverse_partners[closest_sid2],
                                                                   key=itemgetter(1))[0]

        # check if matched sequences are mutually best friends
        if sid1 == closest_sid1_reverse:
            id_pairing.append((sid1, closest_sid2))
            id_pair_to_distance[(sid1, closest_sid2)] = closest_sid1_reverse_dist

    return id_pairing, id_pair_to_distance


def find_possible_partners(seq_ids_ali_1, seq_ids_ali_2,
                           uniprot_to_embl, embl_to_annotation):
    """
    Constructs a dictionary of all possible sequence pairings
    between the two monomer alignments
    
    Parameters
    ----------
    seq_ids_ali_1: list of str
        Uniprot ACs for entries in first alignment
    seq_ids_ali_2: list of str
        Uniprot ACs for entries in second alignment
    uniprot_to_embl: dict of tuple
        Mapping from Uniprot ACs to ENA genomes/CDSs
        this sequence maps to:
        {uniprot_ac: [(ena_genome, ena_cds)]}
    embl_to_annotation: dict of tuple (str, str, int, int)
        Mapping from CDS to genomic location of the CDS
        {cds_id: (genome_id, uniprot_ac, genome_start, genome_end)}
    
    Returns
    -------
    possible_partners: dict of dict of str:int
        uniprot_id_1:{uniprot_id_2_A:genome_distance_A,
                      uniprot_id_2_B:genome_distance_B}
    """
    ali1_genomes_to_cds, ali1_cds_to_uniprot = get_genome_to_cds_list(
        seq_ids_ali_1, uniprot_to_embl
    )

    ali2_genomes_to_cds, ali2_cds_to_uniprot = get_genome_to_cds_list(
        seq_ids_ali_2, uniprot_to_embl
    )

    # store possible pairings here
    possible_partners = defaultdict(dict)

    # iterate over EMBL genomes found in both alignments
    for genome in ali1_genomes_to_cds:
        # do not compare if genome is not found in second alignment at all
        if genome in ali2_genomes_to_cds:

            # compare all CDSs between both alignments
            for cds1 in ali1_genomes_to_cds[genome]:
                for cds2 in ali2_genomes_to_cds[genome]:

                    # doublecheck if there really is start/end info for the two current CDSs
                    if cds1 in embl_to_annotation and cds2 in embl_to_annotation:
                        genome_location_distance = get_distance(
                            embl_to_annotation[cds1], embl_to_annotation[cds2]
                        )

                        uniprot_id1, uniprot_id2 = (
                            ali1_cds_to_uniprot[cds1], ali2_cds_to_uniprot[cds2]
                        )

                        if uniprot_id2 in possible_partners[uniprot_id1]:
                            possible_partners[uniprot_id1][uniprot_id2] = min(
                                possible_partners[uniprot_id1][uniprot_id2],
                                genome_location_distance
                            )
                        else:
                            possible_partners[uniprot_id1][uniprot_id2] = (
                                genome_location_distance
                            )

    return possible_partners


def filter_ids_by_distance(id_pairing,
                           id_pair_to_distance,
                           genome_distance_threshold=999999999):
    """
    Filters the list of paired ids to include only those
    below the genome distance threshold

    Parameters
    ----------
    id_pairing: list of tuple
        matched pairs of ids
    id_pair_to_distance: dict of tuple:int
        genome distance between pairs
    genome_distance_threshold: int
        maximum distance on genome allowed

    Returns
    -------
    filtered_id_pairing: list of tuple
        matched pairs of ids that are closer than 
        the threshold        
    """
    filtered_id_pairing = []

    # TODO: default parameter for genome_distance_threshold should be None
    for id_pair in id_pairing:
        if id_pair in id_pair_to_distance and \
                        id_pair_to_distance[id_pair] <= genome_distance_threshold:
            filtered_id_pairing.append(id_pair)

    return filtered_id_pairing


def plot_distance_distribution(id_pair_to_distance,outfile):
    """
    plots the distribution of genome distances
    
    Parameters
    ----------
    id_pair_to_distance: dict of tuple: int
        pairs of identifiers and their distances on the genome
    outfile: str
        path to file to save plot
    """
    
    distances = list(id_pair_to_distance.values())
    distances = sorted(distances)

    # TODO: could be done using range more cleanly
    cdf = [i for i, _ in enumerate(distances)]

    fig = plt.figure(figsize=(8,6))
    ax1 = fig.gca()
    ax1.set_xscale("log")
    ax1.set_xlim(xmin=1,xmax=max(distances))
    ax1.set_ylabel("Number of sequences")
    ax1.set_xlabel("Genome distance (bases)")
    ax1.plot(distances,cdf)
    
    plt.savefig(outfile)
