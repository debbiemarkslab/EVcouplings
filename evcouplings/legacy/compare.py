import evcouplings.legacy.pdb3d
import evcouplings.legacy.pdb_maps


def compute_distance_map(pdb_id, pdb_chain, output_file=None):
    """
    Compute monomer contact map from single PDB chain
    """

    if output_file is None:
        output_file = pdb_id + pdb_chain + ".txt"

    pdb3d.make_contact_map(
        pdb_id, pdb3d.single_chain(pdb_chain), output_file
    )

    # load contact map
    cmap_new = pdb_maps.ContactMap(output_file)

    # compatibility layer fudge: create older contact map object
    # which has has all the plotting functionality we need
    cmap = cmap_new.submatrix(pdb_chain, pdb_chain).create_old_contact_map()

    return cmap


def add_distances(df, dist_map):
    df = df.copy()
    df.loc[:, "dist"] = [
        dist_map.contact_map[i][j] if i in dist_map.contact_map and j in dist_map.contact_map[i]
        else float("nan")
        for (i, j) in zip(df.i, df.j)
    ]
    return df


def add_precision(df, distance_threshold=5):
    df = df.copy()
    df.loc[:, "tp_{}".format(distance_threshold)] = (
        (df.loc[:, "dist"] <= distance_threshold).cumsum() / df.loc[:, "dist"].notnull().cumsum()
    )
    return df


def ec_scores_compared(ecs, contact_map, dist_cutoff=None, output_file=None):
    x = add_distances(ecs, contact_map)
    if dist_cutoff is not None:
        x = add_precision(x, dist_cutoff)

    if output_file is not None:
        x.to_csv(output_file, index=False)

    return x
