"""
Distance calculations on PDB 3D coordinates

Authors:
  Thomas A. Hopf
  Anna G. Green (remap_complex_chains)
"""

from collections import Counter
from itertools import combinations
from operator import itemgetter
from copy import deepcopy

import numpy as np
import pandas as pd
from numba import jit

from evcouplings.compare.pdb import load_structures
from evcouplings.utils.constants import AA1_to_AA3
from evcouplings.utils.system import create_prefix_folders
from evcouplings.utils.helpers import find_segments


@jit(nopython=True)
def _distances(residues_i, coords_i, residues_j, coords_j, symmetric):
    """
    Compute minimum atom distances between residues. If used on
    a single atom per residue, this function can e.g. also compute
    C_alpha distances.

    Parameters
    ----------
    residues_i : np.array
        Matrix of size N_i x 2, where N_i = number of residues
        in PDB chain used for first axis. Each row of this
        matrix contains the first and last (inclusive) index
        of the atoms comprising this residue in the coords_i
        matrix
    coords_i : np.array
        N_a x 3 matrix containing 3D coordinates of all atoms
        (where N_a is total number of atoms in chain)
    residues_j : np.array
        Like residues_i, but for chain used on second axis
    coords_j : np.array
        Like coords_j, but for chain used on second axis

    Returns
    -------
    dists : np.array
        Matrix of size N_i x N_j containing minimum atom
        distance between residue i and j in dists[i, j]
    """
    LARGE_DIST = 1000000

    N_i, _ = residues_i.shape
    N_j, _ = residues_j.shape

    # matrix to hold final distances
    dists = np.zeros((N_i, N_j))

    # iterate all pairs of residues
    for i in range(N_i):
        for j in range(N_j):
            # limit computation in symmetric case and
            # use previously calculated distance
            if symmetric and i >= j:
                dists[i, j] = dists[j, i]
            else:
                range_i = residues_i[i]
                range_j = residues_j[j]
                min_dist = LARGE_DIST

                # iterate all pairs of atoms for residue pair;
                # end of coord range is inclusive, so have to add 1
                for a_i in range(range_i[0], range_i[1] + 1):
                    for a_j in range(range_j[0], range_j[1] + 1):
                        # compute Euclidean distance between atom pair
                        cur_dist = np.sqrt(
                            np.sum(
                                (coords_i[a_i] - coords_j[a_j]) ** 2
                            )
                        )
                        # store if this is a smaller distance
                        min_dist = min(min_dist, cur_dist)

                dists[i, j] = min_dist

    return dists


class DistanceMap:
    """
    Compute, store and accesss pairwise residue
    distances in PDB 3D structures
    """
    # separator between distance map id and field name in aggregated dataframes
    _id_separator = "::"

    def __init__(self, residues_i, residues_j, dist_matrix, symmetric):
        """
        Create new distance map object

        Parameters
        ----------
        residues_i : pandas.DataFrame
            Table containing residue annotation for
            first axis of distance matrix
        residues_j : pandas.DataFrame
            Table containing residue annotation for
            second axis of distance matrix
        dist_matrix : np.array
            2D matrix containing residue distances
            (of size len(residues_i) x len(residues_j))
        symmetric : bool
            Indicates if distance matrix is symmetric
        """
        self.residues_i = residues_i
        self.residues_j = residues_j
        self.dist_matrix = dist_matrix
        self.symmetric = symmetric

        # create mappings from identifier to entry in distance matrix
        self.id_map_i = {
            id_: i for (i, id_) in enumerate(self.residues_i.id.values)
        }

        self.id_map_j = {
            id_: j for (j, id_) in enumerate(self.residues_j.id.values)
        }

        # attribute that allows to set a unique identifier for this distance map,
        # used during aggregation
        self.id = None

    @classmethod
    def _extract_coords(cls, coords):
        """
        Prepare coordinates as suitable input
        for _distances() function

        Parameters
        ----------
        coords : pandas.DataFrame
            Atom coordinates for PDB chain
            (.coords property of Chain object)

        Returns
        -------
        atom_ranges : np.array
            Matrix of size N_i x 2, where N_i = number
            of residues in PDB chain. Each row of this matrix
            contains the first and last (inclusive) index of
            the atoms comprising this residue in the xyz_coords
            matrix
        xyz_coords : np.array
            N_a x 3 matrix containing 3D coordinates
            of all atoms (where N_a is total number of
            atoms in chain)
        """
        # First, reset index and drop because indices might
        # be discontinuous after removal of atoms (we need that
        # indices correspond to boundaries of each residue in
        # raw numpy array).
        # Second, move index into column,
        # so we can access values after groupby
        C = coords.reset_index(drop=True).reset_index()

        # matrix of 3D coordinates
        xyz_coords = np.stack(
            (C.x.values, C.y.values, C.z.values)
        ).T

        # extract what the first and last atom index
        # of each residue is
        C_grp = C.groupby("residue_index")
        atom_ranges = np.stack(
            (C_grp.first().loc[:, "index"].values,
             C_grp.last().loc[:, "index"].values)
        ).T

        return atom_ranges, xyz_coords

    @classmethod
    def from_coords(cls, chain_i, chain_j=None):
        """
        Compute distance matrix from PDB chain
        coordinates.

        Parameters
        ----------
        chain_i : Chain
            PDB chain to be used for first axis of matrix
        chain_j : Chain, optional (default: None)
            PDB chain to be used for second axis of matrix.
            If not given, will be set to chain_i, resulting
            in a symmetric distance matrix

        Returns
        -------
        DistanceMap
            Distance map computed from given
            coordinates
        """
        ranges_i, coords_i = cls._extract_coords(chain_i.coords)

        # if no second chain given, compute a symmetric distance
        # matrix (mainly relevant for intra-chain contacts)
        if chain_j is None:
            symmetric = True
            chain_j = chain_i
            ranges_j, coords_j = ranges_i, coords_i
        else:
            symmetric = False
            ranges_j, coords_j = cls._extract_coords(chain_j.coords)

        # compute distances using jit-compiled function
        dists = _distances(
            ranges_i, coords_i,
            ranges_j, coords_j,
            symmetric
        )

        # create distance matrix object
        return cls(
            chain_i.residues, chain_j.residues,
            dists, symmetric
        )

    @classmethod
    def from_file(cls, filename):
        """
        Load existing distance map using filename prefix
        (each distance map consist of .csv and .npy file)

        Parameters
        ----------
        filename : str
            Prefix of path to distance map files
            (excluding .csv/.npy)

        Returns
        -------
        DistanceMap
            Loaded distance map
        """
        return cls.from_files(
            filename + ".csv", filename + ".npy"
        )

    @classmethod
    def from_files(cls, residue_table_file, distance_matrix_file):
        """
        Load existing distance map with explicit
        paths to residue table (.csv) and distance
        matrix (.npy). Use DistanceMap.from_file
        to load using joint prefix of both files.

        Parameters
        ----------
        residue_table_file : str or file-like object
            Path to residue table file
            (prefix + .csv)
        distance_matrix_file : str or file-like object
            Path to distance matrix file
            (prefix + .npy)

        Returns
        -------
        DistanceMap
            Loaded distance map
        """
        residues = pd.read_csv(
            residue_table_file,
            index_col=0,
            dtype={
                "id": str,
                "seqres_id": str,
                "coord_id": str,
                "chain_index": int,
            }
        )

        dist_matrix = np.load(distance_matrix_file)

        if "axis" in residues.columns:
            symmetric = False
            residues_i = residues.query("axis == 'i'").drop("axis", axis=1)
            residues_j = residues.query("axis == 'j'").drop("axis", axis=1)
        else:
            symmetric = True
            residues_i = residues
            residues_j = residues

        return cls(
            residues_i, residues_j, dist_matrix, symmetric
        )

    def to_file(self, filename):
        """
        Store distance map in files

        Parameters
        ----------
        filename : str
            Prefix of distance map files
            (will create .csv and .npy file)

        Returns
        -------
        residue_table_filename : str
            Path to residue table (will be filename + .csv)
        dist_mat_filename : str
            Path to distance matrix file in numpy format
            (will be filename + .npy)
        """
        def _add_axis(df, axis):
            return df.assign(
                axis=axis
            )

        if self.symmetric:
            residues = self.residues_i
        else:
            res_i = _add_axis(self.residues_i, "i")
            res_j = _add_axis(self.residues_j, "j")
            residues = pd.concat([res_i, res_j])

        # save residue table
        residue_table_filename = filename + ".csv"
        residues.to_csv(residue_table_filename, index=True)

        # save distance matrix
        dist_mat_filename = filename + ".npy"
        np.save(dist_mat_filename, self.dist_matrix)

        return residue_table_filename, dist_mat_filename

    def dist(self, i, j, raise_na=True):
        """
        Return distance of residue pair

        Parameters
        ----------
        i : int or str
            Identifier of position on first axis
        j : int or str
            Identifier of position on second axis
        raise_na : bool, optional (default: True)
            Raise error if i or j is not
            contained in either axis. If False,
            returns np.nan for undefined entries.

        Returns
        -------
        np.float
            Distance of pair (i, j). If raise_na
            is False and identifiers are not valid,
            distance will be np.nan

        Raises
        ------
        KeyError
            If index i or j is not a valid identifier
            for respective chain
        """
        # internally all identifiers are handled
        # as strings, so convert
        i, j = str(i), str(j)

        # check if identifiers are valid for either axis
        if i not in self.id_map_i:
            if raise_na:
                raise KeyError(
                    "{} not contained in first axis of "
                    "distance map".format(i)
                )
            else:
                return np.nan

        if j not in self.id_map_j:
            if raise_na:
                raise KeyError(
                    "{} not contained in second axis of "
                    "distance map".format(j)
                )
            else:
                return np.nan

        # if valid, return distance of pair
        return self.dist_matrix[
            self.id_map_i[i],
            self.id_map_j[j]
        ]

    def __getitem__(self, identifiers):
        """
        Parameters
        ----------
        identifiers : tuple(str, str) or tuple(int, int)
            Identifiers of residues on first and
            second chain

        Raises
        -------
        KeyError
            If either residue identifier not valid
        """
        i, j = identifiers
        return self.dist(i, j, raise_na=True)

    def contacts(self, max_dist=5.0, min_dist=None):
        """
        Return list of pairs below distance threshold

        Parameters
        ----------
        max_dist : float, optional (default: 5.0)
            Maximum distance for any pair to be
            considered a contact
        min_dist : float, optional (default: None)
            Minimum distance of any pair to be
            returned (may be useful if extracting
            different distance ranges from matrix).
            Distance has to be > min_dist, (not >=).

        Returns
        -------
        contacts : pandas.DataFrame
            Table with residue-residue contacts, with the
            following columns:

            1. id_i: identifier of residue in chain i
            2. id_j: identifier of residue in chain j
            3. dist: pair distance
        """
        # find which entries of matrix fulfill
        # distance criteria
        if min_dist is None:
            cond = np.where(self.dist_matrix <= max_dist)
        else:
            cond = np.where(
                (self.dist_matrix <= max_dist) &
                (self.dist_matrix > min_dist)
            )

        i_all, j_all = cond

        # exclude diagonal entries of matrix since
        # they always have distance 0
        nodiag = i_all != j_all
        i = i_all[nodiag]
        j = j_all[nodiag]

        # extract residue ids and distances for all contacts
        contacts = pd.DataFrame()
        contacts.loc[:, "i"] = self.residues_i.id.values[i]
        contacts.loc[:, "j"] = self.residues_j.id.values[j]
        contacts.loc[:, "dist"] = self.dist_matrix[i, j]

        return contacts

    def transpose(self):
        """
        Transpose distance map (i.e. swap axes)

        Returns
        -------
        DistanceMap
            Transposed copy of distance map
        """
        return DistanceMap(
            self.residues_j, self.residues_i, self.dist_matrix.T, self.symmetric
        )

    @classmethod
    def aggregate(cls, *matrices, intersect=False, agg_func=np.nanmin):
        """
        Aggregate with other distance map(s). Secondary structure will
        be aggregated by assigning the most frequent state across
        all distance matrices; if there are equal counts, H (helix) will
        be chosen over E (strand) over C (coil).

        Parameters
        ----------
        ``*matrices`` : DistanceMap
            ``*args-style`` list of DistanceMaps that
            will be aggregated.

            .. note::

                The id column of each axis may only
                contain numeric residue ids (and no characters
                such as insertion codes)

        intersect : bool, optional (default: False)
            If True, intersect indices of the given
            distance maps. Otherwise, union of indices
            will be used.
        agg_func : function (default: numpy.nanmin)
            Function that will be used to aggregate
            distance matrices. Needs to take a
            parameter "axis" to aggregate over
            all matrices.

        Returns
        -------
        DistanceMap
            Aggregated distance map

        Raises
        ------
        ValueError
            If residue identifiers are not numeric, or
            if intersect is True, but positions on
            axis do not overlap.
        """
        def _sse_count(secstruct_elements):
            # obtain counts for each secondary structure element;
            # do not count nan entries
            counts = Counter(secstruct_elements.dropna())

            # sort items by count (first) and secondary structure (second);
            # this way, most frequent element at the end of list, and
            # prioritizing H over E over C
            sorted_sse = sorted(counts.items(), key=itemgetter(1, 0))

            # if no elements, make nan entry
            if len(sorted_sse) == 0:
                return np.nan
            else:
                return sorted_sse[-1][0]

        def _merge_axis(axis):
            # extract residue dataframes along axis
            # for all given distance maps
            dm = [
                getattr(m, axis) for m in matrices
            ]

            # create set of residue identifiers along axis
            # for each distance map. Note that identifiers
            # have to be numeric for easy sorting, so
            # cast to int first
            try:
                ids = [
                    pd.to_numeric(m.id).astype(int)
                    for m in dm
                ]
            except ValueError as e:
                raise ValueError(
                    "Residue indices must be all numeric for aggregate function (no insertion codes allowed)"
                ) from e

            # turn series into sets
            id_sets = [set(id_list) for id_list in ids]

            # then create final set of identifiers along axis
            # either as union or intersection
            if intersect:
                new_ids = set.intersection(*id_sets)
                if len(new_ids) == 0:
                    raise ValueError(
                        "Intersection of positions on axis "
                        "is empty, try intersect=False instead "
                        "or remove non-overlapping DistanceMap(s)."
                    )
            else:
                new_ids = set.union(*id_sets)

            # create new axis index object for final distance map
            new_axis_df = pd.DataFrame(sorted(new_ids), columns=["id"])

            # create mapping from one distance matrix into the other
            # by aligning indices from 0 in either matrix through
            # join on their residue id
            new_axis_map = new_axis_df.reset_index()
            mappings = [
                new_axis_map.merge(
                    id_list.to_frame("id").reset_index(drop=True).reset_index(),
                    on="id", how="inner",
                    suffixes=("_agg", "_src")
                ) for id_list in ids
            ]

            # turn residue ids back into strings
            new_axis_df.loc[:, "id"] = new_axis_df.loc[:, "id"].astype(str)

            # aggregate all residue dataframes into one joint table;
            # first, add prefix to all column names based on index or id
            # of respective distance map

            # try to use identifier if defined, otherwise just use index in list;
            # make sure the separator string is not contained in the identifier, otherwise replace it
            dm_ids = [
                (str(m.id).replace(cls._id_separator,  "") if m.id is not None else i)
                for i, m in enumerate(matrices)
            ]

            # move residue identifier to index, and prefix all other column names.
            # Note we could use a pd.MultiIndex here instead of the separator approach, but that would mean
            # changing a lot of other code...
            # Note: do not rename columns that already have the separator in case this is because of iterative merging

            # get rid of previous aggregated secondary structure or it will get re-merged in iterative merging setups
            dm_dropped = [
                m.drop(
                    ["sec_struct_3state"], axis=1
                ) if len(m.filter( regex=cls._id_separator + "sec_struct_3state").columns) >= 1 else m
                for m in dm
            ]

            dm_prefixed = [
                m.set_index("id").rename(
                    columns={
                        c: "{}{}{}".format(id_, cls._id_separator, c)
                        for c in m.columns
                        if cls._id_separator not in c
                    }
                ) for id_, m in zip(dm_ids, dm_dropped)
            ]

            # second: join together
            individual_dm_merged = pd.concat(
                dm_prefixed, axis=1, join="inner" if intersect else "outer", sort=True
            )
            # loses index name for some reason due to pd.concat
            individual_dm_merged.index.name = "id"

            # merge secondary structure assignments by identifying
            # most frequent assignment. If there are equal counts,
            # prefer H over E over C.
            # By starting from individual_dm_merged here, will always
            # merge over all individual distance maps, even if performing iterative aggregation
            secstruct_columns = individual_dm_merged.filter(
                regex=cls._id_separator + "sec_struct_3state"
            )

            if len(secstruct_columns.columns) >= 1:
                # identify most frequent state per position
                merged_sse_assignment = secstruct_columns.apply(
                    _sse_count, axis=1
                ).to_frame(
                    "sec_struct_3state"
                )

                # merge to new axis table
                new_axis_df = new_axis_df.merge(
                    merged_sse_assignment.reset_index(), on="id", how="left"
                )

            new_axis_df = new_axis_df.merge(
                individual_dm_merged.reset_index(), on="id", how="left"
            )

            return new_axis_df, mappings

        # make sure all matrices are either symmetric or non-symmetric
        symmetries = np.array([m.symmetric for m in matrices])
        if not np.all(symmetries[0] == symmetries):
            raise ValueError(
                "DistanceMaps are mixed symmetric/non-symmetric."
            )

        # create new axes of distance map
        new_res_i, maps_i = _merge_axis("residues_i")
        new_res_j, maps_j = _merge_axis("residues_j")

        # this is used to collected distance matrices for
        # later aggregtation
        new_mat = np.full(
            (len(matrices), len(new_res_i), len(new_res_j)),
            np.nan
        )

        # put individual matrices into new indexing system
        # of merged distance map
        for k, m in enumerate(matrices):
            i_src, j_src = np.meshgrid(
                maps_i[k].index_src.values,
                maps_j[k].index_src.values,
                indexing="ij"
            )

            i_agg, j_agg = np.meshgrid(
                maps_i[k].index_agg.values,
                maps_j[k].index_agg.values,
                indexing="ij"
            )

            # check we have valid indices (otherwise
            # will crash on empty distance matrices)
            if (len(i_agg) == 0 or len(j_agg) == 0 or
                    len(i_src) == 0 or len(j_src) == 0):
                raise ValueError(
                    "Trying to aggregate distance matrices on empty set of positions."
                )

            new_mat[k][i_agg, j_agg] = m.dist_matrix[i_src, j_src]

        # aggregate
        agg_mat = agg_func(new_mat, axis=0)

        return DistanceMap(
            new_res_i, new_res_j, agg_mat, symmetries[0]
        )

    def structure_coverage(self):
        """
        Find covered residue segments for individual structures
        that this distance map was computed from (either
        directly from structure or through aggregation of
        multiple structures). Only works if all residue identifiers
        of DistanceMap are numeric (i.e., do not have insertion codes)

        Returns
        -------
        coverage : list of tuple
            Returns tuples of the form
            (coverage_i, coverage_j, coverage_id),
            where
            * coverage_i and coverage_j are lists of tuples
              (segment_start, segment_end) of residue coverage
              along axis i and j, with segment_end
              being included in the range
            * coverage_id is the identifier of the individual
              substructure the coverage segments belong to
              (only set if an aggregated structure, None otherwise)
        """
        def _get_coverage_for_axis(axis):
            """
            Determine structural coverage by individual structure for an axis i/j
            """
            # proxy column for determining structure coverage
            coverage_col_name = "coord_id"

            residue_map = getattr(
                self, "residues_" + axis
            )

            # create numeric index from residue IDs,
            # fail if insertion codes are present
            try:
                residue_map = residue_map.assign(
                    id=pd.to_numeric(residue_map.id)
                )
            except ValueError as e:
                raise ValueError(
                    "Residue indices must be all numeric for aggregate function (no insertion codes allowed)"
                ) from e

            residue_map = residue_map.set_index("id")

            if coverage_col_name in residue_map:
                coverage_cols = residue_map[[coverage_col_name]]
            else:
                coverage_cols = residue_map.filter(
                    regex=self._id_separator + coverage_col_name
                )

            def _get_col_name(col_name):
                """
                Get clean identifier name from dataframe columns
                """
                # extract structure identifier (None if not an aggregated structure)
                if col_name == coverage_col_name:
                    return self.id
                else:
                    return col_name.split(self._id_separator)[0]

            # extract coverage segments for all individual structures
            segments = {
                _get_col_name(col_name): find_segments(series.dropna().sort_index().index)
                for col_name, series in coverage_cols.items()
            }

            return segments

        coverage_i = _get_coverage_for_axis("i")
        coverage_j = _get_coverage_for_axis("j")

        # should be the same in both cases, but to be on safe side if
        # users tinker with dataframes
        joint_keys = {
            k for k in coverage_i if k in coverage_j
        }

        coverage = [
            (coverage_i[k], coverage_j[k], k) for k in joint_keys
        ]

        return coverage


def _prepare_structures(structures, pdb_id_list, raise_missing=True):
    """
    Get structures ready for distance calculation

    Parameters
    ----------
    structures : str or dict
        See intra_dists function for explanation
    pdb_id_list:
        List of PDB entries to load
    raise_missing : bool, optional (default: True)
        Raise a ResourceError if any of the input structures can
        not be loaded; otherwise, ignore missing entries.

    Returns
    -------
    dict
        dictionary with lower-case PDB ids as keys
        and PDB objects as value
    """
    # load structures if not yet done so
    if structures is None or isinstance(structures, str):
        structures = load_structures(
            pdb_id_list, structures, raise_missing
        )

    return structures


def _prepare_chain(structures, pdb_id, pdb_chain,
                   atom_filter, mapping, model=0):
    """
    Prepare PDB chain for distance calculation

    Parameters
    ----------
    structures : dict
        Dictionary containing loaded PDB objects
    pdb_id : str
        ID of structure to extract chain from
    pdb_chain: str
        Chain ID to extract
    atom_filter : str
        Filter for this type of atom. Set to None
        if no filtering should be applied
    mapping : dict
        Seqres to Uniprot mapping that will be applied
        to Chain object
    model : int, optional (default: 0)
        Use this model from PDB structure

    Returns
    -------
    Chain
        Chain prepared for distance calculation
    """
    # get chain from structure
    chain = structures[pdb_id].get_chain(pdb_chain, model)

    # filter atoms if option selected
    if atom_filter is not None:
        chain = chain.filter_atoms(atom_filter)

    # remap chain to Uniprot
    chain = chain.remap(mapping)

    return chain


def intra_dists(sifts_result, structures=None, atom_filter=None,
                intersect=False, output_prefix=None, model=0,
                raise_missing=True):
    """
    Compute intra-chain distances in PDB files.

    Parameters
    ----------
    sifts_result : SIFTSResult
        Input structures and mapping to use
        for distance map calculation
    structures : str or dict, optional (default: None)
        If str: Load structures from directory this string
        points to. Missing structures will be fetched
        from web.

        If dict: dictionary with lower-case PDB ids as keys
        and PDB objects as values. This dictionary has to
        contain all necessary structures, missing ones will
        not be fetched. This dictionary can be created using
        pdb.load_structures.
    atom_filter : str, optional (default: None)
        Filter coordinates to contain only these atoms. E.g.
        set to "CA" to compute C_alpha - C_alpha distances
        instead of minimum atom distance over all atoms in
        both residues.
    intersect : bool, optional (default: False)
        If True, intersect indices of the given
        distance maps. Otherwise, union of indices
        will be used.
    output_prefix : str, optional (default: None)
        If given, save individual contact maps
        to files prefixed with this string. The appended
        file suffixes map to row index in sifts_results.hits
    model : int, optional (default: 0)
        Index of model in PDB structure that should be used
    raise_missing : bool, optional (default: True)
        Raise a ResourceError if any of the input structures can
        not be loaded; otherwise, ignore missing entries.

    Returns
    -------
    agg_distmap : DistanceMap
        Computed aggregated distance map
        across all input structures

        Contains an additional attribute aggregated_residue_maps,
        a pd.DataFrame with the concatenated residue maps of all individual
        chains used to compute this DistanceMap. Individual chains
        are linked to the input sifts_results through the column
        sifts_table_index.

        If output_prefix is given, agg_distmap will have an
        additional attribute individual_distance_map_table:
        pd.DataFrame with all individual distance maps that
        went into the aggregated distance map, with
        columns "sifts_table_index" (linking to SIFTS hit table) and
        "residue_table" and "distance_matrix"
        (file names of .csv and .npy files constituting
        the respective distance map).
        Will be None if output_prefix is None.

    Raises
    ------
    ValueError
        If sifts_result is empty (no structure hits)
    ResourceError
        If any structure could not be loaded and raise_missing is True
    """
    if len(sifts_result.hits) == 0:
        raise ValueError(
            "sifts_result is empty (no structure hits, but at least one required)"
        )

    # if no structures given, or path to files, load first
    structures = _prepare_structures(
        structures, sifts_result.hits.pdb_id, raise_missing
    )

    # aggegrated distance map
    agg_distmap = None

    # create output folder if necessary
    if output_prefix is not None:
        create_prefix_folders(output_prefix)

    # collect information about individual distance maps here (only if output_prefix is defined)
    individual_distance_map_info = []

    # collect information about residue map from target sequence to structure
    individual_residue_maps = []

    # compute individual distance maps and aggregate
    for i, r in sifts_result.hits.iterrows():
        # skip missing structures
        if not raise_missing and r["pdb_id"] not in structures:
            continue

        # extract and remap PDB chain
        chain = _prepare_chain(
            structures, r["pdb_id"], r["pdb_chain"],
            atom_filter, sifts_result.mapping[r["mapping_index"]],
            model
        )

        # skip empty chains
        if len(chain.residues) == 0:
            continue

        # compute distance map and set id
        distmap = DistanceMap.from_coords(chain)
        distmap.id = i

        # store information about residues for each individual aggregated distance map
        # (only for axis i since distmap is symmetric)
        individual_residue_maps.append(
            distmap.residues_i.assign(
                sifts_table_index=i
            )
        )

        # save individual distance map
        if output_prefix is not None:
            residue_table_filename, dist_mat_filename = distmap.to_file(
                "{}_{}".format(output_prefix, i)
            )

            individual_distance_map_info.append({
                "sifts_table_index": i,
                "residue_table": residue_table_filename,
                "distance_matrix": dist_mat_filename
            })

        # aggregate
        if agg_distmap is None:
            agg_distmap = distmap
        else:
            agg_distmap = DistanceMap.aggregate(
                agg_distmap, distmap, intersect=intersect
            )

    if agg_distmap is not None:
        if len(individual_distance_map_info) > 0:
            agg_distmap.individual_distance_map_table = pd.DataFrame(
                individual_distance_map_info
            )
        else:
            agg_distmap.individual_distance_map_table = None

        # aggregate residue maps into joint dataframe and attach to distance map
        agg_distmap.aggregated_residue_maps = pd.concat(
            individual_residue_maps
        ).reset_index(drop=True)

    return agg_distmap


def multimer_dists(sifts_result, structures=None, atom_filter=None,
                   intersect=False, output_prefix=None, model=0,
                   raise_missing=True):
    """
    Compute homomultimer distances (between repeated copies of the
    same entity) in PDB file. Resulting distance matrix will be
    symmetric by minimization over upper and lower triangle of matrix,
    even if the complex structure is not symmetric.

    Parameters
    ----------
    sifts_result : SIFTSResult
        Input structures and mapping to use
        for distance map calculation
    structures : str or dict, optional (default: None)
        If str: Load structures from directory this string
        points to. Missing structures will be fetched
        from web.

        If dict: dictionary with lower-case PDB ids as keys
        and PDB objects as values. This dictionary has to
        contain all necessary structures, missing ones will
        not be fetched. This dictionary can be created using
        pdb.load_structures.
    atom_filter : str, optional (default: None)
        Filter coordinates to contain only these atoms. E.g.
        set to "CA" to compute C_alpha - C_alpha distances
        instead of minimum atom distance over all atoms in
        both residues.
    intersect : bool, optional (default: False)
        If True, intersect indices of the given
        distance maps. Otherwise, union of indices
        will be used.
    output_prefix : str, optional (default: None)
        If given, save individual contact maps
        to files prefixed with this string. The appended
        file suffixes map to row index in sifts_results.hits
    model : int, optional (default: 0)
        Index of model in PDB structure that should be used
    raise_missing : bool, optional (default: True)
        Raise a ResourceError if any of the input structures can
        not be loaded; otherwise, ignore missing entries.

    Returns
    -------
    agg_distmap : DistanceMap
        Computed aggregated distance map
        across all input structures

        If output_prefix is given, agg_distmap will have an
        additional attribute individual_distance_map_table:
        pd.DataFrame with all individual distance maps that
        went into the aggregated distance map, with
        columns "sifts_table_index_i", "sifts_table_index_j"
        (linking to SIFTS hit table) and
        "residue_table" and "distance_matrix"
        (file names of .csv and .npy files constituting
        the respective distance map).

        Will be None if output_prefix is None.

    Raises
    ------
    ValueError
        If sifts_result is empty (no structure hits)
    ResourceError
        If any structure could not be loaded and raise_missing is True
    """
    if len(sifts_result.hits) == 0:
        raise ValueError(
            "sifts_result is empty (no structure hits, but at least one required)"
        )

    # if no structures given, or path to files, load first
    structures = _prepare_structures(
        structures, sifts_result.hits.pdb_id, raise_missing
    )

    # aggegrated distance map
    agg_distmap = None

    # create output folder if necessary
    if output_prefix is not None:
        create_prefix_folders(output_prefix)

    # collect information about individual distance maps here (only if output_prefix is defined)
    individual_distance_maps = []

    # go through each structure
    for pdb_id, grp in sifts_result.hits.reset_index().groupby("pdb_id"):
        # skip missing structures
        if not raise_missing and pdb_id not in structures:
            continue

        # extract all chains for this structure
        chains = [
            (
                r["index"],
                _prepare_chain(
                    structures, r["pdb_id"], r["pdb_chain"],
                    atom_filter, sifts_result.mapping[r["mapping_index"]],
                    model
                )
            )
            for i, r in grp.iterrows()
        ]

        # compare all possible pairs of chains
        for (index_i, ch_i), (index_j, ch_j) in combinations(chains, 2):
            # skip empty chains (e.g. residues lost during remapping)
            if len(ch_i.residues) == 0 or len(ch_j.residues) == 0:
                continue

            distmap = DistanceMap.from_coords(ch_i, ch_j)

            # set distance map id
            distmap.id = "{}_{}".format(index_i, index_j)

            # symmetrize matrix (for ECs we are only interested if a pair
            # is close in some combination)
            distmap_transposed = distmap.transpose()
            distmap_transposed.id = distmap.id + "_T"

            distmap_sym = DistanceMap.aggregate(
                distmap, distmap_transposed, intersect=intersect
            )
            distmap_sym.symmetric = True

            # save individual distance map
            if output_prefix is not None:
                residue_table_filename, dist_mat_filename = distmap_sym.to_file("{}_{}_{}".format(
                    output_prefix, index_i, index_j)
                )

                individual_distance_maps.append({
                    "sifts_table_index_i": index_i,
                    "sifts_table_index_j": index_j,
                    "residue_table": residue_table_filename,
                    "distance_matrix": dist_mat_filename
                })

            # aggregate with other chain combinations
            if agg_distmap is None:
                agg_distmap = distmap_sym
            else:
                agg_distmap = DistanceMap.aggregate(
                    agg_distmap, distmap_sym, intersect=intersect
                )

    if agg_distmap is not None:
        if len(individual_distance_maps) > 0:
            agg_distmap.individual_distance_map_table = pd.DataFrame(
                individual_distance_maps
            )
        else:
            agg_distmap.individual_distance_map_table = None

    return agg_distmap


def inter_dists(sifts_result_i, sifts_result_j, structures=None,
                atom_filter=None, intersect=False, output_prefix=None,
                model=0, raise_missing=True):
    """
    Compute inter-chain distances (between different entities)
    in PDB file. Resulting distance map is typically not
    symmetric, with either axis corresponding to either chain.
    Inter-distances are calculated on all combinations of chains
    that have the same PDB id in sifts_result_i and sifts_result_j.

    Parameters
    ----------
    sifts_result_i : SIFTSResult
        Input structures and mapping to use
        for first axis of computed distance map
    sifts_result_j : SIFTSResult
        Input structures and mapping to use
        for second axis of computed distance map
    structures : str or dict, optional (default: None)

        * If str: Load structures from directory this string
          points to. Missing structures will be fetched
          from web.

        * If dict: dictionary with lower-case PDB ids as keys
          and PDB objects as values. This dictionary has to
          contain all necessary structures, missing ones will
          not be fetched. This dictionary can be created using
          pdb.load_structures.

    atom_filter : str, optional (default: None)
        Filter coordinates to contain only these atoms. E.g.
        set to "CA" to compute C_alpha - C_alpha distances
        instead of minimum atom distance over all atoms in
        both residues.
    intersect : bool, optional (default: False)
        If True, intersect indices of the given
        distance maps. Otherwise, union of indices
        will be used.
    output_prefix : str, optional (default: None)
        If given, save individual contact maps
        to files prefixed with this string. The appended
        file suffixes map to row index in sifts_results.hits
    model : int, optional (default: 0)
        Index of model in PDB structure that should be used
    raise_missing : bool, optional (default: True)
        Raise a ResourceError if any of the input structures can
        not be loaded; otherwise, ignore missing entries.

    Returns
    -------
    agg_distmap : DistanceMap
        Computed aggregated distance map
        across all input structures

        If output_prefix is given, agg_distmap will have an
        additional attribute individual_distance_map_table:

        pd.DataFrame with all individual distance maps that
        went into the aggregated distance map, with
        columns "sifts_table_index_i", "sifts_table_index_j"
        (linking to SIFTS hit table) and
        "residue_table" and "distance_matrix"
        (file names of .csv and .npy files constituting
        the respective distance map).

        Will be None if output_prefix is None.

    Raises
    ------
    ValueError
        If sifts_result_i or sifts_result_j is empty
        (no structure hits)
    ResourceError
        If any structure could not be loaded and raise_missing is True
    """
    def _get_chains(sifts_result):
        return {
            i: _prepare_chain(
                structures, r["pdb_id"], r["pdb_chain"],
                atom_filter, sifts_result.mapping[r["mapping_index"]],
                model
            )
            for i, r in sifts_result.hits.iterrows()
            if raise_missing or r["pdb_id"] in structures
        }

    if len(sifts_result_i.hits) == 0 or len(sifts_result_j.hits) == 0:
        raise ValueError(
            "sifts_result_i or sifts_result_j is empty "
            "(no structure hits, but at least one required)"
        )

    # if no structures given, or path to files, load first
    structures = _prepare_structures(
        structures,
        set(sifts_result_i.hits.pdb_id) | set(sifts_result_j.hits.pdb_id),
        raise_missing
    )

    # aggegrated distance map
    agg_distmap = None

    # create output folder if necessary
    if output_prefix is not None:
        create_prefix_folders(output_prefix)

    # collect information about individual distance maps here (only if output_prefix is defined)
    individual_distance_maps = []

    # determine which combinations of chains to look at
    # (anything that has same PDB identifier)
    combis = sifts_result_i.hits.reset_index().merge(
        sifts_result_j.hits.reset_index(),
        on="pdb_id", suffixes=("_i", "_j")
    )

    # extract chains for each subunit
    chains_i = _get_chains(sifts_result_i)
    chains_j = _get_chains(sifts_result_j)

    # go through all chain combinations
    for i, r in combis.iterrows():
        # skip missing structures
        if not raise_missing and r["pdb_id"] not in structures:
            continue

        index_i = r["index_i"]
        index_j = r["index_j"]

        # skip empty chains
        if (len(chains_i[index_i].residues) == 0 or
                len(chains_j[index_j].residues) == 0):
            continue

        # compute distance map for current chain pair
        distmap = DistanceMap.from_coords(
            chains_i[index_i],
            chains_j[index_j],
        )
        # set distance map id
        distmap.id = "{}_{}".format(index_i, index_j)

        # save individual distance map
        if output_prefix is not None:
            residue_table_filename, dist_mat_filename = distmap.to_file("{}_{}_{}".format(
                output_prefix, index_i, index_j)
            )

            individual_distance_maps.append({
                "sifts_table_index_i": index_i,
                "sifts_table_index_j": index_j,
                "residue_table": residue_table_filename,
                "distance_matrix": dist_mat_filename
            })

        # aggregate with other chain combinations
        if agg_distmap is None:
            agg_distmap = distmap
        else:
            agg_distmap = DistanceMap.aggregate(
                agg_distmap, distmap, intersect=intersect
            )

    if agg_distmap is not None:
        if len(individual_distance_maps) > 0:
            agg_distmap.individual_distance_map_table = pd.DataFrame(
                individual_distance_maps
            )
        else:
            agg_distmap.individual_distance_map_table = None

    return agg_distmap


def _remap_sequence(chain, sequence):
    """
    Changes the residue names in an input
    PDB chain to the given sequence (both one 
    letter and three letter codes).
    
    Parameters
    ----------
    chain : Chain
        PDB chain that will be remapped
    sequence :  dict
        Mapping from sequence position (int or str) to residue.
        Residues in the output structures will be renamed to the
        residues in this mapping (without any changes
        what the residues actually are in the structure in terms
        of atoms)
    
    Returns
    -------
    Chain
        PDB chain with updated sequence
    """
    chain = deepcopy(chain)
    # change one letter code
    chain.residues.loc[
        :, "one_letter_code"
    ] = chain.residues.id.map(sequence)

    # and three letter code
    chain.residues.loc[
        :, "three_letter_code"
    ] = chain.residues.one_letter_code.map(AA1_to_AA3)

    # drop anything we could not map
    chain.residues = chain.residues.dropna(
        subset=["one_letter_code", "three_letter_code"]
    )

    return chain


def remap_chains(sifts_result, output_prefix, sequence=None,
                 structures=None, atom_filter=("N", "CA", "C", "O"),
                 model=0, chain_name="A", raise_missing=True):
    """
    Remap a set of PDB chains into the numbering scheme (and
    amino acid sequence) of a target sequence (a.k.a. the poorest
    homology model possible).
    
    (This function is placed here because of close relationship
    to intra_dists and reusing functionality for it).
    
    Parameters
    ----------
    sifts_result : SIFTSResult
        Input structures and mapping to use
        for remapping
    output_prefix : str
        Save remapped structures to files prefixed with this string
    sequence : dict, optional (default: None)
        Mapping from sequence position (int or str) to residue.
        If this parameter is given, residues in the output 
        structures will be renamed to the residues in this
        mapping.

        .. note::

            if side-chain residues are not taken off using atom_filter, this will e.g. happily label
            an actual glutamate as an alanine).

    structures : str or dict, optional (default: None)

        * If str: Load structures from directory this string
          points to. Missing structures will be fetched
          from web.

        * If dict: dictionary with lower-case PDB ids as keys
          and PDB objects as values. This dictionary has to
          contain all necessary structures, missing ones will
          not be fetched. This dictionary can be created using
          pdb.load_structures.

    atom_filter : str, optional (default: ("N", "CA", "C", "O"))
        Filter coordinates to contain only these atoms. If None,
        will retain all atoms; the default value will only keep
        backbone atoms.
    model : int, optional (default: 0)
        Index of model in PDB structure that should be used
    chain_name : str, optional (default: "A")
        Rename the PDB chain to this when saving the file. This
        will not affect the file name, only the name of the chain in 
        the PDB object.
    raise_missing : bool, optional (default: True)
        Raise a ResourceError if any of the input structures can
        not be loaded; otherwise, ignore missing entries.

    Returns
    ------
    remapped : dict
        Mapping from index of each structure hit in sifts_results.hits
        to filename of stored remapped structure
    """
    # if no structures given, or path to files, load first
    structures = _prepare_structures(
        structures, sifts_result.hits.pdb_id, raise_missing
    )

    # create output folder if necessary
    if output_prefix is not None:
        create_prefix_folders(output_prefix)

    # collect remapped chains
    remapped = {}

    # make sure keys in sequence map are strings,
    # since indices in structures are stored as strings
    if sequence is not None:
        sequence = {
            str(k): v for k, v in sequence.items()
        }

    # go through each structure
    for idx, r in sifts_result.hits.iterrows():
        # skip missing structures
        if not raise_missing and r["pdb_id"] not in structures:
            continue

        # extract and remap PDB chain
        chain = _prepare_chain(
            structures, r["pdb_id"], r["pdb_chain"],
            atom_filter, sifts_result.mapping[r["mapping_index"]],
            model
        )

        # if a map from sequence index to residue is given,
        # use it to rename the residues to the target sequence
        if sequence is not None:
            chain = _remap_sequence(chain, sequence)

        # save model coordinates to .pdb file - note that the
        # file name will contain the original chain name in
        # the source PDB file, while the chain itself will be
        # remapped according to chain_name parameter
        filename = "{}_{}_{}_{}.pdb".format(
            output_prefix,
            r["pdb_id"], r["pdb_chain"], r["mapping_index"]
        )

        # save to file
        with open(filename, "w") as f:
            chain.to_file(f, chain_id=chain_name, first_atom_id=1)

        # typecast index so it is regular python type, not numpy
        # (important for yaml dump)
        remapped[int(idx)] = filename

    return remapped


def remap_complex_chains(sifts_result_i, sifts_result_j,
                         sequence_i=None, sequence_j=None, structures=None,
                         atom_filter=("N", "CA", "C", "O"),
                         output_prefix=None, raise_missing=True, chain_name_i="A",
                         chain_name_j="B", model=0):
    """
    Remap a pair of PDB chains from the same structure
    into the numbering scheme (and amino acid sequence) of a
    target sequence.

    Parameters
    ----------
    sifts_result_i : SIFTSResult
        Input structures and mapping to use
        for remapping
    sifts_result_j : SIFTSResult
        Input structures and mapping to use
        for remapping
    output_prefix : str
        Save remapped structures to files prefixed with this string
    sequence_i : dict, optional (default: None)
        Mapping from sequence position (int or str) in the
        first sequence to residue.
        If this parameter is given, residues in the output 
        structures will be renamed to the residues in this
        mapping.

        .. note::

            if side-chain residues are not taken off using atom_filter, this will e.g. happily label
            an actual glutamate as an alanine).
    sequence_j : dict, optional (default: None)
        Same as sequence_j for second sequence.

    structures : str or dict, optional (default: None)

        * If str: Load structures from directory this string
          points to. Missing structures will be fetched
          from web.

        * If dict: dictionary with lower-case PDB ids as keys
          and PDB objects as values. This dictionary has to
          contain all necessary structures, missing ones will
          not be fetched. This dictionary can be created using
          pdb.load_structures.
    atom_filter : str, optional (default: ("N", "CA", "C", "O"))
        Filter coordinates to contain only these atoms. If None,
        will retain all atoms; the default value will only keep
        backbone atoms.
    model : int, optional (default: 0)
        Index of model in PDB structure that should be used
    raise_missing : bool, optional (default: True)
        Raise a ResourceError if any of the input structures can
        not be loaded; otherwise, ignore missing entries.
    chain_name_i : str, optional (default: "A")
        Renames the first chain to this string
    chain_name_j : str, optional (default: "B")
        Renames the second chain to this string

    Returns
    -------
    remapped : dict
        Mapping from index of each structure hit in sifts_results.hits
        to filename of stored remapped structure

    Raises
    ------
    ValueError
        If sifts_result_i or sifts_result_j is empty
        (no structure hits)
    ResourceError
        If any structure could not be loaded and raise_missing is True
    """

    if len(sifts_result_i.hits) == 0 or len(sifts_result_j.hits) == 0:
        raise ValueError(
            "sifts_result_i or sifts_result_j is empty "
            "(no structure hits, but at least one required)"
        )

    # make sure keys in sequence map are strings,
    # since indices in structures are stored as strings
    if sequence_i is not None:
        sequence_i = {
            str(k): v for k, v in sequence_i.items()
        }

    if sequence_j is not None:
        sequence_j = {
            str(k): v for k, v in sequence_j.items()
        }

    # create output folder if necessary
    if output_prefix is not None:
        create_prefix_folders(output_prefix)

    # determine which combinations of chains to look at
    # (anything that has same PDB identifier)
    combis = sifts_result_i.hits.reset_index().merge(
        sifts_result_j.hits.reset_index(),
        on="pdb_id", suffixes=("_i", "_j")
    )

    # if no structures given, or path to files, load first
    structures = _prepare_structures(
        structures, combis.pdb_id, raise_missing
    )

    remapped = {}

    # go through all chain combinations
    for i, r in combis.iterrows():

        # extract and remap PDB chain
        chain_i = _prepare_chain(
            structures, r["pdb_id"], r["pdb_chain_i"],
            atom_filter, sifts_result_i.mapping[r["mapping_index_i"]],
            model
        )

        # if a map from sequence index to residue is given,
        # use it to rename the residues to the target sequence
        if sequence_i is not None:
            # change one letter code
            chain_i = _remap_sequence(chain_i, sequence_i)

        # if the user specified a new chain name, change the chain name
        original_pdb_chain_i = r["pdb_chain_i"]

        # extract and remap PDB chain
        chain_j = _prepare_chain(
            structures, r["pdb_id"], r["pdb_chain_j"],
            atom_filter, sifts_result_j.mapping[r["mapping_index_j"]],
            model
        )

        # if a map from sequence index to residue is given,
        # use it to rename the residues to the target sequence
        if sequence_j is not None:
            # change one letter code
            chain_j = _remap_sequence(chain_j, sequence_j)

        # if the user specified a new chain name, change the chain name
        original_pdb_chain_j = r["pdb_chain_j"]

        # save model coordinates to .pdb file
        filename = "{}_{}_{}_{}_{}_{}.pdb".format(
            output_prefix,
            r["pdb_id"], original_pdb_chain_i,
            r["mapping_index_i"],
            original_pdb_chain_j,
            r["mapping_index_j"]
        )

        # save to file
        with open(filename, "w") as f:
            chain_i.to_file(
                f, chain_id=chain_name_i, first_atom_id=1, end=False
            )
            chain_j.to_file(
                f, chain_id=chain_name_j, first_atom_id=len(chain_i.coords) + 1
            )

        # typecast index so it is regular python type, not numpy
        # (important for yaml dump)
        remapped[int(i)] = filename

    return remapped
