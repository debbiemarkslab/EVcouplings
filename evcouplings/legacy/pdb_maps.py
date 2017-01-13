"""
Library to handle pdb3d results and interface with EVcouplings

(1) contact maps
(2) DSSP annotation

TODO:
- implement "classical" contact map features

Thomas A. Hopf, 20.02.2015
"""

from sys import stderr
from copy import deepcopy
from collections import defaultdict
import numpy

import evcouplings.legacy.pdb3d as pdb3d


class ContactMap:
    """
    Pairwise 3D distance map
    """
    def __init__(self, filename, uniprot_index_transform=None,
                 uniprot_ac_filter=None):
        """
        uniprot_index_transform: dictionary of functions to
        transform indices

        ac filter: dictionary of Uniprot ACs to be kept
        """
        df = pdb3d.read_results(filename, na_values=pdb3d._NA_VALUE)

        # identify all available selectors
        self.selectors = set(df["sel_i"])
        self.selectors.update(df["sel_j"])

        # filter out any part of contact map that does
        # not match target ACs for a given selector
        # (but not PDB residues that have not been matched
        # to UniProt residues)
        if uniprot_ac_filter is not None:
            for s, target_ac in uniprot_ac_filter.items():
                df = df.loc[((df.sel_i != s) | (df.up_ac_i.isnull()) |
                            (df.up_ac_i == target_ac)) &
                            ((df.sel_j != s) | (df.up_ac_j.isnull()) |
                            (df.up_ac_j == target_ac))]

        # make sure there is only one Uniprot AC per selector
        for s in self.selectors:
            acs = set(df.loc[df.sel_i == s, "up_ac_i"].dropna())
            acs.update(set(df.loc[df.sel_j == s, "up_ac_j"].dropna()))

            if len(acs) > 1:
                raise ValueError("Multiple Uniprot ACs for selector {}: {}"
                                 .format(s, acs))

        # apply Uniprot index transformations
        if uniprot_index_transform is not None:
            for (selector, transform) in uniprot_index_transform.items():
                if selector not in self.selectors:
                    print("Selector not in dataframe:", selector, file=stderr)
                    continue

                df.loc[df.sel_i == selector, "up_index_i"] = \
                    df.loc[df.sel_i == selector, "up_index_i"].map(transform)
                df.loc[df.sel_j == selector, "up_index_j"] = \
                    df.loc[df.sel_j == selector, "up_index_j"].map(transform)

        # create index mapping dictionaries
        self.uniprot_to_pdb_index = {}
        self.pdb_to_uniprot_index = {}
        for s in self.selectors:
            # we can't cast Uniprot indices to int before NaNs are removed
            index_list = df.loc[df.sel_i == s, ["up_index_i", "pdb_index_i"]].dropna()
            index_map = {
                int(u): p for u, p in zip(index_list.up_index_i, index_list.pdb_index_i)
            }

            index_list_rev = df.loc[df.sel_j == s, ["up_index_j", "pdb_index_j"]].dropna()
            index_map.update(
                {int(u): p for u, p in zip(index_list_rev.up_index_j, index_list_rev.pdb_index_j)}
            )

            self.uniprot_to_pdb_index[s] = index_map
            self.pdb_to_uniprot_index[s] = {p: u for (u, p) in index_map.items()}

        self.df = df

    def submatrix(self, selector_1, selector_2, s1_uniprot=True, s2_uniprot=True):
        """
        Get a distance matrix object for a pair of selectors.
        Use s1_uniprot and s2_uniprot to define if Uniprot or PDB
        indices are used to construct each axis of the distance matrix
        """
        def _pdb_sort(x):
            return (int(x[:-1]), x[-1]) if x[-1].isalpha() else int(x)

        pos_fwd = (self.df.sel_i == selector_1) & (self.df.sel_j == selector_2)
        pos_bwd = (self.df.sel_j == selector_1) & (self.df.sel_i == selector_2)
        pos_all = pos_fwd | pos_bwd

        # symmetric case: intra-contacts
        if selector_1 == selector_2:
            if s1_uniprot != s2_uniprot:
                raise ValueError("Cannot mix PDB and Uniprot numbering for intra-contacts.")

            if s1_uniprot:
                sub_df = self.df.loc[pos_all].dropna(subset=["up_index_i", "up_index_j"])
                pos_list = sorted(set(sub_df.up_index_i.astype(int)) | set(sub_df.up_index_j.astype(int)))
                index_col_i = sub_df.up_index_i.astype(int)
                index_col_j = sub_df.up_index_j.astype(int)
            else:
                sub_df = self.df.loc[pos_all].dropna(subset=["pdb_index_i", "pdb_index_j"])
                # sort by integer value, but take care of insertion codes (e.g. 87A)
                pos_list = sorted(
                    set(sub_df.pdb_index_i.astype(str)) | set(sub_df.pdb_index_j.astype(str)), key=_pdb_sort
                )
                index_col_i = sub_df.pdb_index_i.astype(str)
                index_col_j = sub_df.pdb_index_j.astype(str)

            pos_to_index = {p: i for i, p in enumerate(pos_list)}
            dist_matrix = numpy.empty((len(pos_list), len(pos_list)))
            dist_matrix.fill(numpy.NAN)

            for i, j, dist in zip(index_col_i, index_col_j, sub_df.dist):
                dist_matrix[pos_to_index[i], pos_to_index[j]] = dist
                dist_matrix[pos_to_index[j], pos_to_index[i]] = dist

            pos_list_1 = pos_list_2 = pos_list

        else:
            # non-symmetric case (inter-contacts)
            # usually, columns shouldn't be mixed between forward and backward,
            # but for robustness assume they might be
            s1_col, s1_type = ("up_index", int) if s1_uniprot else ("pdb_index", str)
            s2_col, s2_type = ("up_index", int) if s2_uniprot else ("pdb_index", str)

            sub_df_fwd = self.df.loc[pos_fwd].dropna(subset=[s1_col+"_i", s2_col+"_j"])
            sub_df_bwd = self.df.loc[pos_bwd].dropna(subset=[s2_col+"_i", s1_col+"_j"])

            if s1_uniprot:
                pos_list_1 = sorted(set(sub_df_fwd.up_index_i.astype(int)) | set(sub_df_bwd.up_index_j.astype(int)))
            else:
                pos_list_1 = sorted(
                    set(sub_df_fwd.pdb_index_i.astype(str)) | set(sub_df_bwd.pdb_index_j.astype(str)), key=_pdb_sort
                )
            pos_to_index_1 = {p: i for i, p in enumerate(pos_list_1)}

            if s2_uniprot:
                pos_list_2 = sorted(set(sub_df_fwd.up_index_j.astype(int)) | set(sub_df_bwd.up_index_i.astype(int)))
            else:
                pos_list_2 = sorted(
                    set(sub_df_fwd.pdb_index_j.astype(str)) | set(sub_df_bwd.pdb_index_i.astype(str)), key=_pdb_sort
                )
            pos_to_index_2 = {p: i for i, p in enumerate(pos_list_2)}

            dist_matrix = numpy.empty((len(pos_list_1), len(pos_list_2)))
            dist_matrix.fill(numpy.NAN)

            for i, j, dist in zip(
                    sub_df_fwd.loc[:, s1_col+"_i"].astype(s1_type),
                    sub_df_fwd.loc[:, s2_col+"_j"].astype(s2_type),
                    sub_df_fwd.dist
                    ):
                dist_matrix[pos_to_index_1[i], pos_to_index_2[j]] = dist

            for j, i, dist in zip(
                    sub_df_bwd.loc[:, s2_col+"_i"].astype(s2_type),
                    sub_df_bwd.loc[:, s1_col+"_j"].astype(s1_type),
                    sub_df_bwd.dist
                    ):
                dist_matrix[pos_to_index_1[i], pos_to_index_2[j]] = dist

        return DistanceMatrix(dist_matrix, pos_list_1, pos_list_2, selector_1, selector_2)

    def submatrix_aggregate(self, selectors_list, intersect=False,
                            aggregate_func=numpy.nanmin, **kwargs):
        """
        Get a folded distance matrix object for a list of
        selector pairs. Selector pairs have to be congruent
        (i.e. refer to equivalent chains in a multimer) or
        results will be nonsense.

        selectors_list example: [("A", "B"), ("P", "O"), ...]

        **kwargs will be passed on to submatrix() function
        """
        matrices = []
        for (s_1, s_2) in selectors_list:
            matrices.append(self.submatrix(s_1, s_2, **kwargs))

        return DistanceMatrix.aggregate(matrices, intersect, aggregate_func)

    def all_close_to_selector(self, selector_1, selector_2, dist_threshold):
        """
        Find all Uniprot residue objects in selector 1 closer
        than dist_threshold to any object in selector 2
        """
        resi = set(self.df.loc[(self.df.sel_i == selector_1) &
                               (self.df.sel_j == selector_2) &
                               (self.df.dist <= dist_threshold)
                               ].up_index_i.dropna().astype(int))

        resi.update(self.df.loc[(self.df.sel_j == selector_1) &
                                (self.df.sel_i == selector_2) &
                                (self.df.dist <= dist_threshold)
                                ].up_index_j.dropna().astype(int))

        return sorted(resi)

    def all_close_to_object(self, selector_1, selector_2, pdb_index, name,
                            dist_threshold):
        """
        Find all Uniprot residue objects in selector 1 closer
        than dist_threshold to PDB residue object in selector_2
        (e.g. heteroatom)

        pdb_index: string (will be cast to str otherwise)
        """

        resi = set(self.df.loc[(self.df.sel_i == selector_1) &
                               (self.df.sel_j == selector_2) &
                               (self.df.pdb_index_j.astype(str) == str(pdb_index)) &
                               (self.df.pdb_res_j == name) &
                               (self.df.dist <= dist_threshold)
                               ].up_index_i.dropna().astype(int))

        resi.update(self.df.loc[(self.df.sel_j == selector_1) &
                                (self.df.sel_i == selector_2) &
                                (self.df.pdb_index_i.astype(str) == str(pdb_index)) &
                                (self.df.pdb_res_i == name) &
                                (self.df.dist <= dist_threshold)
                                ].up_index_j.dropna().astype(int))

        return sorted(resi)


class DistanceMatrix:
    """
    Stores pairwise distances between two lists of objects
    """
    def __init__(self, dist_matrix, pos_list_1, pos_list_2,
                 sel_1=None, sel_2=None, raw_matrix=None):
        self.dist_matrix = dist_matrix
        self.pos_list_1 = pos_list_1
        self.pos_list_2 = pos_list_2
        self.sel_1 = sel_1
        self.sel_2 = sel_2
        self.raw_matrix = raw_matrix

        self.pos_to_index_1 = {p: i for i, p in enumerate(pos_list_1)}
        self.pos_to_index_2 = {p: i for i, p in enumerate(pos_list_2)}

        self.dist = defaultdict(lambda: defaultdict(lambda: float("nan")))
        for i in self.pos_list_1:
            for j in self.pos_list_2:
                self.dist[i][j] = self.dist_matrix[self.pos_to_index_1[i], self.pos_to_index_2[j]]

    @staticmethod
    def aggregate(matrices, intersect=False, aggregate_func=numpy.nanmin):
        def _pdb_sort(x):
            if isinstance(x, str):
                return (int(x[:-1]), x[-1]) if x[-1].isalpha() else int(x)
            else:
                return int(x)

        # compile list of positions along both axis
        # (either intersection or union)
        all_pos_1 = [set(mat.pos_list_1) for mat in matrices]
        all_pos_2 = [set(mat.pos_list_2) for mat in matrices]
        if intersect:
            pos_list_1 = set.intersection(*all_pos_1)
            pos_list_2 = set.intersection(*all_pos_2)
        else:
            pos_list_1 = set.union(*all_pos_1)
            pos_list_2 = set.union(*all_pos_2)

        sel_1 = [mat.sel_1 for mat in matrices]
        sel_2 = [mat.sel_2 for mat in matrices]

        pos_list_1 = sorted(pos_list_1, key=_pdb_sort)
        pos_list_2 = sorted(pos_list_2, key=_pdb_sort)

        # initialize and fill matrix
        dist_matrix = numpy.empty((len(pos_list_1), len(pos_list_2), len(matrices)))
        dist_matrix.fill(numpy.NAN)

        for m_i, mat in enumerate(matrices):
            for i, p1 in enumerate(pos_list_1):
                for j, p2 in enumerate(pos_list_2):
                    if p1 in mat.pos_to_index_1 and p2 in mat.pos_to_index_2:
                        dist_matrix[i, j, m_i] = mat.dist_matrix[
                                                 mat.pos_to_index_1[p1],
                                                 mat.pos_to_index_2[p2]]

        # apply aggregate function to combine values from different submatrices
        dist_matrix_agg = aggregate_func(dist_matrix, axis=2)

        return DistanceMatrix(
            dist_matrix_agg, pos_list_1, pos_list_2, sel_1, sel_2, raw_matrix=dist_matrix
        )

    def remap_indices(self, mapping_dict_1, mapping_dict_2=None):
        """
        Remap indices of DistanceMatrix object
        """
        # assume symmetric mapping if only first mapping is given
        if mapping_dict_2 is None:
            mapping_dict_2 = mapping_dict_1

        # make sure indices are all strings
        mapping_dict_1 = {str(k): str(v) for (k, v) in mapping_dict_1.items()}
        mapping_dict_2 = {str(k): str(v) for (k, v) in mapping_dict_2.items()}

        pos_list_1 = [mapping_dict_1.get(p, None) for p in self.pos_list_1]
        pos_list_2 = [mapping_dict_2.get(p, None) for p in self.pos_list_2]

        # create new distance matrix with transformed indices
        return DistanceMatrix(
            self.dist_matrix, pos_list_1, pos_list_2,
            self.sel_1, self.sel_2, self.raw_matrix
        )

    def create_old_contact_map(self):
        """
        Create contact_map.py-style ContactMap
        from distance matrix object.
        """
        from evcouplings.legacy.contact_map import ContactMap
        distance_dict = defaultdict(defaultdict)

        # convert into integer-indexed dictionary as expected by ContactMap class
        for pos_1 in self.dist:
            for pos_2 in self.dist[pos_1]:
                if pos_1 is not None and pos_2 is not None:
                    distance_dict[int(pos_1)][int(pos_2)] = self.dist[pos_1][pos_2]

        # TODO: initialize other fields of object

        return ContactMap(distance_dict=distance_dict)

    def write(self, out_file, sequence=None, selector="A", comments=None):
        """
        Save distance matrix as file in contact map format
        """
        if sequence is None:
            sequence = {}

        with open(out_file, "w") as f:
            if comments is not None:
                for c in comments:
                    print("# " + c, file=f)

            print(
                " ".join(
                    ["sel_i", "up_index_i", "res_i", "up_ac_i",
                     "pdb_index_i", "pdb_res_i", "is_res_i",
                     "sel_j", "up_index_j", "res_j", "up_ac_j",
                     "pdb_index_j", "pdb_res_j", "is_res_j",
                     "dist"]
                ), file=f
            )

            for p1 in self.pos_list_1:
                for p2 in self.pos_list_2:
                    if p1 < p2:
                        print(
                            selector, p1, sequence.get(p1, pdb3d._NA_VALUE), pdb3d._NA_VALUE,
                            pdb3d._NA_VALUE, pdb3d._NA_VALUE, "1",
                            selector, p2, sequence.get(p2, pdb3d._NA_VALUE), pdb3d._NA_VALUE,
                            pdb3d._NA_VALUE, pdb3d._NA_VALUE, "1",
                            self.dist[p1][p2],
                            file=f
                        )


class DSSPMap:
    """
    DSSP annotation map
    """
    _NUMERIC_FEATURES = ["ss", "acc", "rel_acc", "phi", "psi"]

    # reduction to 3-state secondary structure annotation
    _REDUCTION = {"H": "H", "G": "H", "I": "H",
                  "E": "E", "B": "E",
                  "-": "-", "T": "-", "S": "-"}

    def __init__(self, filename, uniprot_index_transform=None,
                 uniprot_ac_filter=None):
        df = pdb3d.read_results(filename, {"uniprot_ac": "-"})
        self.chains = set(df["pdb_chain"])

        # filter out any part of contact map that does
        # not match target ACs for a given chain
        # (but not PDB residues that have not been matched
        # to UniProt residues)
        if uniprot_ac_filter is not None:
            for s, target_ac in uniprot_ac_filter.items():
                df = df.loc[((df.pdb_chain != s) | (df.uniprot_ac.isnull()) |
                            (df.uniprot_ac == target_ac))
                            ]

        # make sure there is only one Uniprot AC per selector
        for s in self.chains:
            acs = set(df.loc[df.pdb_chain == s, "uniprot_ac"].dropna())

            if len(acs) > 1:
                raise ValueError("Multiple Uniprot ACs for chain {}: {}"
                                 .format(s, acs))

        # apply Uniprot index transformations
        if uniprot_index_transform is not None:
            for (chain, transform) in uniprot_index_transform.items():
                if chain not in self.chains:
                    print("Chain not in dataframe:", chain, file=stderr)
                    continue

                df.loc[df.pdb_chain == chain, "up_index"] = \
                    df.loc[df.pdb_chain == chain, "up_index"].map(transform)

        # generate secondary structure reduction to helix, sheet, other
        df.loc[:, "ss_3state"] = df.loc[:, "ss"].map(
            lambda x: DSSPMap._REDUCTION[x] if x in DSSPMap._REDUCTION else "-"
        )

        self.df = df

    def get_dict(self, chain, feature):
        df_chain = self.df.loc[(self.df.pdb_chain == chain) &
                               (self.df.uniprot_ac.notnull())]

        values = df_chain.loc[:, feature]
        if feature in DSSPMap._NUMERIC_FEATURES:
            values = values.convert_objects(convert_numeric=True)

        return dict(zip(df_chain.up_index.astype(int),
                        values))
