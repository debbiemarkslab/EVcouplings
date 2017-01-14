"""
PDB structure handling based on MMTF format

Authors:
  Thomas A. Hopf
"""

from collections import OrderedDict, Iterable
from mmtf import fetch, parse
import numpy as np
import pandas as pd

# Mapping from MMTF secondary structure codes to DSSP symbols
MMTF_DSSP_CODE_MAP = {
    0: "I",  # pi helix
    1: "S",  # bend
    2: "H",  # alpha helix
    3: "E",  # extended
    4: "G",  # 3-10 helix
    5: "B",  # bridge
    6: "T",  # turn
    7: "C",  # coil
    -1: "",  # undefined
}

# Reduction to 3-state secondary structure annotation
DSSP_3_STATE_MAP = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "C": "C",
    "T": "C",
    "S": "C",
}


class Chain:
    """
    Container for PDB chain residue and coordinate information
    """
    def __init__(self, residues, coords):
        """
        Create new PDB chain, indexed by residue coordinate
        indeces

        Parameters
        ----------
        residues : pandas.DataFrame
            List of residues (as computed by PDB.get_chain())
        coords : pandas.DataFrame
            List of atom coordinates (as computed by
            PDB.get_chain())
        """
        self.residues = residues
        self.coords = coords

    def _update_ids(self, ids):
        """
        Update residue identifiers, and remove any
        residue that does not have new id. Also
        removes corresponding atom coordinates.

        Parameters
        ----------
        ids : pandas.Series (or list-like)
            New identifiers to be assigned to
            residue table. Has to be of same length
            in same order as residue table.

        Returns
        -------
        Chain
            Chain with new residue identifiers
        """
        residues = self.residues.copy()
        residues.loc[:, "id"] = ids.copy()
        residues = residues.dropna(subset=["id"])

        # drop coordinates of residues that were not kept
        coords = self.coords.loc[
            self.coords.residue_index.isin(residues.index)
        ]

        return Chain(residues, coords)

    def to_seqres(self):
        """
        Return copy of chain with main index set to
        SEQRES numbering. Residues that do not have
        a SEQRES id will be dropped.

        Returns
        -------
        Chain
            Chain with seqres IDs as main index
        """
        return self._update_ids(
            self.residues.loc[:, "seqres_id"]
        )

    def filter_atoms(self, atom_name="CA"):
        """
        Filter coordinates of chain, e.g. to
        compute C_alpha-C_alpha distances

        Parameters
        ----------
        atom_name : str, optional (default: "CA")
            Name of atoms to keep

        Returns
        -------
        Chain
            Chain containing only filtered atoms (and those
            residues that have such an atom)
        """
        coords = self.coords.loc[
            self.coords.atom_name == atom_name
        ].copy()

        residues = self.residues.loc[
            self.residues.index.isin(coords.residue_index)
        ].copy()

        return Chain(residues, coords)

    def remap(self, mapping, source_id="seqres_id"):
        """
        Remap chain into different numbering scheme
        (e.g. from seqres to uniprot numbering)

        Parameters
        ----------
        mapping : dict
            Mapping of residue identifiers from
            source_id (current main index of PDB chain)
            to new identifiers.

            mapping may either be:
            1. dict(str -> str) to map individual residue
               IDs. Keys and values of dictionary will be
               typecast to string before the mapping, so it
               is possible to pass in integer values too
               (if the source or target IDs are numbers)
            2. dict((int, int) -> (int, int)) to map ranges
               of numbers to ranges of numbers. This should
               typically be only used with RESSEQ or UniProt
               numbering. End index or range is *inclusive*
               Note that residue IDs in the end will still
               be handled as strings when mapping.

        source_id: {"seqres_id", "coord_id", "id"}, optional (default: "seqres_id")
            Residue identifier in chain to map *from*
            (will be used as key to access mapping)

        Returns
        -------
        Chain
            Chain with remapped numbering ("id" column
            in residues DataFrame)
        """
        # get one key to test which type of mapping we have
        # (range-based, or individual residues)
        test_key = next(iter(mapping.keys()))

        # test for range-based mapping
        if isinstance(test_key, Iterable) and not isinstance(test_key, str):
            # build up inidividual residue mapping
            final_mapping = {}
            for (source_start, source_end), (target_start, target_end) in mapping.items():
                source = map(
                    str, range(source_start, source_end + 1)
                )

                target = map(
                    str, range(target_start, target_end + 1)
                )

                final_mapping.update(
                    dict(zip(source, target))
                )
        else:
            # individual residue mapping, make sure all strings
            final_mapping = {
                str(s): str(t) for (s, t) in mapping.items()
            }

        # remap identifiers using mapping
        ids = self.residues.loc[:, source_id].map(
            final_mapping, na_action="ignore"
        )

        # create remapped chain
        return self._update_ids(ids)


class PDB:
    """
    Wrapper around PDB MMTF decoder object to access residue and
    coordinate information
    """
    def __init__(self, mmtf):
        """
        Initialize by further decoding information in mmtf object.

        Normally one should use from_file() and from_id() class
        methods to create object.

        Parameters
        ----------
        mmtf : mmtf.api.mmtf_reader.MMTFDecoder
            MMTF decoder object (as returned by fetch or parse
            function in mmtf module)
        """
        def _get_range(object_counts):
            """
            Extract index ranges for chains, residues and atoms
            """
            last_element = np.cumsum(object_counts, dtype=int)
            first_element = np.concatenate(
                (np.zeros(1, dtype=int), last_element[:-1])
            )

            return first_element, last_element

        # store raw MMTF decoder
        self.mmtf = mmtf

        # Step 1: summarize basic information about model
        # number of models in structure
        self.num_models = len(mmtf.chains_per_model)

        self.first_chain_index, self.last_chain_index = _get_range(
            mmtf.chains_per_model
        )

        # collect list which chain corresponds to which entity
        self.chain_to_entity = {}
        for i, ent in enumerate(mmtf.entity_list):
            for c in ent["chainIndexList"]:
                self.chain_to_entity[c] = i

        # Step 2: identify residues and corresponding atom indices

        # first/last index of residue (aka group) for each chain;
        # index these lists with index of chain
        self.first_residue_index, self.last_residue_index = _get_range(
            mmtf.groups_per_chain
        )

        # store explicit information about composition of residues
        def _group_info(field):
            return np.array(
                [x[field] for x in mmtf.group_list]
            )

        # three and one letter code names of different groups
        self._residue_names_3 = _group_info("groupName")
        self._residue_names_1 = _group_info("singleLetterCode")

        # atom types and overall number of atoms in each type of residue
        self._residue_type_atom_names = _group_info("atomNameList")
        self._residue_type_num_atoms = np.array([len(x) for x in self._residue_type_atom_names])

        # compute first and last atom index for each residue/group
        # (by fetching corresponding length for each group based on group type)
        self._residue_num_atoms = self._residue_type_num_atoms[mmtf.group_type_list]

        self.first_atom_index, self.last_atom_index = _get_range(
            self._residue_num_atoms
        )

        # assemble residue ID strings
        self.residue_ids = np.array([
            "{}{}".format(group_id, ins_code.replace("\x00", ""))
            for group_id, ins_code
            in zip(mmtf.group_id_list, mmtf.ins_code_list)
        ])

        # map secondary structure codes into DSSP symbols
        self.sec_struct = np.array(
            [MMTF_DSSP_CODE_MAP[x] for x in mmtf.sec_struct_list]
        )

    @classmethod
    def from_file(cls, filename):
        """
        Initialize structure from MMTF file

        Parameters
        ----------
        filename : str
            Path of MMTF file

        Returns
        -------
        PDB
            initialized PDB structure
        """
        return cls(parse(filename))

    @classmethod
    def from_id(cls, pdb_id):
        """
        Initialize structure by PDB ID (fetches
        structure from RCSB servers)

        Parameters
        ----------
        pdb_id : str
            PDB identifier (e.g. 1hzx)

        Returns
        -------
        PDB
            initialized PDB structure
        """
        return cls(fetch(pdb_id))

    def get_chain(self, chain, model=0):
        """
        Extract residue information and atom coordinates
        for a given chain in PDB structure

        Parameters
        ----------
        chain : str
            Name of chain to be extracted (e.g. "A")
        model : int, optional (default: 0)
            Index of model to be extracted

        Returns
        -------
        Chain
            namedtuple containing DataFrames listing residues
            and atom coordinates
        """
        if not (0 <= model < self.num_models):
            raise ValueError(
                "Illegal model index, can be from 0 up to {}".format(
                    self.num_models - 1
                )
            )

        # first and last index of chains corresponding to selected model
        first_chain_index = self.first_chain_index[model]
        last_chain_index = self.last_chain_index[model]

        # which model chains match our target PDB chain?
        chain_names = np.array(
            self.mmtf.chain_name_list[first_chain_index:last_chain_index]
        )

        # indices of chains that match chain name, in current model
        indices = np.arange(first_chain_index, last_chain_index, dtype=int)
        target_chain_indeces = indices[chain_names == chain]

        if len(target_chain_indeces) == 0:
            raise ValueError(
                "No chains with given name found"
            )

        # collect internal indeces of all residues/groups in chain
        residue_indeces = np.concatenate(
            np.array([
                np.arange(self.first_residue_index[i], self.last_residue_index[i])
                for i in target_chain_indeces
            ])
        )

        # chain indeces and identifiers for all residues
        # (not to be confused with chain name!);
        # maximum length 4 characters according to MMTF spec
        chain_indeces = np.concatenate([
            np.full(
                self.last_residue_index[i] - self.first_residue_index[i],
                i, dtype=int
            ) for i in target_chain_indeces
        ])

        chain_ids = np.array(self.mmtf.chain_id_list)[chain_indeces]

        # create dataframe representation of selected chain
        m = self.mmtf
        group_types = m.group_type_list[residue_indeces]

        res = OrderedDict([
            ("id", self.residue_ids[residue_indeces]),
            ("seqres_id", m.sequence_index_list[residue_indeces]),
            ("coord_id", self.residue_ids[residue_indeces]),
            ("one_letter_code", self._residue_names_1[group_types]),
            ("three_letter_code", self._residue_names_3[group_types]),
            ("chain_index", chain_indeces),
            ("chain_id", chain_ids),
            ("sec_struct", self.sec_struct[residue_indeces]),
        ])

        res_df = pd.DataFrame(res)

        # shift seqres indexing to start at 1;
        # However, do not add to positions without sequence index (-1)
        res_df.loc[res_df.seqres_id >= 0, "seqres_id"] += 1

        # turn all indeces into strings and create proper NaN values
        res_df.loc[:, "coord_id"] = (
            res_df.loc[:, "coord_id"].astype(str)
        )

        res_df.loc[:, "seqres_id"] = (
            res_df.loc[:, "seqres_id"].astype(str).replace("-1", np.nan)
        )
        # copy updated coordinate indeces
        res_df.loc[:, "id"] = res_df.loc[:, "coord_id"]

        res_df.loc[:, "one_letter_code"] = res_df.loc[:, "one_letter_code"].replace("?", np.nan)
        res_df.loc[:, "sec_struct"] = res_df.loc[:, "sec_struct"].replace("", np.nan)

        # reduction to 3-state secondary structure (following Rost & Sander)
        res_df.loc[:, "sec_struct_3state"] = res_df.loc[:, "sec_struct"].map(
            lambda x: DSSP_3_STATE_MAP[x], na_action="ignore"
        )

        # finally, get atom names and coordinates for all residues
        atom_first = self.first_atom_index[residue_indeces]
        atom_last = self.last_atom_index[residue_indeces]
        atom_names = np.concatenate(self._residue_type_atom_names[group_types])
        residue_number = np.repeat(res_df.index, atom_last - atom_first)
        atom_indeces = np.concatenate([
            np.arange(self.first_atom_index[i], self.last_atom_index[i])
            for i in residue_indeces
        ])

        coords = OrderedDict([
            ("residue_index", residue_number),
            ("atom_id", self.mmtf.atom_id_list[atom_indeces]),
            ("atom_name", atom_names),
            ("x", self.mmtf.x_coord_list[atom_indeces]),
            ("y", self.mmtf.y_coord_list[atom_indeces]),
            ("z", self.mmtf.z_coord_list[atom_indeces]),
        ])

        coord_df = pd.DataFrame(coords)

        return Chain(res_df, coord_df)
