"""
PDB structure handling based on MMTF format

Authors:
  Thomas A. Hopf
"""

from collections import namedtuple, OrderedDict
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

# Store residue and coordinate information for a PDB chain
Chain = namedtuple("Chain", ["residues", "coords"])


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
            ("resseq_index", m.sequence_index_list[residue_indeces]),
            ("coord_index", self.residue_ids[residue_indeces]),
            ("one_letter_code", self._residue_names_1[group_types]),
            ("three_letter_code", self._residue_names_3[group_types]),
            ("chain_index", chain_indeces),
            ("chain_id", chain_ids),
            ("sec_struct", self.sec_struct[residue_indeces]),
        ])

        res_df = pd.DataFrame(res)

        # shift seqres indexing to start at 1;
        # However, do not add to positions without sequence index (-1)
        res_df.loc[res_df.resseq_index >= 0, "resseq_index"] += 1

        # turn all indeces into strings and create proper NaN values
        res_df.loc[:, "coord_index"] = (
            res_df.loc[:, "coord_index"].astype(str)
        )

        res_df.loc[:, "resseq_index"] = (
            res_df.loc[:, "resseq_index"].astype(str).replace("-1", np.nan)
        )

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
