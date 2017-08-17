"""
PDB structure handling based on MMTF format

Authors:
  Thomas A. Hopf
"""

from collections import OrderedDict, Iterable
from os import path
from urllib.error import HTTPError

from mmtf import fetch, parse
import numpy as np
import pandas as pd

from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.constants import AA3_to_AA1
from evcouplings.utils.helpers import DefaultOrderedDict
from evcouplings.utils.system import (
    valid_file, ResourceError, tempdir
)

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

# format string for PDB ATOM records
PDB_FORMAT = (
    "{atom:<6s}{atom_id:>5} "
    "{atom_name:4s}{alt_loc_ind:1s}{residue_name:<3s} "
    "{chain_id:1s}{residue_id:>4}{ins_code:1}   "
    "{x_coord:>8.3f}{y_coord:>8.3f}{z_coord:>8.3f}"
    "{occupancy:>6.2f}{temp_factor:>6.2f}          "
    "{element_symbol:>2}{charge:>2}"
)


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

        # reset atom index to consecutive numbers from 0
        coords = coords.reset_index(drop=True)

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
        atom_name : str or list-like, optional (default: "CA")
            Name(s) of atoms to keep

        Returns
        -------
        Chain
            Chain containing only filtered atoms (and those
            residues that have such an atom)
        """
        if isinstance(atom_name, str):
            sel = self.coords.atom_name == atom_name
        else:
            sel = self.coords.atom_name.isin(atom_name)

        # update dataframe to rows having the right atom(s)
        coords = self.coords.loc[sel].copy()

        # reset atom index to consecutive numbers from 0
        coords = coords.reset_index(drop=True)

        # if there are residues without any atoms, remove
        # the entire residue
        residues = self.residues.loc[
            self.residues.index.isin(coords.residue_index)
        ].copy()

        return Chain(residues, coords)

    def filter_positions(self, positions):
        """
        Select a subset of positions from the chain
        
        Parameters
        ----------
        positions : list-like
            Set of residues that will be kept

        Returns
        -------
        Chain
            Chain containing only the selected residues
        """
        # map all positions to be strings
        positions = [str(p) for p in positions]

        residues = self.residues.loc[
            self.residues.id.isin(positions)
        ].copy()

        # drop coordinates of residues that were not kept
        coords = self.coords.loc[
            self.coords.residue_index.isin(residues.index)
        ]

        # reset atom index to consecutive numbers from 0
        coords = coords.reset_index(drop=True)

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
               numbering. End index or range is \*inclusive*\
               Note that residue IDs in the end will still
               be handled as strings when mapping.

        source_id: {"seqres_id", "coord_id", "id"}, optional (default: "seqres_id")
            Residue identifier in chain to map \*from*\
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

    def to_file(self, fileobj, chain_id="A", end=True):
        """
        Write chain to a file in PDB format (mmCIF not yet
        supported).
        
        Note that PDB files written this function may not 
        be 100% compliant with the PDB format standards,
        in particular:

        * some HETATM records may turn into ATOM records
          when starting from an mmtf file, if the record
          has a one-letter code (such as MSE / M).

        * code does not print TER record at the end of
          a peptide chain
        
        Parameters
        ----------
        fileobj : file-like object
            Write to this file handle
        chain_id : str, optional (default: "A")
            Assign this chain name in file (allows to redefine
            chain name from whatever chain was originally)
        end : bool, optional (default: True)
            Print "END" record after chain (signals end of PDB file)
        """
        # merge residue-level information and atom-level information
        # in one joint table (i.e. the way data is presented in a
        # PDB/mmCIF file)
        x = self.coords.merge(
            self.residues, left_on="residue_index", right_index=True
        )

        # write one atom at a time
        for idx, r in x.iterrows():
            # split residue ID into position and insertion code
            cid = str(r["id"])
            if cid[-1].isalpha():
                coord_id = cid[:-1]
                ins_code = cid[-1]
            else:
                coord_id = cid
                ins_code = ""

            # atom element
            element = r["element"].upper()

            # need to split atom name into element and specifier
            # (e.g. beta carbon element:C, specifier:B) so we
            # can correctly justify in the 4-column atom
            # name field: first 2 (right-justified) are
            # element, second 2 (left-justified) are specifier
            src_atom_name = r["atom_name"]

            # to make things more complicated, there are cases like
            # HE21 (CNS) or 1HE2 (PDB) which break if assuming
            # that atom_element == element. In these cases, we
            # just use the full raw string
            if len(src_atom_name) == 4:
                atom_name = src_atom_name
            else:
                atom_element = src_atom_name[0:len(element)]
                atom_spec = src_atom_name[len(element):]
                atom_name = "{:>2s}{:<2s}".format(atom_element, atom_spec)

            # print charge if we have one (optional)
            charge = r["charge"]
            # test < and > to exclude nan values
            if charge < 0 or charge > 0:
                charge_sign = "-" if charge < 0 else "+"
                charge_value = abs(charge)
                charge_str = "{}{}".format(charge_value, charge_sign)
            else:
                charge_str = ""

            # format line and write
            s = PDB_FORMAT.format(
                atom="HETATM" if r["hetatm"] else "ATOM",
                atom_id=r["atom_id"],
                atom_name=atom_name,
                alt_loc_ind=r["alt_loc"],
                residue_name=r["three_letter_code"],
                chain_id=chain_id,
                residue_id=coord_id,
                ins_code=ins_code,
                x_coord=r["x"],
                y_coord=r["y"],
                z_coord=r["z"],
                occupancy=r["occupancy"],
                temp_factor=r["b_factor"],
                element_symbol=element,
                charge=charge_str,
            )
            fileobj.write(s + "\n")

        if end:
            fileobj.write("END" + 77 * " " + "\n")


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
        self._residue_type_elements = _group_info("elementList")
        self._residue_type_charges = _group_info("formalChargeList")

        self._residue_type_num_atoms = np.array([len(x) for x in self._residue_type_atom_names])

        # prepare alternative location list as numpy array, with empty strings
        self.alt_loc_list = np.array(
            [x.replace("\x00", "") for x in mmtf.alt_loc_list]
        )

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
        try:
            return cls(parse(filename))
        except FileNotFoundError as e:
            raise ResourceError(
                "Could not find file {}".format(filename)
            ) from e

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
        try:
            return cls(fetch(pdb_id))
        except HTTPError as e:
            raise ResourceError(
                "Could not fetch MMTF data for {}".format(pdb_id)
            ) from e

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
            Chain object containing DataFrames listing residues
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

        # proxy for HETATM records - will not work e.g. for MSE,
        # which is listed as "M"
        res_df.loc[:, "hetatm"] = res_df.one_letter_code.isnull()

        # finally, get atom names and coordinates for all residues
        atom_first = self.first_atom_index[residue_indeces]
        atom_last = self.last_atom_index[residue_indeces]
        atom_names = np.concatenate(self._residue_type_atom_names[group_types])
        elements = np.concatenate(self._residue_type_elements[group_types])
        charges = np.concatenate(self._residue_type_charges[group_types])

        residue_number = np.repeat(res_df.index, atom_last - atom_first)
        atom_indices = np.concatenate([
            np.arange(self.first_atom_index[i], self.last_atom_index[i])
            for i in residue_indeces
        ])

        coords = OrderedDict([
            ("residue_index", residue_number),
            ("atom_id", self.mmtf.atom_id_list[atom_indices]),
            ("atom_name", atom_names),
            ("element", elements),
            ("charge", charges),
            ("x", self.mmtf.x_coord_list[atom_indices]),
            ("y", self.mmtf.y_coord_list[atom_indices]),
            ("z", self.mmtf.z_coord_list[atom_indices]),
            ("alt_loc", self.alt_loc_list[atom_indices]),
            ("occupancy", self.mmtf.occupancy_list[atom_indices]),
            ("b_factor", self.mmtf.b_factor_list[atom_indices]),
        ])

        coord_df = pd.DataFrame(coords)

        return Chain(res_df, coord_df)


class ClassicPDB:
    """
    Class to handle "classic" PDB and mmCIF formats
    (for new mmtf format see PDB class above). Wraps
    around Biopython PDB functionality to provide a consistent
    interface.
    
    Unlike the PDB class (based on mmtf), this object will
    not be able to extract SEQRES indices corresponding to
    ATOM-record residue indices.
    """
    def __init__(self, structure):
        """
        Initialize from Biopython Structure object

        Normally one should use from_file() class
        method to create object.

        Parameters
        ----------
        structure : Bio.PDB.Structure.Structure
            Biopython structure object (returned by
            PDBParser, MMCIFParser or MMTFParser)
        """
        self.structure = structure
        self.models = [
            m.get_id() for m in self.structure
        ]
        self.model_to_chains = {
            m: [
                c.get_id() for c in self.structure[m]
            ] for m in self.models
        }


    @classmethod
    def from_file(cls, filename, file_format="pdb"):
        """
        Initialize structure from PDB/mmCIF file

        Parameters
        ----------
        filename : str
            Path of file
        file_format : {"pdb", "cif"}, optional (default: "pdb")
            Format of structure (old PDB format or mmCIF)

        Returns
        -------
        ClassicPDB
            Initialized PDB structure
        """
        try:
            if file_format == "pdb":
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
            elif file_format == "cif":
                from Bio.PDB import FastMMCIFParser
                parser = FastMMCIFParser(QUIET=True)
            else:
                raise InvalidParameterError(
                    "Invalid file_format, valid options are: pdb, cif"
                )

            structure = parser.get_structure("", filename)
            return cls(structure)
        except FileNotFoundError as e:
            raise ResourceError(
                "Could not find file {}".format(filename)
            ) from e

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
        from urllib.error import URLError
        from Bio.PDB import PDBList
        pdblist = PDBList()

        try:
            # download PDB file to temporary directory
            pdb_file = pdblist.retrieve_pdb_file(pdb_id, pdir=tempdir())
            return cls.from_file(pdb_file, file_format="pdb")
        except URLError as e:
            raise ResourceError(
                "Could not fetch PDB data for {}".format(pdb_id)
            ) from e

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
            Chain object containing DataFrames listing residues
            and atom coordinates
        """
        # check requested model is valid
        if model not in self.models:
            raise ValueError(
                "Invalid model, valid models are: " +
                ",".join(map(str, self.models))
            )

        # check requested chain is valid
        if chain not in self.model_to_chains[model]:
            raise ValueError(
                "Invalid chain, valid chains are: " +
                ",".join(self.model_to_chains[model])
            )

        # get current chain
        c = self.structure[model][chain]

        res = DefaultOrderedDict(list)
        coords = DefaultOrderedDict(list)

        # iterate through all residues in chain and
        # accumulate information for dataframe building
        for r_idx, r in enumerate(c):
            het_flag, pos, ins_code = r.get_id()
            ins_code = ins_code.replace(" ", "")
            # print("x" + het_flag + "x", pos, "." + ins_code + ".")

            residue_id = "{}{}".format(pos, ins_code)
            res["id"].append(residue_id)
            # we don't have seqres ID from atom records, unless we
            # parsed additional annotation in header / mmCIF dict
            res["seqres_id"].append(np.nan)
            res["coord_id"].append(residue_id)

            # residue name in 3- and 1-letter code
            resname = r.get_resname()
            if resname in AA3_to_AA1:
                resname_one = AA3_to_AA1[resname]
            else:
                resname_one = np.nan
            res["one_letter_code"].append(resname_one)
            res["three_letter_code"].append(r.get_resname())

            # information that is only straightforward to get from mmtf
            res["chain_index"] = np.nan
            res["chain_id"] = np.nan
            res["sec_struct"] = np.nan
            res["sec_struct_3state"] = np.nan

            # at least we can identify easily what the hetatms
            # are (as opposed to mmtf)
            res["hetatm"].append(het_flag != " ")

            # now iterate through all atoms for current residue
            # and accumulate information; unpack atoms
            # with multiple locations (altloc)
            for a_idx, a in enumerate(r.get_unpacked_list()):
                # this index links residues to coords
                coords["residue_index"].append(r_idx)
                # atom-specific information
                coords["atom_id"].append(a.get_serial_number())
                coords["atom_name"].append(a.get_name())
                coords["element"].append(a.element)
                # cannot get charge information from Biopython?
                coords["charge"].append(np.nan)
                x, y, z = a.get_coord()
                coords["x"].append(x)
                coords["y"].append(y)
                coords["z"].append(z)
                coords["alt_loc"].append(
                    a.get_altloc().replace(" ", "")
                )
                coords["occupancy"].append(a.get_occupancy())
                coords["b_factor"].append(a.get_bfactor())

        # create residue table
        res_df = pd.DataFrame(res)

        # make sure that residue ids are strings
        res_df.loc[:, "coord_id"] = (
            res_df.loc[:, "coord_id"].astype(str)
        )

        # create coordinate table
        coord_df = pd.DataFrame(coords)

        return Chain(res_df, coord_df)


def load_structures(pdb_ids, structure_dir=None, raise_missing=True):
    """
    Load PDB structures from files / web

    Parameters
    ----------
    pdb_ids : Iterable
        List / iterable containing PDB identifiers
        to be loaded.
    structure_dir : str, optional (default: None)
        Path to directory with structures. Structures
        filenames must be in the format 5p21.mmtf.
        If a file can not be found, will try to fetch
        from web instead.
    raise_missing : bool, optional (default: True)
        Raise a ResourceError exception if any of the
        PDB IDs cannot be loaded. If False, missing
        entries will be ignored.

    Returns
    -------
    structures : dict(str -> PDB)
        Dictionary containing loaded structures.
        Keys (PDB identifiers) will be lower-case.

    Raises
    ------
    ResourceError
        Raised if raise_missing is True and any of the given
        PDB IDs cannot be loaded.
    """
    # collect loaded structures in dict(id -> PDB)
    structures = {}

    # load structure by structure
    for pdb_id in set(pdb_ids):
        pdb_id = pdb_id.lower()

        has_file = False
        if structure_dir is not None:
            structure_file = path.join(structure_dir, pdb_id + ".mmtf")
            has_file = valid_file(structure_file)

        try:
            # see if we can load locally from disk
            if has_file:
                structures[pdb_id] = PDB.from_file(structure_file)
            else:
                # otherwise fetch from web
                structures[pdb_id] = PDB.from_id(pdb_id)
        except (ResourceError, UnicodeDecodeError):
            # ResourceError: invalid PDB ID
            # UnicodeDecodeError: some random problem with mmtf library
            if raise_missing:
                raise

    return structures
