"""
Uniprot to PDB structure identification and
index mapping using the SIFTS database
(https://www.ebi.ac.uk/pdbe/docs/sifts/)

This functionality is centered around the
pdb_chain_uniprot.csv table available from SIFTS.
(ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)

Authors:
  Thomas A. Hopf
"""

from os import path
import json
from collections import OrderedDict

import pandas as pd
import requests

from evcouplings.align.alignment import (
    Alignment, read_fasta, parse_header
)

from evcouplings.align.protocol import jackhmmer_search
from evcouplings.align.tools import read_hmmer_domtbl
from evcouplings.compare.mapping import alignment_index_mapping, map_indices
from evcouplings.utils.system import (
    get_urllib, ResourceError, valid_file, tempdir
)
from evcouplings.utils.config import parse_config
from evcouplings.utils.helpers import range_overlap

UNIPROT_MAPPING_URL = "http://www.uniprot.org/mapping/"
SIFTS_URL = "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz"
SIFTS_REST_API = "http://www.ebi.ac.uk/pdbe/api/mappings/uniprot_segments/{}"

# TODO: make this default parametrization more explicit (e.g. a config file in repository)
# these parameters are fed as a default into SIFTS.by_alignment so that the method can be
# easily used without a configuration file/any further setup
JACKHMMER_CONFIG = """
prefix:
sequence_id:
sequence_file:
region:
first_index: 1

use_bitscores: True
domain_threshold: 0.5
sequence_threshold: 0.5
iterations: 1
database: sequence_database

extract_annotation: False
cpu: 1
nobias: False
reuse_alignment: False
checkpoints_hmm: False
checkpoints_ali: False

# database
jackhmmer: jackhmmer
sequence_database:
sequence_download_url: http://www.uniprot.org/uniprot/{}.fasta
"""


def fetch_uniprot_mapping(ids, from_="ACC", to="ACC", format="fasta"):
    """
    Fetch data from UniProt ID mapping service
    (e.g. download set of sequences)

    Parameters
    ----------
    ids : list(str)
        List of UniProt identifiers for which to
        retrieve mapping
    from_ : str, optional (default: "ACC")
        Source identifier (i.e. contained in "ids" list)
    to : str, optional (default: "ACC")
        Target identifier (to which source should be mapped)
    format : str, optional (default: "fasta")
        Output format to request from Uniprot server

    Returns
    -------
    str:
        Response from UniProt server
    """
    params = {
        "from": from_,
        "to": to,
        "format": format,
        "query": " ".join(ids)
    }
    url = UNIPROT_MAPPING_URL
    r = requests.post(url, data=params)

    if r.status_code != requests.codes.ok:
        raise ResourceError(
            "Invalid status code ({}) for URL: {}".format(
                r.status_code, url
            )
        )

    return r.text


def find_homologs_jackhmmer(**kwargs):
    """
    Identify homologs using jackhmmer

    Parameters
    ----------
    **kwargs
        Passed into jackhmmer_search protocol
        (see documentation for available options)

    Returns
    -------
    ali : evcouplings.align.Alignment
        Alignment of homologs of query sequence
        in sequence database
    hits : pandas.DataFrame
        Tabular representation of hits
    """
    # load default configuration
    config = parse_config(JACKHMMER_CONFIG)

    # update with overrides from kwargs
    config = {
        **config,
        **kwargs,
    }

    # create temporary output if no prefix is given
    if config["prefix"] is None:
        config["prefix"] = path.join(tempdir(), "compare")

    # run jackhmmer against sequence database
    ar = jackhmmer_search(**config)

    with open(ar["raw_alignment_file"]) as a:
        ali = Alignment.from_file(a, "stockholm")

    # write alignment as FASTA file for easier checking by hand,
    # if necessary
    with open(config["prefix"] + "_raw.fasta", "w") as f:
        ali.write(f)

    # read hmmer hittable and simplify
    hits = read_hmmer_domtbl(ar["hittable_file"])

    hits.loc[:, "uniprot_ac"] = hits.loc[:, "target_name"].map(lambda x: x.split("|")[1])
    hits.loc[:, "uniprot_id"] = hits.loc[:, "target_name"].map(lambda x: x.split("|")[2])

    hits = hits.rename(
        columns={
            "domain_score": "bitscore",
            "domain_i_Evalue": "e_value",
            "ali_from": "alignment_start",
            "ali_to": "alignment_end",
        }
    )

    hits.loc[:, "alignment_start"] = pd.to_numeric(hits.alignment_start).astype(int)
    hits.loc[:, "alignment_end"] = pd.to_numeric(hits.alignment_end).astype(int)

    hits.loc[:, "alignment_id"] = (
        hits.target_name + "/" +
        hits.alignment_start.astype(str) + "-" +
        hits.alignment_end.astype(str)
    )

    hits = hits.loc[
        :, ["alignment_id", "uniprot_ac", "uniprot_id", "alignment_start",
            "alignment_end", "bitscore", "e_value"]
    ]

    return ali, hits


class SIFTSResult:
    """
    Store results of SIFTS structure/mapping identification.

    (Full class defined for easify modification of fields)
    """
    def __init__(self, hits, mapping):
        """
        Create new SIFTS structure / mapping record.

        Parameters
        ----------
        hits : pandas.DataFrame
            Table with identified PDB chains
        mapping : dict
            Mapping from seqres to Uniprot numbering
            for each PDB chain
            (index by mapping_index column in hits
            dataframe)
        """
        self.hits = hits
        self.mapping = mapping


class SIFTS:
    """
    Provide Uniprot to PDB mapping data and functions
    starting from SIFTS mapping table.
    """
    def __init__(self, sifts_table_file, sequence_file=None):
        """
        Create new SIFTS mapper from mapping table.

        Note that creation of the mapping files, if not existing,
        takes a while.

        Parameters
        ----------
        sifts_table_file : str
            Path to *corrected* SIFTS pdb_chain_uniprot.csv
            To generate this file, point to an empty file path.
        sequence_file : str, optional (default: None)
            Path to file containing all UniProt sequences
            in SIFTS (used for homology-based identification
            of structures).
            Note: This file can be created using the
            create_sequence_file() method.
        """
        # test if table exists, if not, download and modify
        if not valid_file(sifts_table_file):
            self._create_mapping_table(sifts_table_file)

        self.table = pd.read_csv(
            sifts_table_file, comment="#"
        )

        # final table has still some entries where lengths do not match,
        # remove thlse
        self.table = self.table.query(
            "(resseq_end - resseq_start) == (uniprot_end - uniprot_start)"
        )

        self.sequence_file = sequence_file

        # if path for sequence file given, but not there, create
        if sequence_file is not None and not valid_file(sequence_file):
            self.create_sequence_file(sequence_file)

        # add Uniprot ID column if we have sequence mapping
        # from FASTA file
        if self.sequence_file is not None:
            self._add_uniprot_ids()

    def _create_mapping_table(self, sifts_table_file):
        """
        Create modified SIFTS mapping table (based on
        file at SIFTS_URL). For some of the entries,
        the Uniprot sequence ranges do not map to a.
        SEQRES sequence range of the same length. These
        PDB IDs will be entirely replaced by a segment-
        based mapping extracted from the SIFTS REST API.

        Parameters
        ----------
        sifts_table_file : str
            Path where computed table will be stored
        """
        def extract_rows(M, pdb_id):
            res = []

            M = M[pdb_id.lower()]["UniProt"]

            for uniprot_ac, Ms in M.items():
                for x in Ms["mappings"]:
                    res.append({
                        "pdb_id": pdb_id,
                        "pdb_chain": x["chain_id"],
                        "uniprot_ac": uniprot_ac,
                        "resseq_start": x["start"]["residue_number"],
                        "resseq_end": x["end"]["residue_number"],
                        "coord_start": (
                            str(x["start"]["author_residue_number"]) +
                            x["start"]["author_insertion_code"].replace(" ", "")
                        ),
                        "coord_end": (
                            str(x["end"]["author_residue_number"]) +
                            x["end"]["author_insertion_code"].replace(" ", "")
                        ),
                        "uniprot_start": x["unp_start"],
                        "uniprot_end": x["unp_end"],
                    })

            return res

        get_urllib(SIFTS_URL, sifts_table_file)

        # load table and rename columns for internal use, if SIFTS
        # ever decided to rename theirs
        table = pd.read_csv(
            sifts_table_file, comment="#",
            compression="gzip"
        ).rename(
            columns={
                "PDB": "pdb_id",
                "CHAIN": "pdb_chain",
                "SP_PRIMARY": "uniprot_ac",
                "RES_BEG": "resseq_start",
                "RES_END": "resseq_end",
                "PDB_BEG": "coord_start",
                "PDB_END": "coord_end",
                "SP_BEG": "uniprot_start",
                "SP_END": "uniprot_end",
            }
        )

        # identify problematic PDB IDs
        problematic_ids = table.query(
            "(resseq_end - resseq_start) != (uniprot_end - uniprot_start)"
        ).pdb_id.unique()

        # collect new mappings from segment based REST API
        res = []
        for i, pdb_id in enumerate(problematic_ids):
            r = requests.get(
                SIFTS_REST_API.format(pdb_id.lower())
            )
            mapping = json.loads(r.text)

            res += extract_rows(mapping, pdb_id)

        # remove bad PDB IDs from table and add new mapping
        new_table = table.loc[~table.pdb_id.isin(problematic_ids)]
        new_table = new_table.append(
            pd.DataFrame(res).loc[:, table.columns]
        )

        # save for later reuse
        new_table.to_csv(sifts_table_file, index=False)

    def _add_uniprot_ids(self):
        """
        Add Uniprot ID column to SIFTS table based on
        AC to ID mapping extracted from sequence database
        """
        # iterate through headers in sequence file and store
        # AC to ID mapping
        ac_to_id = {}
        with open(self.sequence_file) as f:
            for seq_id, _ in read_fasta(f):
                _, ac, id_ = seq_id.split(" ")[0].split("|")
                ac_to_id[ac] = id_

        # add column to dataframe
        self.table.loc[:, "uniprot_id"] = self.table.loc[:, "uniprot_ac"].map(ac_to_id)

    def create_sequence_file(self, output_file):
        """
        Create FASTA sequence file containing all UniProt
        sequences of proteins in SIFTS. This file is required
        for homology-based structure identification and
        index remapping.
        This function will also automatically associate
        the sequence file with the SIFTS object.

        Parameters
        ----------
        output_file : str
            Path at which to store sequence file
        """
        ids = self.table.uniprot_ac.unique().tolist()

        CHUNK_SIZE = 1000
        chunks = [
            ids[i:i + CHUNK_SIZE] for i in range(0, len(ids), CHUNK_SIZE)
        ]

        with open(output_file, "w") as f:
            for ch in chunks:
                # fetch sequence chunk
                seqs = fetch_uniprot_mapping(ch)

                # then store to FASTA file
                f.write(seqs)

        self.sequence_file = output_file

        # add Uniprot ID column to SIFTS table
        self._add_uniprot_ids()

    def _create_sequence_file(self, output_file):
        """
        Create FASTA sequence file containing all UniProt
        sequences of proteins in SIFTS. This file is required
        for homology-based structure identification and
        index remapping.
        This function will also automatically associate
        the sequence file with the SIFTS object.

        Note: this would be the nicer function, but unfortunately
        the UniProt server frequently closes the connection running it

        Parameters
        ----------
        output_file : str
            Path at which to store sequence file
        """
        # fetch all the sequences
        seqs = fetch_uniprot_mapping(
            self.table.uniprot_ac.unique().tolist()
        )

        # then store to FASTA file
        with open(output_file, "w") as f:
            f.write(seqs)

        self.sequence_file = output_file

    def _finalize_hits(self, hit_segments):
        """
        Create final hit/mapping record from
        table of segments in PDB chains in
        SIFTS file.

        Parameters
        ----------
        hit_segments : pd.DataFrame
            Subset of self.table that will be
            turned into final mapping record

        Returns
        -------
        SIFTSResult
            Identified hits plus index mappings
            to Uniprot
        """
        # compile final set of hits
        hits = []

        # compile mapping from Uniprot to seqres for
        # each final hit
        mappings = {}

        # go through all SIFTS segments per PDB chain
        for i, ((pdb_id, pdb_chain), chain_grp) in enumerate(
            hit_segments.groupby(["pdb_id", "pdb_chain"])
        ):
            # put segments together in one segment-based
            # mapping for chain; this will be used by pdb.Chain.remap()
            mapping = {
                (r["resseq_start"], r["resseq_end"]): (r["uniprot_start"], r["uniprot_end"])
                for j, r in chain_grp.iterrows()
            }

            # append current hit and mapping
            hits.append([pdb_id, pdb_chain, i])
            mappings[i] = mapping

        # create final hit representation as DataFrame
        hits_df = pd.DataFrame(
            hits, columns=["pdb_id", "pdb_chain", "mapping_index"]
        )

        return SIFTSResult(hits_df, mappings)

    def by_pdb_id(self, pdb_id, pdb_chain=None, uniprot_id=None):
        """
        Find structures and mapping by PDB id
        and chain name

        Parameters
        ----------
        pdb_id : str
            4-letter PDB identifier
        pdb_chain : str, optional (default: None)
            PDB chain name (if not given, all
            chains for PDB entry will be returned)
        uniprot_id : str, optional (default: None)
            Filter to keep only this Uniprot accession
            number or identifier (necessary for chimeras,
            or multi-chain complexes with different proteins)

        Returns
        -------
        SIFTSResult
            Identified hits plus index mappings
            to Uniprot

        Raises
        ------
        ValueError
            If selected segments in PDB file do
            not unambigously map to one Uniprot
            entry
        """
        pdb_id = pdb_id.lower()
        query = "pdb_id == @pdb_id"

        # filter by PDB chain if selected
        if pdb_chain is not None:
            query += " and pdb_chain == @pdb_chain"

        # filter by UniProt AC/ID if selected
        # (to remove chimeras)
        if uniprot_id is not None:
            if "uniprot_id" in self.table.columns:
                query += (" and (uniprot_ac == @uniprot_id or "
                          "uniprot_id == @uniprot_id)")
            else:
                query += " and uniprot_ac == @uniprot_id"

        x = self.table.query(query)

        # check we only have one protein (might not
        # be the case with multiple chains, or with
        # chimeras)
        if len(x.uniprot_ac.unique()) > 1:
            id_list = ", ".join(x.uniprot_ac.unique())

            if "uniprot_id" in self.table.columns:
                id_list += " or " + ", ".join(x.uniprot_id.unique())

            raise ValueError(
                "Multiple Uniprot sequences on chains, "
                "please disambiguate using uniprot_id "
                "parameter: {}".format(id_list)
            )

        # create hit and mapping result
        return self._finalize_hits(x)

    def by_uniprot_id(self, uniprot_id, reduce_chains=False):
        """
        Find structures and mapping by Uniprot
        access number.

        Parameters
        ----------
        uniprot_ac : str
            Find PDB structures for this Uniprot accession
            number. If sequence_file was given while creating
            the SIFTS object, Uniprot identifiers can also be
            used.
        reduce_chains : bool, optional (Default: True)
            If true, keep only first chain per PDB ID
            (i.e. remove redundant occurrences of same
            protein in PDB structures). Should be set to
            False to identify homomultimeric contacts.

        Returns
        -------
        SIFTSResult
            Record of hits and mappings found for this
            Uniprot protein. See by_pdb_id() for detailed
            explanation of fields.
        """
        query = "uniprot_ac == @uniprot_id"

        if "uniprot_id" in self.table.columns:
            query += " or uniprot_id == @uniprot_id"

        x = self.table.query(query)

        hit_table = self._finalize_hits(x)

        # only retain one chain if this option is active
        if reduce_chains:
            hit_table.hits = hit_table.hits.groupby(
                "pdb_id"
            ).first().reset_index()

        return hit_table

    def by_alignment(self, min_overlap=20, reduce_chains=False, **kwargs):
        """
        Find structures by sequence alignment between
        query sequence and sequences in PDB.

        # TODO: offer option to start from HMM profile for this

        Parameters
        ----------
        min_overlap : int, optional (default: 20)
            Require at least this many aligned positions
            with the target structure
        reduce_chains : bool, optional (Default: True)
            If true, keep only first chain per PDB ID
            (i.e. remove redundant occurrences of same
            protein in PDB structures). Should be set to
            False to identify homomultimeric contacts.
        **kwargs
            Passed into jackhmmer_search protocol
            (see documentation for available options).
            Additionally, if "prefix" is given, individual
            mappings will be saved to files suffixed by
            the respective key in mapping table.

        Returns
        -------
        SIFTSResult
            Record of hits and mappings found for this
            query sequence by alignment. See by_pdb_id()
            for detailed explanation of fields.
        """
        def _create_mapping(r):
            _, query_start, query_end = parse_header(ali.ids[0])

            # create mapping from query into PDB Uniprot sequence
            # A_i will be query sequence indices, A_j Uniprot sequence indices
            m = map_indices(
                ali[0], query_start, query_end,
                ali[r["alignment_id"]], r["alignment_start"], r["alignment_end"]
            )

            # create mapping from PDB Uniprot into seqres numbering
            # j will be Uniprot sequence index, k seqres index
            n = pd.DataFrame(
                {
                    "j": list(range(r["uniprot_start"], r["uniprot_end"] + 1)),
                    "k": list(range(r["resseq_start"], r["resseq_end"] + 1)),
                }
            )

            # need to convert to strings since other mapping has indices as strings
            n.loc[:, "j"] = n.j.astype(str)
            n.loc[:, "k"] = n.k.astype(str)

            # join over Uniprot indices (i.e. j);
            # get rid of any position that is not aligned
            mn = m.merge(n, on="j", how="inner").dropna()

            # extract final mapping from seqres (k) to query (i)
            map_ = dict(
                zip(mn.k, mn.i)
            )

            return map_, mn

        if self.sequence_file is None:
            raise ValueError(
                "Need to have SIFTS sequence file. "
                "Create using create_sequence_file() "
                "method or constructor."
            )

        ali, hits = find_homologs_jackhmmer(
            sequence_database=self.sequence_file, **kwargs
        )

        # merge with internal table to identify overlap of
        # aligned regions and regions with structural coverage
        hits = hits.merge(
            self.table, on="uniprot_ac", suffixes=("", "_")
        )

        # add 1 to end of range since overlap function treats
        # ends as exclusive, while ends here are inclusive
        hits.loc[:, "overlap"] = [
            range_overlap(
                (r["uniprot_start"], r["uniprot_end"] + 1),
                (r["alignment_start"], r["alignment_end"] + 1)
            ) for i, r in hits.iterrows()
        ]

        # collect complete index mappings in here...
        mappings = {}
        # ... as well as dataframe rows for assignment of hit to mapping
        mapping_rows = []

        # complication: if there are multiple segments per hit and chain, we should
        # reduce these into a single mapping (even though split mappings
        # are possible in principle) so we can count unique number of hits etc.
        hit_columns = ["alignment_id", "pdb_id", "pdb_chain"]
        for i, (hit, grp) in enumerate(
            hits.groupby(hit_columns)
        ):
            agg_mapping = {}
            agg_df = pd.DataFrame()
            # go through each segment
            for j, r in grp.iterrows():
                # compute mapping for that particular segment
                map_j, map_j_df = _create_mapping(r)

                # add to overall mapping dictionary for this hit
                agg_mapping.update(map_j)
                agg_df = agg_df.append(map_j_df)

            # store assignment of group to mapping index
            mapping_rows.append(
                list(hit) + [i, len(grp) > 1]
            )

            mappings[i] = agg_mapping

            # store index mappings if filename prefix is given
            prefix = kwargs.get("prefix", None)
            if prefix is not None:
                agg_df = agg_df.rename(
                    columns={
                        "j": "uniprot_of_pdb_index",
                        "A_j": "uniprot_of_pdb_residue",
                        "k": "pdb_seqres_index",
                    }
                )

                agg_df.to_csv(
                    "{}_mapping{}.csv".format(prefix, i), index=False
                )

        # create dataframe from mapping rows
        mapping_df = pd.DataFrame(
            mapping_rows, columns=hit_columns + [
                "mapping_index", "grouped_segments",
            ]
        )

        # now group again, to aggregate full hit dataframe
        def _agg_type(x):
            if x == "overlap":
                return "sum"
            elif x.endswith("_start"):
                return "min"
            elif x.endswith("end"):
                return "max"
            else:
                return "first"

        agg_types = OrderedDict(
            [(c, _agg_type(c)) for c in hits.columns
             if c not in hit_columns]
        )

        hits_grouped = hits.groupby(
            hit_columns
        ).agg(agg_types).reset_index()

        # join with mapping information
        hits_grouped = hits_grouped.merge(
            mapping_df, on=hit_columns
        )

        # remove hits with too little residue coverage
        hits_grouped = hits_grouped.query("overlap >= @min_overlap")

        hits_grouped.loc[:, "bitscore"] = pd.to_numeric(
            hits_grouped.loc[:, "bitscore"], errors="coerce"
        )
        hits_grouped = hits_grouped.sort_values(by="bitscore", ascending=False)

        # if requested, only keep one chain per PDB;
        # sort by score before this to keep best hit
        if reduce_chains:
            hits_grouped = hits_grouped.groupby("pdb_id").first().reset_index()
            # sort again, just to be sure...
            hits_grouped = hits_grouped.sort_values(by="bitscore", ascending=False)

        # remove any zombie mappings we did not keep in table
        mappings = {
            idx: map_ for idx, map_ in mappings.items()
            if idx in hits_grouped.mapping_index.values
        }

        return SIFTSResult(hits_grouped, mappings)
