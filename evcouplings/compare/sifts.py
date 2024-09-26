"""
Uniprot to PDB structure identification and
index mapping using the SIFTS database
(https://www.ebi.ac.uk/pdbe/docs/sifts/)

This functionality is centered around the
pdb_chain_uniprot.csv table available from SIFTS.
(ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)

Authors:
  Thomas A. Hopf
  Anna G. Green (find_homologs)
  Chan Kang (find_homologs)
"""

from io import StringIO
from os import path
from collections import OrderedDict
from copy import deepcopy
import time
import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

from evcouplings.align.alignment import (
    Alignment, read_fasta, parse_header, write_fasta
)
from evcouplings.align.protocol import (
    jackhmmer_search, hmmbuild_and_search
)
from evcouplings.align.tools import read_hmmer_domtbl
from evcouplings.compare.mapping import map_indices
from evcouplings.utils.system import (
    get_urllib, ResourceError, valid_file, tempdir, temp
)
from evcouplings.utils.config import (
    parse_config, check_required, InvalidParameterError
)
from evcouplings.utils.helpers import range_overlap

UNIPROT_MAPPING_URL = "https://rest.uniprot.org"
SIFTS_URL = "ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_segments_observed.csv.gz"
SIFTS_REST_API = "http://www.ebi.ac.uk/pdbe/api/mappings/uniprot_segments/{}"

# TODO: make this default parametrization more explicit (e.g. a config file in repository)
# these parameters are fed as a default into SIFTS.by_alignment so that the method can be
# easily used without a configuration file/any further setup
HMMER_CONFIG = """
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
sequence_download_url: http://rest.uniprot.org/uniprot/{}.fasta
"""


def fetch_uniprot_mapping(ids, from_db="UniProtKB_AC-ID", to_db="UniProtKB", format="fasta", isoforms=True,
                          polling_interval=3, retry_kws=None):
    """
    Fetch data from UniProt ID mapping service
    (e.g. download set of sequences)

    Updated to match 2022 UniProt update, code is a simpflied version of
    https://www.uniprot.org/help/id_mapping

    For valid from_db/to_db pairs check https://rest.uniprot.org/configure/idmapping/fields

    Parameters
    ----------
    ids : list(str)
        List of UniProt identifiers for which to
        retrieve mapping
    from_db : str, optional (default: "UniProtKB_AC-ID")
        Source identifier (i.e. contained in "ids" list)
    to_db : str, optional (default: "UniProtKB")
        Target identifier (to which source should be mapped)
    format : str, optional (default: "fasta")
        Output format to request from Uniprot server
    isoforms : bool, optional (default: True)
        Include isoform sequences in retrieval. Note that API will return *all*
        isoforms for entry, not just the requested ones, i.e. output will
        need to be filtered after retrieval. Output may also contain duplicate
        entries if multiple isoforms were present in requested identifier set.
    polling_interval : int, optional (default: 3)
        Number of seconds to wait before polling for request results
    retry_kws : dict, optional (default: None)
        Retry parameters supplied as kwargs to requests.adapters.Retry

    Returns
    -------
    result: requests.models.Response
        Response from UniProt server. Use result.text or result.json()
        to access results
    """
    if retry_kws is None:
        retry_kws = {
            "total": 5,
            "backoff_factor": 0.25,
            "status_forcelist": [500, 502, 503, 504]
        }

    # submit request
    def submit_request():
        r = requests.post(
            f"{UNIPROT_MAPPING_URL}/idmapping/run",
            data={
                "from": from_db,
                "to": to_db,
                "ids": ",".join(ids)
            },
        )
        r.raise_for_status()
        return r.json()["jobId"]

    def check_id_mapping_results_ready(job_id):
        while True:
            request = session.get(f"{UNIPROT_MAPPING_URL}/idmapping/status/{job_id}")
            request.raise_for_status()
            j = request.json()
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                    time.sleep(polling_interval)
                else:
                    raise Exception(j["jobStatus"])
            else:
                return bool(j["results"] or j["failedIds"])

    def get_id_mapping_results_link(job_id):
        url = f"{UNIPROT_MAPPING_URL}/idmapping/details/{job_id}"
        request = session.get(url)
        request.raise_for_status()
        return request.json()["redirectURL"]

    job_id = submit_request()

    retries = Retry(
        **retry_kws
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    if check_id_mapping_results_ready(job_id):
        result_url = get_id_mapping_results_link(job_id)

        # for simplicity, stream results (paginate requests to ensure more stable result,
        # rather than paginating response of very big unstable request):
        if "/stream/" not in result_url:
            result_url = result_url.replace(
                "/results/", "/results/stream/"
            )

        # append return format to URL
        result_url += f"?format={format}"

        # add isoform retrieval parameter if requested
        if isoforms:
            result_url += "&includeIsoform=true"

        result = session.get(result_url)
        result.raise_for_status()

        return result


def find_homologs(pdb_alignment_method="jackhmmer", **kwargs):
    """
    Identify homologs using jackhmmer or hmmbuild/hmmsearch

    Parameters
    ----------
    pdb_alignment_method : {"jackhmmer", "hmmsearch"}, 
             optional (default: "jackhmmer")
        Sequence alignment method used for searching the PDB
    **kwargs
        Passed into jackhmmer / hmmbuild_and_search protocol
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
    config = parse_config(HMMER_CONFIG)

    # update with overrides from kwargs
    config = {
        **config,
        **kwargs,
    }

    # create temporary output if no prefix is given
    if config["prefix"] is None:
        config["prefix"] = path.join(tempdir(), "compare")

    check_required(
        config, ["prefix"]
    )

    # run hmmsearch (possibly preceded by hmmbuild)
    if pdb_alignment_method == "hmmsearch":
        # set up config to run hmmbuild_and_search on the unfiltered alignment file
        updated_config = deepcopy(config)
        updated_config["alignment_file"] = config.get("raw_focus_alignment_file")
        ar = hmmbuild_and_search(**updated_config)

        # For hmmbuild and search, we have to read the raw focus alignment file
        # to guarantee that the query sequence is present
        with open(ar["raw_focus_alignment_file"]) as a:
            ali = Alignment.from_file(a, "fasta")

    # run jackhmmer against sequence database
    # at this point we have already checked to ensure
    # that the input is either jackhmmer or hmmsearch
    elif pdb_alignment_method == "jackhmmer":
        ar = jackhmmer_search(**config)

        with open(ar["raw_alignment_file"]) as a:
            ali = Alignment.from_file(a, "stockholm")

        # write alignment as FASTA file for easier checking by hand,
        # if necessary
        with open(config["prefix"] + "_raw.fasta", "w") as f:
            ali.write(f)
    else:
        raise InvalidParameterError(
            "Invalid pdb_alignment_method selected. Valid options are: " +
            ", ".join(["jackhmmer", "hmmsearch"])
        )

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
            "hmm_from": "hmm_start",
            "hmm_to": "hmm_end",
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
        # remove these
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
        the Uniprot sequence ranges do not map to a
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

        # download SIFTS table (gzip-compressed csv) to temp file
        temp_download_file = temp()
        get_urllib(SIFTS_URL, temp_download_file)

        # load table and rename columns for internal use, if SIFTS
        # ever decided to rename theirs
        table = pd.read_csv(
            temp_download_file, comment="#",
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

        # TODO: remove the following if new segment-based table proves as robust solution
        """
        # this block disabled for now due to use of new table
        # based on observed UniProt segments
        # - can probably be removed eventually

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

        # also disabled due to use of new table based on observed
        # UniProt segments - can probably be removed eventually 
        
        new_table = new_table.append(
            pd.DataFrame(res).loc[:, table.columns]
        )
        """

        # save for later reuse
        table.to_csv(sifts_table_file, index=False)

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

    def create_sequence_file(self, output_file, chunk_size=1000, max_retries=100):
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
        chunk_size : int, optional (default: 1000)
            Retrieve sequences from UniProt in chunks of this size
            (too large chunks cause the mapping service to stall)
        max_retries : int, optional (default: 100)
            Allow this many retries when fetching sequences
            from UniProt ID mapping service, which unfortunately
            often suffers from connection failures.
        """
        # unique identifiers *including* isoforms (used for filtering final table)
        ids = self.table.uniprot_ac.unique().tolist()

        # unique identifiers *excluding* isoforms (used for retrieval, to avoid
        # fetching sequences twice)
        ids_no_isoform = self.table.uniprot_ac.map(
            lambda id_: id_.split("-")[0]
        ).unique().tolist()

        # retrieve sequences in chunks since ID mapping service
        # tends to fail on large requests
        id_chunks = [
            ids_no_isoform[i:i + chunk_size] for i in range(0, len(ids_no_isoform), chunk_size)
        ]

        # store individual retrieved chunks as list of strings
        seq_chunks = []

        # keep track of how many retries were necessary and
        # abort if number exceeds max_retries
        num_retries = 0

        for i, ch in enumerate(id_chunks):
            # fetch sequence chunk;
            # if there is a problem retry as long as we stay within
            # maximum number of retries;
            # latest version also has limited number of retries inside the mapping function
            while True:
                try:
                    logging.info(f"Fetching chunk {i + 1} of {len(id_chunks)}...")
                    seqs = fetch_uniprot_mapping(ch).text
                    break
                except (requests.ConnectionError, requests.HTTPError) as e:
                    # count as failed try
                    num_retries += 1
                    logging.warning(
                        f"Error fetching chunk {i + 1} of {len(id_chunks)}, current retry #: {num_retries}"
                    )

                    # if we retried too often, abort
                    if num_retries > max_retries:
                        raise ResourceError(
                            "Could not fetch sequences for SIFTS mapping tables from UniProt since "
                            "maximum number of retries after connection errors was exceeded. Retry "
                            "at a later time, or call SIFTS.create_sequence_file() with a higher value "
                            "for max_retries."
                        ) from e

            # rename identifiers in sequence file, so
            # we can circumvent Uniprot sequence identifiers
            # being prefixed by hmmer if a hit has exactly the
            # same identifier as the query sequence
            seqs = seqs.replace(
                ">sp|", ">evsp|",
            ).replace(
                ">tr|", ">evtr|",
            )

            assert seqs.endswith("\n")

            # store for parsing and output
            seq_chunks.append(seqs)

        # create set representation for efficient lookup
        ids_set = set(ids)

        # only retain sequences that we actually had in input list
        # (API will return all isoforms)
        filtered_seqs = [
            (seq_id, seq) for seq_id, seq in read_fasta(
                StringIO("".join(seq_chunks))
            )
            if seq_id.split("|")[1] in ids_set
        ]

        # store sequences to FASTA file in one go at the end
        with open(output_file, "w") as f:
            write_fasta(filtered_seqs, f)

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
            Defines the behaviour of find_homologs() function
            used to find homologs by sequence alignment:
            - which alignment method is used 
              (pdb_alignment_method: {"jackhmmer", "hmmsearch"}, 
              default: "jackhmmer"),
            - parameters passed into the protocol for the selected
              alignment method (evcouplings.align.jackhmmer_search or
              evcouplings.align.hmmbuild_and_search).
              
              Default parameters are set in the HMMER_CONFIG string in this
              module, other parameters will need to be overriden; these
              minimally are:
              - for pdb_alignment_method == "jackhmmer":
                - sequence_id : str, identifier of target sequence
                - jackhmmer : str, path to jackhmmer binary if not on path                
              - for pdb_alignment_method == "hmmsearch":
                - sequence_id : str, identifier of target sequence
                - raw_focus_alignment_file : str, path to input alignment file  
                - hmmbuild : str, path to hmmbuild binary if not on path
                - hmmsearch : str, path to search binary if not on path
            - additionally, if "prefix" is given,
              individual mappings will be saved to files suffixed
              by the respective key in mapping table.

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

        ali, hits = find_homologs(
            sequence_database=self.sequence_file, 
            **kwargs
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
            agg_df_raw = []
            # go through each segment
            for j, r in grp.iterrows():
                # compute mapping for that particular segment
                map_j, map_j_df = _create_mapping(r)

                # add to overall mapping dictionary for this hit
                agg_mapping.update(map_j)
                agg_df_raw.append(map_j_df)

            agg_df = pd.concat(agg_df_raw)

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

        # split residue name into residue number and (potential) insertion code
        # for proper sorting
        def _split_insertion_code(res):
            # safeguard function by typecasting to string (in some instances
            # this turned out to be an integer, presumably due to pandas automatic
            # typecast)
            res = str(res)

            # should not happen but just to be safe
            assert len(res) >= 1

            # check if we have an insertion code or not
            if res[-1].isalpha():
                return int(res[:-1]), res[-1]
            else:
                return int(res), ""

        # now group again, to aggregate full hit dataframe;
        # note that coord_start and coord_end are strings that may contain
        # insertion codes so need to treat accordingly
        def _agg_type(x):
            if x == "overlap":
                return "sum"
            elif x == "coord_start":
                return lambda l: sorted(l, key=_split_insertion_code)[0]
            elif x == "coord_end":
                return lambda l: sorted(l, key=_split_insertion_code)[-1]
            elif x.endswith("_start"):
                return "min"
            elif x.endswith("_end"):
                return "max"
            else:
                return "first"

        agg_types = OrderedDict(
            [(c, _agg_type(c)) for c in hits.columns
             if c not in hit_columns]
        )

        # only aggregate if we have anything to aggregate,
        # otherwise pandas drops the index columns
        # alignment_id, pdb_id, pdb_chain and things go
        # wrong horribly in the following join
        if len(hits) > 0:
            hits_grouped = hits.groupby(
                hit_columns
            ).agg(agg_types).reset_index()
        else:
            hits_grouped = hits

        # join with mapping information
        hits_grouped = hits_grouped.merge(
            mapping_df, on=hit_columns
        )

        # remove hits with too little residue coverage
        hits_grouped = hits_grouped.query("overlap >= @min_overlap")

        hits_grouped = hits_grouped.assign(
            bitscore=pd.to_numeric(hits_grouped.bitscore, errors="coerce")
        ).sort_values(
            by="bitscore", ascending=False
        )

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
