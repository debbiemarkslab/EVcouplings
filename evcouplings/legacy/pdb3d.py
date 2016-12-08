"""
Library to handle basic 3D structure calculations
 
 (1) PDB file distance calculations
 (2) DSSP annotations
 (3) index mapping using SIFTS
 (4) PDB file modification

Thomas A. Hopf, 31.01.2015

"""
from os import system
from tempfile import mkstemp
from os import path
from collections import defaultdict, namedtuple

import numpy
from scipy.spatial.distance import cdist

_DIST_FORMAT = "{:.3f}"
_NA_VALUE = "-"

_PDB_DOWNLOAD_CMD = "curl -s http://files.rcsb.org/view/{pdb_id}.pdb > {outfile}"
_SIFTS_DOWNLOAD_CMD = "curl -s ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{pdb_id}.xml.gz | zcat > {outfile}"

DSSP_BINARY = "../binaries/DSSP_MAC.EXE"
# extracted from Bio.PDB.DSSP.MAX_ACC and changed to one letter code
MAX_ACC = {'C': 135.0, 'D': 163.0, 'S': 130.0, 'N': 157.0, 'Q': 198.0, 'K': 205.0, 
           'I': 169.0, 'P': 136.0, 'T': 142.0, 'F': 197.0, 'A': 106.0, 'G': 84.0, 
           'H': 184.0, 'L': 164.0, 'R': 248.0, 'W': 227.0, 'V': 142.0, 'E': 194.0, 
           'Y': 222.0, 'M': 188.0}

# pre-defined distance and filter functions

def filter_water(het, resseq, icode):
    """
    Filter water heteroatoms
    """
    return het == "W"

def filter_heteroatoms(het, resseq, icode):
    """
    Filter all heteroatomss
    """
    return het != " "

def minimum_atom_distance(residue_i, residue_j):
    """
    Calculate minimum atom distance between two residues
    """
    return numpy.min(cdist(residue_i.coords, residue_j.coords, 'euclidean'))

def atom_distance(residue_i, residue_j, atom_type="CA"):
    """
    Calculate atom-atom distance between two residues 
    (e.g. C_alpha - C_alpha)
    """
    return 0 # TODO


# pre-defined selectors

def single_chain(chain, filter_=filter_heteroatoms, model=0):
    """
    Single-chain selector
    """   
    return multi_chain([chain], filter_, model)

def multi_chain(chains, filter_=filter_heteroatoms, model=0):
    """
    Multi-chain selector
    """
    return {c:(model, c, filter_) for c in chains}


# 3D distance calculation on single PDB structures

Residue = namedtuple('Residue', ['name', 'atoms', 'coords'])

def calculate_distances(pdb_structure, selectors, 
                        distance_function=minimum_atom_distance):
    """
    Calculate all-against all distances between objects defined by selectors
    """
    from itertools import combinations_with_replacement
 
    # create molecule selections and store coordinates in NumPy arrays
    # selector_coordinates = defaultdict()
    selector_residues = defaultdict(list)
    selector_res_names = {}

    for selector, (model, chain, filter_) in selectors.items():
        # get all residues/molecules in selected model and chain
        chain_in_model = pdb_structure[model][chain]

        # exclude heteroatom flag from sorting
        residue_list = [res.get_id() for res in chain_in_model]
        selector_res_names[selector] = {
            res:chain_in_model[res].get_resname() for res in residue_list
        }
        
        # apply filter function if given
        if filter_ is not None:
            residue_list = [(het, resseq, icode) for (het, resseq, icode) 
                            in residue_list if not filter_(het, resseq, icode)
            ]

        # compile atom coordinates in one NumPy array per residue
        for res in residue_list:
            atom_names = [atom.get_name() for atom in chain_in_model[res]]
            atom_coordinates = numpy.array([atom.get_coord() for atom 
                                            in chain_in_model[res]]
            )
            selector_residues[selector].append(
                Residue(res, atom_names, atom_coordinates)
            )

    # compare all possible combinations of selectors
    distances = defaultdict(lambda: defaultdict())
    for s_1, s_2 in combinations_with_replacement(selectors, r=2):
        residues_s_1 = selector_residues[s_1]
        residues_s_2 = selector_residues[s_2]
        
        # iterate all pairs of residues between selectors
        for r_1 in residues_s_1:
            for r_2 in residues_s_2:
                if s_1 != s_2 or r_1.name < r_2.name:
                    distances[(s_1, s_2)][(r_1.name, r_2.name)] = (
                        distance_function(r_1, r_2)
                    )

    return distances, selector_res_names

def load_structure(pdb_file):
    """
    Load and parse PDB file into struture object
    """
    from Bio.PDB.PDBParser import PDBParser
    parser = PDBParser(QUIET=True)
    return parser.get_structure("", pdb_file) 

# SIFTS file parsing

ResidueMapping = namedtuple('ResidueMapping', 
    ['pdb_chain', 'pdb_index', 'pdb_residue', 
     'uniprot_index', 'uniprot_residue', 'uniprot_ac']
)


def df_to_pdb3d_map(df, pdb_chain, uniprot_sequence=_NA_VALUE):
    """
    Transform DataFrame index mapping from number_mapping module
    into pdb3d index mapping format.
    """
    mapping = {}

    for i, row in df.dropna().iterrows():
        mapping[(pdb_chain, str(row["query_pos"]))] = ResidueMapping(
            pdb_chain, str(row["query_pos"]), row["query_pdb_res"],
            int(row["target_pos"]), row["target_res"], uniprot_sequence
        )

    return mapping


def parse_sifts_mapping(sifts_file):
    """
    Extract SIFTS mapping
    """
    from xml.dom import minidom
    xmldoc = minidom.parse(sifts_file)

    uniprot_ac_id_mapping = {}

    # identify Uniprot AC/IDs used in mapping
    for r in xmldoc.getElementsByTagName('mapRegion'):
        for db in r.getElementsByTagName("db"):
            # print db.attributes["dbAccessionId"].value
            # print db.attributes.keys()
            if ("dbSource" in db.attributes.keys() and 
                    db.attributes["dbSource"].value == "UniProt" and
                    "dbAccessionId" in db.attributes.keys()
                ):
                for detail in r.getElementsByTagName("dbDetail"):
                    if ("dbSource" in detail.attributes.keys() and 
                            detail.attributes["dbSource"].value == "UniProt" and
                            "property" in detail.attributes.keys() and 
                            detail.attributes["property"].value == "secondaryId"
                        ):
                        uniprot_id = detail.childNodes[0].data
                        uniprot_ac = db.attributes["dbAccessionId"].value
                        uniprot_ac_id_mapping[uniprot_ac] = uniprot_id
    
    # Iterate over SIFTS mapping for all chains/ residues 
    mapping = {}
    itemlist = xmldoc.getElementsByTagName('residue') 
    
    for s in itemlist:
        cross_refs = s.getElementsByTagName('crossRefDb') 
        pdb_res_num, pdb_res_name, pdb_chain = None, None, None
        up_res_num, up_res_name, up_ac_number = None, None, None
        
        for r in cross_refs:
            if r.attributes['dbSource'].value == "PDB":
                pdb_res_num = r.attributes['dbResNum'].value
                pdb_res_name = r.attributes['dbResName'].value
                pdb_chain = r.attributes['dbChainId'].value
            elif r.attributes['dbSource'].value == "UniProt":
                up_res_num = r.attributes['dbResNum'].value
                up_res_name = r.attributes['dbResName'].value
                up_ac_number = r.attributes['dbAccessionId'].value

        # print pdb_res_num, pdb_chain, pdb_res_name, "  ", \
        #       up_res_num, up_res_name, up_ac_number
        if pdb_res_num is not None and up_res_num is not None:
            mapping[(pdb_chain, pdb_res_num)] = ResidueMapping(
                pdb_chain, pdb_res_num, pdb_res_name, 
                up_res_num, up_res_name, up_ac_number
            )

    return mapping

# contact map generation

def _retrieve_file(outfile, pdb_id, fetch_cmd):
    """
    Download PDB/SIFTS file to temporary file or given file path if file does 
    not yet exist.
    """
    if outfile is None:
        outfile_handle, outfile = mkstemp()
        system(fetch_cmd.format(pdb_id=pdb_id.lower(), outfile=outfile))
    elif not path.exists(outfile):
        system(fetch_cmd.format(pdb_id=pdb_id.lower(), outfile=outfile))

    return outfile

def _map_residue(mapping, pdb_res_name, het, resi, icode, chain):
    """
    Caveat:
    Does not check if residue to be mapped is a HETATM,
    s.t. things like MSE can be still mapped to Uniprot.

    To cover the (unlikely, see link below) case, of a 
    numbering overlap between residues and HETATM residues,
    we check if the SIFTS PDB residue name and the BioPython
    residue name agree.

    See http://www.biopython.org/pipermail/biopython-dev/2011-January/008647.html
    for discussion of how likely this problem is to occur
    (apparently very unlikely after PDB cleanup).
    """
    het = het.replace(" ", "")
    pdb_res_name = pdb_res_name.replace(" ", "")
    is_res = (het == "")
    pdb_index = str(resi) + icode.strip()

    if ((chain, pdb_index) in mapping and
            mapping[(chain, pdb_index)].pdb_residue == pdb_res_name):
        map_up = mapping[(chain, pdb_index)]

        return (map_up.uniprot_index, 
                map_up.uniprot_residue,
                map_up.uniprot_ac,
                pdb_index, pdb_res_name,
                int(is_res)
        )
    else:
        return (_NA_VALUE, _NA_VALUE, _NA_VALUE, pdb_index, 
                pdb_res_name, int(is_res)
        )

def make_contact_map(pdb_id, selectors, out_file,
                     pdb_file_name=None, sifts_file_name=None, 
                     use_sifts=True, verbose=True,
                     distance_function=minimum_atom_distance,
                     custom_mapping=None):
    """
    Calculate contact map using given selectors and store to out_file

    The resulting file will be ordered by BioPython PDB residue IDs, 
    so make sure to resort table before assuming any ordering in the 
    numbering.    
    """
    pdb_file_name = _retrieve_file(pdb_file_name, pdb_id, _PDB_DOWNLOAD_CMD)
    
    #  get SIFTS mapping if to be used
    if use_sifts:
        sifts_file_name = _retrieve_file(sifts_file_name, pdb_id, _SIFTS_DOWNLOAD_CMD)
        mapping = parse_sifts_mapping(sifts_file_name)
    elif custom_mapping is not None:
        mapping = custom_mapping
    else:
        mapping = {}
    
    if verbose:
        print("PDB file:", pdb_file_name)
        print("SIFTS file:", sifts_file_name)

    # calculate distances
    dists, selector_res_names = calculate_distances(
        load_structure(pdb_file_name), selectors, distance_function
    )
    
    # write data to contact map file
    if out_file is not None:
        with open(out_file, "w") as f:
            print(
                " ".join(
                ["sel_i", "up_index_i", "res_i", "up_ac_i", 
                 "pdb_index_i", "pdb_res_i", "is_res_i", 
                 "sel_j", "up_index_j", "res_j", "up_ac_j", 
                 "pdb_index_j", "pdb_res_j", "is_res_j", 
                 "dist"]), file=f
            )

            # iterate all selectors
            for (s_1, s_2), selector_dists in dists.items():
                (_, chain_i, _) = selectors[s_1]
                (_, chain_j, _) = selectors[s_2]
                res_names_1 = selector_res_names[s_1]
                res_names_2 = selector_res_names[s_2]

                # iterate all residue pairs for current pair of selectors
                for (res_i, res_j), dist in sorted(selector_dists.items()):
                    (het_i, resi_i, icode_i) = res_i
                    (het_j, resi_j, icode_j) = res_j
                    mapped_i = _map_residue(mapping, res_names_1[res_i], 
                                            het_i, resi_i, icode_i, chain_i
                    )
                    mapped_j = _map_residue(mapping, res_names_2[res_j],
                                            het_j, resi_j, icode_j, chain_j
                    )

                    print(
                        s_1, " ".join(map(str, mapped_i)),
                        s_2, " ".join(map(str, mapped_j)),
                        _DIST_FORMAT.format(dist),
                        file=f
                    )
   
    return dists, mapping

def read_results(filename, na_values=None):
    from pandas import read_csv
    return read_csv(filename, sep=" ", na_values=na_values, comment="#")

# secondary structure / solvent accessibility calculations

def _map_residue_dssp(mapping, pdb_res_name, het, resi, icode, chain):
    """
    Maps DSSP output to Uniprot with SIFTS 
    (different to contact map code because no hetatm check and
    one-letter code amino acids used by BioPython).
    """
    pdb_res_name = pdb_res_name.replace(" ", "")
    pdb_index = str(resi) + icode.strip()

    if (chain, pdb_index) in mapping:
        map_up = mapping[(chain, pdb_index)]

        return (map_up.uniprot_index, 
                map_up.uniprot_residue,
                map_up.uniprot_ac,
                pdb_index, pdb_res_name
        )
    else:
        return (_NA_VALUE, _NA_VALUE, _NA_VALUE, pdb_index, 
                pdb_res_name
        )

def make_dssp_annotation(pdb_id, out_file, pdb_file_name=None, 
                         sifts_file_name=None, use_sifts=True,
                         verbose=True, dssp_binary=DSSP_BINARY,
                         custom_mapping=None):
    """
    Calculate DSSP annotation for PDB file
    """
    from Bio.PDB.DSSP import dssp_dict_from_pdb_file

    pdb_file_name = _retrieve_file(pdb_file_name, pdb_id, _PDB_DOWNLOAD_CMD)

    #  get SIFTS mapping if to be used
    if use_sifts:
        sifts_file_name = _retrieve_file(sifts_file_name, pdb_id, _SIFTS_DOWNLOAD_CMD)
        mapping = parse_sifts_mapping(sifts_file_name)
    elif custom_mapping is not None:
        mapping = custom_mapping
    else:
        mapping = {}

    dssp_dict, dssp_dict_keys = dssp_dict_from_pdb_file(pdb_file_name, DSSP=dssp_binary)

    if verbose:
        print("PDB file:", pdb_file_name)
        print("SIFTS file:", sifts_file_name)

    if out_file is not None:
        with open(out_file, "w") as f:
            print(
                " ".join(
                    ["up_index", "res", "uniprot_ac", 
                     "pdb_index", "pdb_res", "pdb_chain",
                     "ss", "acc", "rel_acc", "phi", "psi"]
                ), file=f
            )

            for res, values in sorted(dssp_dict.items()):
                (chain, (het, resi, icode)) = res
                res_letter, ss, acc, phi, psi = dssp_dict[res][0:5]
                rel_acc = ("{:.2f}".format(acc/MAX_ACC[res_letter]) 
                           if res_letter in MAX_ACC else "-")

                print(
                    " ".join(map(str, _map_residue_dssp(mapping, res_letter, 
                                         het, resi, icode, chain))),
                    end="", file=f
                )
                print(chain, file=f, end="")
                print(ss, acc, rel_acc, phi, psi, file=f)
            
    return dssp_dict, mapping

# PDB file modification

def renumber_pdb_file(pdb_file, output_pdb_file, mapping, eliminate=False, 
                      chains_to_keep=None, keep_hetatm=True):
    """
    Remap residue indices in a PDB file
    
    mapping: {chain: {pdb_residue: new_residue_index}}
    eliminate: delete any residue that is not contained in mapping
    keep_chains: chains that will not be deleted if eliminate==True
    keep_hetatm: keep all HETATMs (e.g. ligands)
    """
    # make sure all residue keys in mapping are strings
    mapping_ = {}
    for chain, chain_mapping in mapping.items():
        mapping_[str(chain)] = {str(pdb_res): str(to_res) 
                                for (pdb_res, to_res) in chain_mapping.items()}
      
    with open(pdb_file) as f:
        with open(output_pdb_file, "w") as fo:
            for line in f:
                line = line.rstrip()
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    resi, chain = line[22:26].strip(), line[21].strip()
                    if chain in mapping_ and resi in mapping_[chain]:
                        resi_mapped = mapping_[chain][resi]
                        print("{}{:>4}{}".format(line[0:22], resi_mapped, line[26:]), file=fo)
                    else:
                        if (not eliminate 
                            or (chains_to_keep is not None and chain in chains_to_keep)
                            or (keep_hetatm and line.startswith("HETATM"))):
                            print(line, file=fo)
                else:
                    # secondary structure will be broken so skip those lines
                    if not line.startswith("HELIX") and not line.startswith("SHEET"):
                        print(line, file=fo)


def renumber_pdb_to_uniprot(pdb_id, output_pdb_file, chains=None,
                            pdb_file_name=None, sifts_file_name=None,
                            eliminate=True):
    """
    Renumber PDB to Uniprot using renumber_pdb_file function.
    """
    from sys import stderr

    pdb_file_name = _retrieve_file(pdb_file_name, pdb_id, _PDB_DOWNLOAD_CMD)
    sifts_file_name = _retrieve_file(sifts_file_name, pdb_id, _SIFTS_DOWNLOAD_CMD)
    mapping = parse_sifts_mapping(sifts_file_name)

    chain_to_mapping = defaultdict(dict)
    chain_to_uniprot_ac = defaultdict(set)

    for (chain, res_index), res_map in mapping.items():
        # skip chains if subset list given
        if chains is not None and chain not in chains:
            continue

        chain_to_uniprot_ac[chain].add(res_map.uniprot_ac)
        chain_to_mapping[chain][res_map.pdb_index] = res_map.uniprot_index

    renumber_pdb_file(pdb_file_name, output_pdb_file, chain_to_mapping, eliminate=eliminate)

    for chain, acs in chain_to_uniprot_ac.items():
        if len(acs) > 1:
            print("Warning: chain {} contains multiple Uniprot ACs:".format(chain), ",".join(acs), file=stderr)
