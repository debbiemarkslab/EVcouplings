"""
Functions for structure prediction using CNSsolve 1.21

Authors:
  Thomas A. Hopf
"""

import os
from os import path
import pandas as pd
from pkg_resources import resource_filename

from evcouplings.fold.restraints import (
    ec_dist_restraints, secstruct_dist_restraints, secstruct_angle_restraints
)
from evcouplings.fold.tools import run_cns
from evcouplings.utils.config import InvalidParameterError
from evcouplings.utils.constants import AA1_to_AA3
from evcouplings.utils.helpers import render_template
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources, temp, valid_file
)


def cns_seq_file(sequence, output_file=None, residues_per_line=16):
    """
    Generate a CNS .seq file for a given protein sequence

    Parameters
    ----------
    sequence : str
        Amino acid sequence in one-letter code
    output_file : str, optional (default: None)
        Save 3-letter code sequence to this file
        (if None, will create temporary file)
    residues_per_line : int, optional (default: 16)
        Print this many residues on each line
        of .seq file

    Returns
    -------
    output_file : str
        Path to file with sequence
        (useful if temporary file was
        generated)

    Raises
    ------
    InvalidParameterError
        If sequence contains invalid symbol
    """
    if output_file is None:
        output_file = temp()

    with open(output_file, "w") as f:
        # split sequence into parts per line
        lines = [
            sequence[i: i + residues_per_line]
            for i in range(0, len(sequence), residues_per_line)
        ]

        # go through lines and transform into 3-letter code
        for line in lines:
            try:
                l3 = " ".join(
                    [AA1_to_AA3[aa] for aa in line]
                )
            except KeyError as e:
                raise InvalidParameterError(
                    "Invalid amino acid could not be mapped"
                ) from e

            f.write(l3 + "\n")

    return output_file


def _cns_render_template(template_name, mapping):
    """
    Render an included CNS template .inp

    Parameters
    ----------
    template_name : str
        Name of CNS template (e.g. dg_sa)
    mapping : dict
        Values to be substituted into template

    Returns
    -------
    str
        Rendered template
    """
    # get path of template within package
    template_file = resource_filename(
        __name__, "cns_templates/{}.inp".format(template_name)
    )

    verify_resources(
        "CNS template does not exist: {}".format(template_file),
        template_file
    )

    return render_template(template_file, mapping)


def cns_mtf_inp(seq_infile, mtf_outfile, first_index=1, disulfide_bridges=None):
    """
    Create CNS input script (.inp) to create molecular
    topology file (.mtf) from sequence (.seq file)

    Parameters
    ----------
    seq_infile : str
        Path to .seq input file (create using cns_seq_file())
    mtf_outfile : str
        Path where generated .mtf file should be stored
    first_index : int, optional (default: 1)
        Index of first residue in sequence
    disulfide_bridges: list or pandas.DataFrame, optional (default: None)
        Position pairs that should be linked by a disulfide
        bridge. Can be:

        * list of tuples (i, j)

        * dataframe with columns i and j for positions, and A_i and A_j
          for amino acid symbols. Will automatically select those pairs
          (i, j) where A_i and A_j are 'C'.

    Returns
    -------
    str:
        Input script
    """
    # determine if disulfide bridge information will go into input script or not
    if disulfide_bridges is None:
        disulfides = []
    else:
        # if dataframe, extract (i, j) pairs where both residues are cysteine
        if isinstance(disulfide_bridges, pd.DataFrame):
            cys_pairs = disulfide_bridges.query("A_i == 'C' and A_j == 'C'")
            pair_list = zip(cys_pairs.i, cys_pairs.j)
        else:
            pair_list = disulfide_bridges

        # add index from 1 to list, since template needs running index
        # for fields
        disulfides = [
            (idx, i, j) for idx, (i, j) in enumerate(pair_list, start=1)
        ]

    return _cns_render_template(
        "generate_seq",
        {
            "renumber_index": first_index,
            "sequence_infile": seq_infile,
            "mtf_outfile": mtf_outfile,
            "disulfide_list": disulfides
        }
    )


def cns_extended_inp(mtf_infile, pdb_outfile):
    """
    Create CNS iput script (.inp) to create extended PDB file
    from molecular topology file (.mtf)

    Parameters
    ----------
    mtf_infile : str
        Path to .mtf topology file
    pdb_outfile : str
        Path where extended .pdb file will be stored

    Returns
    -------
    str:
        Input script
    """
    return _cns_render_template(
        "generate_extended",
        {
            "mtf_infile": mtf_infile,
            "pdb_outfile": pdb_outfile,
        }
    )


def cns_dgsa_inp(pdb_infile, mtf_infile, outfile_prefix,
                 ec_pair_tbl_infile, ss_dist_tbl_infile,
                 ss_angle_tbl_infile, num_structures=20,
                 log_level="quiet"):
    """
    Create CNS iput script (.inp) to fold extended PDB file
    using distance geometry and simulated annealing with
    distance and dihedral angle constraints

    Parameters
    ----------
    pdb_infile : str
        Path to extended PDB structure that will be folded
    mtf_infile : str
        Path to molecular topology file corresponding to
        pdb_infile
    outfile_prefix : str
        Prefix of output files
    ec_pair_tbl_infile:
        Path to .tbl file with distance restraints for
        EC pairs
    ss_dist_tbl_infile : str
        Path to .tbl file with distance restraints for
        secondary structure
    ss_angle_tbl_infile : str
        Path to .tbl file with dihedral angle restraints
        for secondary structure
    num_structures : int, optional (default: 20)
        Number of trial structures to generate
    log_level : str, optional (default: "quiet")
        Log output level of CNS. Set to "verbose" to obtain
        information about restraint violations.

    Returns
    -------
    str:
        Input script
    """
    return _cns_render_template(
        "dg_sa",
        {
            "pdb_infile": pdb_infile,
            "mtf_infile": mtf_infile,
            "num_structures": num_structures,
            "ec_pair_tbl_infile": ec_pair_tbl_infile,
            "ss_dist_tbl_infile": ss_dist_tbl_infile,
            "ss_angle_tbl_infile": ss_angle_tbl_infile,
            "pdb_outfile_basename": outfile_prefix,
            "hbond_tbl_infile": "",
            "log_level": log_level,
            "md_cool_noe_scale_factor": 5,
            "ss_dist_noe_avg_mode": "cent",
            "ec_pair_noe_avg_mode": "cent",
        }
    )


def cns_generate_easy_inp(pdb_infile, pdb_outfile, mtf_outfile):
    """
    Create CNS input script (.inp) to run generate_easy
    protocol (here, to add hydrogen bonds to models)

    Parameters
    ----------
    pdb_infile : str
        Path to 3D structure to which hydrogens will be added
    pdb_outfile : str
        Path where to to store updated structure
    mtf_outfile : str
        Path where to store molecular topology file corresponding
        to updated structure

    Returns
    -------
    str:
        Input script
    """
    return _cns_render_template(
        "generate_easy",
        {
            "pdb_infile": pdb_infile,
            "mtf_outfile": mtf_outfile,
            "pdb_outfile": pdb_outfile,
            "hydrogen_flag": "true",
            "pdb_o_format": "false",
            "ile_cd_becomes": "",  # default: CD1
            "ot1_becomes": "",     # default: O
            "ot2_becomes": "",     # default: OXT
        }
    )


def cns_minimize_inp(pdb_infile, mtf_infile, pdb_outfile, num_cycles=5):
    """
    Create CNS input script (.inp) to minimize model

    Parameters
    ----------
    pdb_infile : str
        Path to PDB structure that should be minimized
        (created using generate_easy protocol)
    mtf_infile : str
        Path to corresponding .mtf topology file of
        PDB structure (created using generate_easy protocol)
    pdb_outfile : str
        Path where minimized structure will be stored
    num_cycles : int, optional (default: 5)
        Number of minimization cycles

    Returns
    -------
    str:
        Input script
    """
    return _cns_render_template(
        "model_minimize",
        {
            "pdb_infile": pdb_infile,
            "mtf_infile": mtf_infile,
            "pdb_outfile": pdb_outfile,
            "num_cycles": num_cycles,
            "use_cryst": "false",
            "space_group": "",
        }
    )


def cns_dist_restraint(resid_i, atom_i, resid_j, atom_j,
                       dist, lower, upper, weight=None,
                       comment=None):
    """
    Create a CNS distance restraint string

    Parameters
    ----------
    resid_i : int
        Index of first residue
    atom_i : str
        Name of selected atom in first residue
    resid_j : int
        Index of second residue
    atom_j : str
        Name of selected atom in second residue
    dist : float
        Restrain distance between residues to this value
    lower : float
        Lower bound delta on distance
    upper : float
        Upper bound delta on distance
    weight : float, optional (default: None)
        Weight for distance restraint
    comment : str, optional (default: None)
        Print comment at the end of restraint line

    Returns
    -------
    r : str
        Distance restraint
    """
    if weight is not None:
        weight_str = "weight {} ".format(weight)
    else:
        weight_str = ""

    if comment is not None:
        comment_str = "! {}".format(comment)
    else:
        comment_str = ""

    r = (
        "assign (resid {} and name {}) (resid {} and name {})  "
        "{} {} {} {}{}".format(
            resid_i, atom_i, resid_j, atom_j, dist, lower, upper,
            weight_str, comment_str
        )
    )

    return r


def cns_dihedral_restraint(resid_i, atom_i, resid_j, atom_j,
                           resid_k, atom_k, resid_l, atom_l,
                           energy_constant, degrees, range,
                           exponent, comment=None):
    """
    Create a CNS dihedral angle restraint string

    Parameters
    ----------
    resid_i : int
        Index of first residue
    atom_i : str
        Name of selected atom in first residue
    resid_j : int
        Index of second residue
    atom_j : str
        Name of selected atom in second residue
    resid_k : int
        Index of third residue
    atom_k : str
        Name of selected atom in third residue
    resid_l : int
        Index of fourth residue
    atom_l : str
        Name of selected atom in fourth residue
    energy_constant : float
        Energy constant for restraint
    degrees : float
        Restrain angle to this value
    range : float
        Acceptable range around angle
    exponent : int
        Exponent of dihedral angle energy
    comment : str, optional (default: None)
        Print comment at the end of restraint line

    Returns
    -------
    r : str
        Dihedral angle restraint
    """
    if comment is not None:
        comment_str = " ! {}".format(comment)
    else:
        comment_str = ""

    r = (
        "assign (resid {} and name {}) (resid {} and name {}) "
        "(resid {} and name {}) (resid {} and name {})"
        "  {} {} {} {}{}".format(
            resid_i, atom_i, resid_j, atom_j,
            resid_k, atom_k, resid_l, atom_l,
            energy_constant, degrees, range,
            exponent, comment_str
        )
    )

    return r


def cns_dgsa_fold(residues, ec_pairs, prefix, config_file=None,
                  secstruct_column="sec_struct_3state",
                  num_structures=20, min_cycles=5,
                  log_level=None, binary="cns"):
    """
    Predict 3D structure coordinates using distance geometry
    and simulated annealing-based folding protocol
     
    Parameters
    ----------
    residues : pandas.DataFrame
        Table containing positions (column i), residue
        type (column A_i), and secondary structure for
        each position
    ec_pairs : pandas.DataFrame
        Table with EC pairs that will be turned
        into distance restraints
        (with columns i, j, A_i, A_j)
    prefix : str
        Prefix for output files (can include directories).
        Folders will be created automatically.
    config_file : str, optional (default: None)
        Path to config file with folding settings. If None,
        will use default settings included in package
        (restraints.yml)
    secstruct_column : str, optional (default: sec_struct_3state)
        Column name in residues dataframe from which secondary
        structure will be extracted (has to be H, E, or C).
    num_structures : int, optional (default: 20)
        Number of trial structures to generate
    min_cycles : int, optional (default: 5)
        Number of minimization cycles at end of protocol
    log_level : {None, "quiet", "verbose"}, optional (default: None)
        Don't keep CNS log files, or switch to different degrees
        of verbosity ("verbose" needed to obtain violation information)
    binary : str, optional (default: "cns")
        Path of CNS binary

    Returns
    -------
    final_models : dict
        Mapping from model name to path of model
    """
    def _run_inp(inp_str, output_prefix):
        with open(output_prefix + ".inp", "w") as f:
            f.write(inp_str)

        if log_level is not None:
            log_file = output_prefix + ".log"
        else:
            log_file = None

        run_cns(inp_str, log_file=log_file, binary=binary)

    # make sure output directory exists
    create_prefix_folders(prefix)

    # CNS doesn't like paths above a certain length, so we
    # will change into working directory and keep paths short.
    # For this reason, extract path and filename prefix
    dir_, rootname = path.split(prefix)
    cwd = os.getcwd()

    if dir_ != "":
        os.chdir(dir_)

    # create restraints (EC pairs and secondary structure-based)
    ec_tbl = rootname + "_couplings.tbl"
    ss_dist_tbl = rootname + "_ss_distance.tbl"
    ss_angle_tbl = rootname + "_ss_angle.tbl"

    ec_dist_restraints(
        ec_pairs, ec_tbl, cns_dist_restraint,
        config_file
    )

    secstruct_dist_restraints(
        residues, ss_dist_tbl, cns_dist_restraint,
        config_file, secstruct_column
    )

    secstruct_angle_restraints(
        residues, ss_angle_tbl, cns_dihedral_restraint,
        config_file, secstruct_column
    )

    # create sequence file
    seq = "".join(residues.A_i)
    seq_file = rootname + ".seq"
    cns_seq_file(seq, seq_file)

    # set up input files for folding
    # make molecular topology file (will be written to mtf_file)
    mtf_file = rootname + ".mtf"
    _run_inp(
        cns_mtf_inp(
            seq_file, mtf_file, first_index=residues.i.min(),
            disulfide_bridges=None
        ), mtf_file
    )

    # make extended PDB file (will be in extended_file)
    extended_file = rootname + "_extended.pdb"
    _run_inp(
        cns_extended_inp(
            mtf_file, extended_file
        ), extended_file
    )

    # fold using dg_sa protocol (filenames will have suffixes _1, _2, ...)

    # have to pass either quiet or verbose to CNS (but will not store
    # log file if log_level is None).
    if log_level is None:
        dgsa_log_level = "quiet"
    else:
        dgsa_log_level = log_level

    _run_inp(
        cns_dgsa_inp(
            extended_file, mtf_file, rootname,
            ec_tbl, ss_dist_tbl, ss_angle_tbl,
            num_structures=num_structures,
            log_level=dgsa_log_level
        ), rootname + "_dgsa"
    )

    # add hydrogen atoms and minimize (for all
    # generated candidate structures from dg_sa)

    # keep track of final predicted structures
    final_models = {}

    for i in range(1, num_structures + 1):
        input_root = "{}_{}".format(rootname, i)
        input_model = input_root + ".pdb"

        # check if we actually got the model from dg_sa
        if not valid_file(input_model):
            continue

        # run generate_easy protocol to add hydrogen atoms
        easy_pdb = input_root + "_h.pdb"
        easy_mtf = input_root + "_h.mtf"
        _run_inp(
            cns_generate_easy_inp(
                input_model, easy_pdb, easy_mtf
            ), input_root + "_h"
        )

        # then minimize
        min_pdb = input_root + "_hMIN.pdb"

        _run_inp(
            cns_minimize_inp(
                easy_pdb, easy_mtf, min_pdb,
                num_cycles=min_cycles
            ), input_root + "_hMIN"
        )

        if valid_file(min_pdb):
            final_models[min_pdb] = path.join(
                dir_, min_pdb
            )

    # change back into original directory
    os.chdir(cwd)

    return final_models
