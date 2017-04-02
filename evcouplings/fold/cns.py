"""
Functions for structure prediction using CNSsolve 1.21

Authors:
  Thomas A. Hopf
"""

from pkg_resources import resource_filename

import pandas as pd

from evcouplings.utils.config import (
    read_config_file, InvalidParameterError
)
from evcouplings.utils.constants import AA1_to_AA3
from evcouplings.utils.helpers import render_template
from evcouplings.utils.system import verify_resources


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


def _folding_config(config_file=None):
    """
    Load CNS folding configuration
    
    Parameters
    ----------
    config_file: str, optional (default: None)
        Path to configuration file. If None,
        loads default configuration included
        with package.

    Returns
    -------
    dict
        Loaded configuration
    """
    if config_file is None:
        # get path of config within package
        config_file = resource_filename(
            __name__, "cns_templates/restraints.yml"
        )

    # check if config file exists and read
    verify_resources(
        "Folding config file does not exist or is empty", config_file
    )

    return read_config_file(config_file)


def cns_dist_restraint(resid_i, name_i, resid_j, name_j,
                       dist, lower, upper, weight=None,
                       comment=None):
    """
    # TODO
    # TODO: add weight handling
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
            resid_i, name_i, resid_j, name_j, dist, lower, upper,
            weight_str, comment_str
        )
    )
    return r


def secstruct_dist_restraints(residues, output_file, config_file=None,
                              secstruct_column="sec_struct_3state"):
    """
    Create .tbl file with dihedral angle restraints
    based on secondary structure prediction

    Logic based on choose_CNS_constraint_set.m,
    lines 519-1162

    Parameters
    ----------
    residues : pandas.DataFrame
        Table containing positions (column i), residue
        type (column A_i), and secondary structure for
        each position
    output_file : str
        Path to file in which restraints will be saved
    config_file : str, optional (default: None)
        Path to config file with folding settings. If None,
        will use default settings included in package
        (restraints.yml).
    secstruct_column : str, optional (default: sec_struct_3state)
        Column name in residues dataframe from which secondary
        structure will be extracted (has to be H, E, or C).
    """
    def _range_equal(start, end, char):
        """
        Check if secondary structure substring consists
        of one secondary structure state
        """
        range_str = "".join(
            [secstruct[pos] for pos in range(start, end + 1)]
        )
        return range_str == len(range_str) * char

    # get configuration (default or user-supplied)
    cfg = _folding_config(config_file)["secstruct_distance_restraints"]

    # extract amino acids and secondary structure into dictionary
    secstruct = dict(zip(residues.i, residues[secstruct_column]))
    aa = dict(zip(residues.i, residues.A_i))

    i_min = residues.i.min()
    i_max = residues.i.max()
    weight = cfg["weight"]

    with open(output_file, "w") as f:
        # go through secondary structure elements
        for sse, name in [("E", "strand"), ("H", "helix")]:
            # get distance restraint subconfig for current
            # secondary structure state
            sse_cfg = cfg[name]

            # define distance constraints based on increasing
            # sequence distance, and if the secondary structure
            # element reaches out that far. Specific distance restraints
            # are defined in config file for each sequence_dist
            for seq_dist, atoms in sorted(sse_cfg.items()):
                # now look at each position and the secondary
                # structure upstream to define the appropriate restraints
                for i in range(i_min, i_max - seq_dist + 1):
                    j = i + seq_dist
                    # test if upstream residues all have the
                    # same secondary structure state
                    if _range_equal(i, j, sse):
                        # go through all atom pairs and put constraints on them
                        for (atom1, atom2), (dist, range_) in atoms.items():
                            # can't put CB restraint if residue is a glycine
                            if ((atom1 == "CB" and aa[i] == "G") or
                                    (atom2 == "CB" and aa[j] == "G")):
                                continue

                            # write distance restraint
                            r = cns_dist_restraint(
                                i, atom1, j, atom2,
                                dist, range_, range_, weight,
                                AA1_to_AA3[aa[i]] + " " + AA1_to_AA3[aa[j]]
                            )
                            f.write(r + "\n")
