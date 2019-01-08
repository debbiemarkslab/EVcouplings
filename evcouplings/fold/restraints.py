"""
Functions for generating distance restraints from
evolutionary couplings and secondary structure predictions

Authors:
  Thomas A. Hopf
  Anna G. Green (docking restraints)
"""

from pkg_resources import resource_filename
from evcouplings.utils.config import read_config_file
from evcouplings.utils.constants import AA1_to_AA3
from evcouplings.utils.system import verify_resources


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

def _docking_config(config_file=None):
    """
    Load docking configuration

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
            __name__, "cns_templates/haddock_restraints.yml"
        )

    # check if config file exists and read
    verify_resources(
        "Folding config file does not exist or is empty", config_file
    )

    return read_config_file(config_file)


def secstruct_dist_restraints(residues, output_file,
                              restraint_formatter, config_file=None,
                              secstruct_column="sec_struct_3state"):
    """
    Create .tbl file with distance restraints
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
    restraint_formatter : function
        Function called to create string representation of restraint
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
            # sequence distance, and test if the secondary structure
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
                            r = restraint_formatter(
                                i, atom1, j, atom2,
                                dist=dist,
                                lower=range_,
                                upper=range_,
                                weight=weight,
                                comment=AA1_to_AA3[aa[i]] + " " + AA1_to_AA3[aa[j]]
                            )
                            f.write(r + "\n")


def secstruct_angle_restraints(residues, output_file,
                               restraint_formatter, config_file=None,
                               secstruct_column="sec_struct_3state"):
    """
    Create .tbl file with dihedral angle restraints
    based on secondary structure prediction

    Logic based on make_cns_angle_constraints.pl

    Parameters
    ----------
    residues : pandas.DataFrame
        Table containing positions (column i), residue
        type (column A_i), and secondary structure for
        each position
    output_file : str
        Path to file in which restraints will be saved
    restraint_formatter : function, optional
        Function called to create string representation of restraint
    config_file : str, optional (default: None)
        Path to config file with folding settings. If None,
        will use default settings included in package
        (restraints.yml).
    secstruct_column : str, optional (default: sec_struct_3state)
        Column name in residues dataframe from which secondary
        structure will be extracted (has to be H, E, or C).
    """

    def _phi(pos, sse):
        sse_cfg = cfg[sse]["phi"]
        return restraint_formatter(
            pos, "C",
            pos + 1, "N",
            pos + 1, "CA",
            pos + 1, "C",
            **sse_cfg
        )

    def _psi(pos, sse):
        sse_cfg = cfg[sse]["psi"]
        return restraint_formatter(
            pos, "N",
            pos, "CA",
            pos, "C",
            pos + 1, "N",
            **sse_cfg
        )

    # get configuration (default or user-supplied)
    cfg = _folding_config(config_file)["secstruct_angle_restraints"]

    # extract amino acids and secondary structure into dictionary
    secstruct = dict(zip(residues.i, residues[secstruct_column]))
    aa = dict(zip(residues.i, residues.A_i))

    i_min = residues.i.min()
    i_max = residues.i.max()

    with open(output_file, "w") as f:
        # go through all positions
        for i in range(i_min, i_max - 1):
            # check if two subsequent identical secondary structure states
            # helix
            if secstruct[i] == "H" and secstruct[i + 1] == "H":
                f.write(_phi(i, "helix") + "\n")
                f.write(_psi(i, "helix") + "\n")
            # strand
            elif secstruct[i] == "E" and secstruct[i + 1] == "E":
                f.write(_phi(i, "strand") + "\n")
                f.write(_psi(i, "strand") + "\n")


def ec_dist_restraints(ec_pairs, output_file,
                       restraint_formatter, config_file=None):
    """
    Create .tbl file with distance restraints
    based on evolutionary couplings

    Logic based on choose_CNS_constraint_set.m,
    lines 449-515

    Parameters
    ----------
    ec_pairs : pandas.DataFrame
        Table with EC pairs that will be turned
        into distance restraints
        (with columns i, j, A_i, A_j)
    output_file : str
        Path to file in which restraints will be saved
    restraint_formatter : function
        Function called to create string representation of restraint
    config_file : str, optional (default: None)
        Path to config file with folding settings. If None,
        will use default settings included in package
        (restraints.yml).
    """
    # get configuration (default or user-supplied)
    cfg = _folding_config(config_file)["pair_distance_restraints"]

    with open(output_file, "w") as f:
        # create distance restraints per EC row in table
        for idx, ec in ec_pairs.iterrows():
            i, j, aa_i, aa_j = ec["i"], ec["j"], ec["A_i"], ec["A_j"]

            for type_ in ["c_alpha", "c_beta", "tertiary_atom"]:
                tcfg = cfg[type_]

                # check if we want this type of restraint first
                if not tcfg["use"]:
                    continue

                # restraint weighting: currently only support none,
                # or fixed numerical value
                if isinstance(tcfg["weight"], str):
                    # TODO: implement restraint weighting functions eventually
                    raise NotImplementedError(
                        "Restraint weighting functions not yet implemented: " +
                        tcfg["weight"]
                    )
                else:
                    weight = tcfg["weight"]

                # determine which atoms to put restraint on
                # can be residue-type specific dict or fixed value
                atoms = tcfg["atoms"]
                if isinstance(atoms, dict):
                    atom_i = atoms[aa_i]
                    atom_j = atoms[aa_j]
                else:
                    atom_i = atoms
                    atom_j = atoms

                # skip if we would put a CB restraint on glycine residues;
                # this should be generalized to skip any invalid selection eventually
                if ((aa_i == "G" and atom_i == "CB") or
                        (aa_j == "G" and atom_j == "CB")):
                    continue

                # write restraint
                r = restraint_formatter(
                    i, atom_i, j, atom_j,
                    dist=tcfg["dist"],
                    lower=tcfg["lower"],
                    upper=tcfg["upper"],
                    weight=weight,
                    comment=AA1_to_AA3[aa_i] + " " + AA1_to_AA3[aa_j]
                )
                f.write(r + "\n")


def docking_restraints(ec_pairs, output_file,
                       restraint_formatter, config_file=None):
    """
    Create .tbl file with distance restraints
    for docking

    Parameters
    ----------
    ec_pairs : pandas.DataFrame
        Table with EC pairs that will be turned
        into distance restraints
        (with columns i, j, A_i, A_j, segment_i, segment_j)
    output_file : str
        Path to file in which restraints will be saved
    restraint_formatter : function
        Function called to create string representation of restraint
    config_file : str, optional (default: None)
        Path to config file with folding settings. If None,
        will use default settings included in package
        (restraints.yml).
    """
    # get configuration (default or user-supplied)
    cfg = _docking_config(config_file)["docking_restraints"]

    with open(output_file, "w") as f:
        # create distance restraints per EC row in table
        for idx, ec in ec_pairs.iterrows():
            i, j, aa_i, aa_j, segment_i, segment_j = (
                ec["i"], ec["j"], ec["A_i"], ec["A_j"], ec["segment_i"], ec["segment_j"]
            )

            # extract chain names based on segment names
            # A_1 -> A, B_1 -> B
            chain_i = segment_i[0]
            chain_j = segment_j[0]

            # write i to j restraint
            r = restraint_formatter(
                i, chain_i, j, chain_j,
                dist=cfg["dist"],
                lower=cfg["lower"],
                upper=cfg["upper"],
            )
            f.write(r + "\n")
