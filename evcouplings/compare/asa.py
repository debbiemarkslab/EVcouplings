import pandas as pd
from evcouplings.utils.system import run, verify_resources, valid_file
from Bio.PDB import make_dssp_dict
import ruamel.yaml as yaml
import numpy as np

# Amino acid surface area values from Tien et al, 2013 (empirical values)
AA_SURFACE_AREA = {
    "A": 121,
    "R": 265,
    "D": 187,
    "N": 187,
    "C": 148,
    "E": 214,
    "Q": 214,
    "G": 97,
    "H": 216,
    "I": 195,
    "L": 191,
    "K": 230,
    "M": 103,
    "F": 228,
    "P": 154,
    "S": 143,
    "T": 163,
    "W": 264,
    "Y": 255,
    "V": 165,
    "X": np.nan
}

# run dssp

def run_dssp(binary, infile, outfile):
    """
    Runs DSSP on an input pdb file

    Parameters
    ----------
    binary: str
        path to DSSP binary
    infile: str
        path to input file
    outfile: str
        path to output file
    """
    cmd = [
        binary,
        "-i", infile,
        "-o", outfile
    ]
    return_code, stdout, stderr = run(cmd)

    verify_resources(
            "DSSP returned empty file: "
            "stdout={} stderr={} file={}".format(
                stdout, stderr, outfile
            ),
            outfile
        )

def read_dssp_output(filename):
    """
    Reads the output files from DSSP and converts them into a pandas DataFrame

    Parameters
    ----------
    filename: str
        Path to output file from DSSP

    Returns
    -------
        pd.DataFrame with columns i, res, asa
        Representing residue number, identity, and accesisble surface area
    """

    dssp_dict, _ = make_dssp_dict(filename)
    data = []
    for key, value in dssp_dict.items():

        # keys are formatted as (chain, ("", i, ""))
        i = key[1][1]

        res = value[0]
        asa = value[2]

        data.append({
            "i": i,
            "res": res,
            "asa": asa
        })

    return pd.DataFrame(data)

def calculate_rsa(dataframe, AA_SURFACE_AREA, output_column="rsa"):
    """
    Converts raw accessible surface area to relative accessible surface area,
    by dividing the raw accessible surface area by the max accessible surface area

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe of raw accessible surface area
    AA_SURFACE_AREA: dict of str: numeric
        Values of max accessible surface area to use for conversion
    output_column: str
        Name of output column to create

    Returns
    -------
        pd.DataFrame
    """

    dataframe.loc[:, output_column] = [x.asa/AA_SURFACE_AREA[x.res] for _, x in dataframe.iterrows()]
    return dataframe

def asa_run(pdb_file, dssp_output_file, rsa_output_file, dssp_binary):
    """
    Paramters
    ---------
    file: str
        path to pdb file on which to run DSSP
    dssp_output_file: str
        path to save dssp file
    rsa_output_file: str
        path to save rsa output file
    dssp_binary: str
        path to dssp binary

    Returns
    -------
    pd.DataFrame with relative accessible surface area for each position in PDB
    """

    run_dssp(dssp_binary, pdb_file, dssp_output_file)

    d = read_dssp_output(dssp_output_file)
    d = calculate_rsa(d, AA_SURFACE_AREA)

    return d

def combine_asa(remapped_pdb_files, dssp_binary, outcfg):
    """
    Parameters
    ----------
    remapped_pdb_files: list of str
        path to all remapped pdb files to be analyzed
    prefix: str
        path to DSSP binary
    outcfg: dict
        output configuration
    """

    # Initialize a dataframe to contain the asa information
    data = pd.DataFrame({
            "i": [],
            "res": [],
            "asa": [],
            "rsa": []
    })

    outcfg["dssp_output_files"] = []
    outcfg["rsa_output_files"] = []

    # If no remapped pdb files, return empty df
    if len(remapped_pdb_files) == 0:
        return pd.DataFrame({
            "i": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "min": np.nan
        }, index=[0])

    # run dssp for each remapped_pdb_file
    for file in remapped_pdb_files:
        if valid_file(file):

            # the DSSP and RSA files will be saved as with same prefix as PDB
            prefix = file.split(".pdb")[0]
            dssp_output_file = prefix + ".dssp"
            rsa_output_file = prefix + "_rsa.csv"

            d = asa_run(file, dssp_output_file, rsa_output_file, dssp_binary)

            # add information to combined dataframe
            data = pd.concat([data, d])

            # save the output files
            outcfg["dssp_output_files"].append(dssp_output_file)
            outcfg["rsa_output_files"].append(rsa_output_file)

    # group the dataframe of RSA by residue
    means = data.groupby("i").rsa.mean()
    maxes = data.groupby("i").rsa.max()
    mins = data.groupby("i").rsa.min()

    return pd.DataFrame({
        "i": means.index,
        "mean": list(means),
        "max": list(maxes),
        "min": list(mins)
    }), outcfg

def add_asa(ec_df, asa, asa_column):
    """
    Add a column for the accessible surface area for each residue i and j to a DataFrame

    Parameters
    ----------
    ec_df: pd.DataFrame
        dataframe with columns i, j, segment_i, and segment_j
    asa: pd.DataFrame
        dataframe containing accessible surface area information
    asa_column: str
        name of column in asa df to use

    Returns
    -------
    pd.DataFrame
    """
    # make a dictionary of residue and segment pointint to accesible surface area value
    s_to_e = {(x,y):z for x,y,z in zip(asa.i, asa.segment_i, asa[asa_column])}

    # Add the accesible surface area for position i
    ec_df["asa_i"] =[s_to_e[(x,y)] if (x,y) in s_to_e else np.nan for x,y in zip(ec_df.i, ec_df.segment_i)]

    # Add the accessible surface area for position j
    ec_df["asa_j"] =[s_to_e[(x,y)] if (x,y) in s_to_e else np.nan for x,y in zip(ec_df.j, ec_df.segment_j)]

    return ec_df
