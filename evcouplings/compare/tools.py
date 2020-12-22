"""
Wrappers for tools for comparison of evolutionary
couplings to experimental structures

Authors:
  Anna G. Green
"""
from evcouplings.utils.system import (
    run, verify_resources
)

def run_dssp(infile, outfile, binary="dssp"):
    """
    Runs DSSP on an input pdb file

    Parameters
    ----------
    infile: str
        path to input file
    outfile: str
        path to output file
    binary: str
        path to DSSP binary
    """
    cmd = [
        binary,
        "-i", infile,
        "-o", outfile
    ]
    return_code, stdout, stderr = run(cmd, check_returncode=False)

    # verify_resources(
    #     "DSSP returned empty file: "
    #     "stdout={} stderr={} file={}".format(
    #         stdout, stderr, outfile
    #     ),
    #     outfile
    # )