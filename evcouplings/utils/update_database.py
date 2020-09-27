"""
command-line app to update the necessary databases

Authors:
  Benjamin Schubert

"""
import ftplib
import datetime
import os
import errno
import zlib

from functools import partial
from pathlib import Path

import click

from evcouplings.compare import SIFTS
from evcouplings.utils import Progressbar

UNIPROT_URL = "ftp.uniprot.org"
UNIPROT_CWD = "/pub/databases/uniprot/current_release/knowledgebase/complete/"
UNIPROT_FILE = "uniprot_{type}.fasta.gz"

DB_URL = "ftp.uniprot.org"
DB_CWD = "/pub/databases/uniprot/uniref/{type}/"
DB_FILE = "{type}.fasta.gz"

DB_SUFFIX = "{type}_{year}_{month}.fasta"
DB_CURRENT = "{type}_current.fasta"

SIFTS_SUFFIX = "pdb_chain_uniprot_plus_{year}_{month}_{day}.{extension}"
SIFTS_CURRENT = "pdb_chain_uniprot_plus_current.{extension}"


def symlink_force(target, link_name):
    """
    Creates or overwrites an existing symlink

    Parameters
    ----------
    target : str
        the target file path
    link_name : str
        the symlink name

    """
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def download_ftp_file(ftp_url, ftp_cwd, file_url, output_path, file_handling="wb", gziped=False, verbose=False):
    """
    Downloads a gzip file from a remote ftp server and
    decompresses it on the fly into an output file

    Parameters
    ----------
    ftp_url : str
        the FTP server url
    ftp_cwd : str
        the FTP directory of the file to download
    file_url : str
        the file name that gets downloaded
    output_path : str
        the path to the output file on the local system
    file_handling : str
        the file handling mode (default: 'wb')
    verbose : bool
        determines whether a progressbar is printed
    """
    def _callback(_bar, decompressor, chunk):
        if gziped:
            out.write(decompressor.decompress(chunk))
        else:
            out.write(chunk)
        if verbose:
            _bar += len(chunk)

    ftp = ftplib.FTP(ftp_url)
    ftp.login()
    ftp.cwd(ftp_cwd)
    with open(output_path, file_handling) as out:
        filesize = ftp.size(file_url)
        pbar = Progressbar(filesize) if verbose else None

        # automatic header detection
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 32)
        callback = partial(_callback, pbar, decompressor)
        ftp.retrbinary('RETR %s' % file_url, callback,
                       blocksize=8192)
    ftp.quit()


def run(**kwargs):
    """
    Exposes command line interface as a Python function.

    Parameters
    ----------
    kwargs
        See click.option decorators for app() function
    """
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    verbose = kwargs.get("verbose", False)
    symlink = kwargs.get("symlink", False)

    # update SIFTS file
    if verbose:
        print("Updating SIFTS")

    SIFTS_dir = os.path.abspath(kwargs.get("sifts", os.path.realpath(__file__)))
    # create directory if it does not exist
    # ignores if directory on the way already exist
    dir = Path(SIFTS_dir)
    dir.mkdir(parents=True, exist_ok=True)
    sifts = os.path.join(SIFTS_dir, SIFTS_SUFFIX)
    sifts_curr = os.path.join(SIFTS_dir, SIFTS_CURRENT)
    sifts_table = sifts.format(year=year, month=month, day=day, extension="csv")
    sifts_fasta = sifts.format(year=year, month=month, day=day, extension="fasta")
    s_new = SIFTS(sifts.format(year=year, month=month, day=day, extension="csv"))
    s_new.create_sequence_file(sifts.format(year=year, month=month, day=day, extension="fasta"))

    # set symlink to "<file>_current"
    if symlink:
        symlink_force(sifts_table, sifts_curr.format(extension="csv"))
        symlink_force(sifts_fasta, sifts_curr.format(extension="fasta"))

    # update uniref
    db_path = os.path.abspath(kwargs.get("db", os.path.realpath(__file__)))
    for db_type in ["uniprot", "uniref100", "uniref90" ]:

        if verbose:
            print("Updating", db_type)

        # if not existent create folder db_path/db_type
        db_full_path = os.path.join(db_path, db_type)
        dir = Path(db_full_path)
        dir.mkdir(parents=True, exist_ok=True)

        if db_type == "uniprot":
            # download Swiss and TrEMBL and concatinate both
            out_path = os.path.join(db_full_path, DB_SUFFIX.format(type=db_type, year=year, month=month))
            db_curr = os.path.join(db_full_path, DB_CURRENT.format(type=db_type))
            for i, type_d in enumerate(["sprot", "trembl"]):
                if i:
                    file_url = UNIPROT_FILE.format(type=type_d)
                    download_ftp_file(UNIPROT_URL, UNIPROT_CWD, file_url, out_path, gziped=True,
                                      file_handling="ab", verbose=verbose)
                else:
                    file_url = UNIPROT_FILE.format(type=type_d)
                    download_ftp_file(UNIPROT_URL, UNIPROT_CWD, file_url, out_path, gziped=True, verbose=verbose)
        else:
            # download uniref db
            db_file = DB_FILE.format(type=db_type)
            db_cwd = DB_CWD.format(type=db_type)
            out_path = os.path.join(db_full_path, DB_SUFFIX.format(type=db_type, year=year, month=month))
            db_curr = os.path.join(db_full_path, DB_CURRENT.format(type=db_type))
            download_ftp_file(DB_URL, db_cwd, db_file, out_path, gziped=True, verbose=verbose)

        if symlink:
            symlink_force(out_path, db_curr)


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
# run settings
@click.option("-s", "--sifts", default="/n/groups/marks/databases/SIFTS/", help="SIFTS output directory")
@click.option("-d", "--db", default="/n/groups/marks/databases/jackhmmer/", help="Uniprot output directory")
@click.option("-l", "--symlink", default=False, is_flag=True,
              help="Creates symlink with ending '_current.' pointing to the newly created db files")
@click.option("-v", "--verbose", default=False, is_flag=True, help="Enables verbose output")
def app(**kwargs):
    """
    Update database command line interface
    """
    run(**kwargs)


if __name__ == "__main__":
    app()
