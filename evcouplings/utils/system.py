"""
System-level calls to external tools, directory creation, etc.

Authors:
  Thomas A. Hopf
"""


import os
from os import path
import tempfile
import subprocess
import urllib.request
import shutil
import requests


class ResourceError(Exception):
    """
    Exception for missing resources (files, URLs, ...)
    """


class ExternalToolError(Exception):
    """
    Exception for failing external calculations
    """


def run(cmd, stdin=None, check_returncode=True,
        working_dir=None, shell=False, env=None):
    """
    Run external program as subprocess.

    Parameters
    ----------
    cmd : str or list of str
        Command (and  optional command line arguments)
    stdin : str or byte sequence, optional (default: None)
        Input to be sent to STDIN of the process
    check_returncode : bool, optional (default=True)
        Verify if call had returncode == 0, otherwise raise
        ExternalToolError
    working_dir : str, optional (default: None)
        Change to this directory before running command
    shell : bool, optional (default: False)
        Invoke shell when calling subprocess (default: False)
    env : dict, optional (default: None)
        Use this environment for executing the subprocess

    Returns
    -------
    int
        Return code of process
    stdout
        Byte string with stdout output
    stderr
        Byte string of stderr output

    Raises
    ------
    ExternalToolError
    """
    try:
        with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.PIPE, universal_newlines=True,
                cwd=working_dir, shell=shell, env=env
        ) as proc:
            (stdout, stderr) = proc.communicate(stdin)
            return_code = proc.returncode

            if check_returncode and return_code != 0:
                raise ExternalToolError(
                    "Call failed:\ncmd={}\nreturncode={}\nstdout={}\nstderr={}".format(
                        cmd, return_code, stdout, stderr
                    )
                )

        return return_code, stdout, stderr

    except (OSError, ValueError) as e:
        raise ExternalToolError(
            "Call to external tool failed and did not return: {}".format(cmd)
        ) from e


def valid_file(file_path):
    """
    Verify if a file exists and is not empty.

    Parameters
    ----------
    file_path : str
        Path to file to check

    Returns
    -------
    bool
        True if file exists and is non-zero size,
        False otherwise.
    """
    try:
        return os.stat(file_path).st_size > 0
    except (OSError, TypeError):
        # catch TypeError for nonsense paths, e.g. None
        return False


def verify_resources(message, *args):
    """
    Verify if a set of files exists and is not empty.

    Parameters
    ----------
    message : str
        Message to display with raised ResourceError
    *args : List of str
        Path(s) of file(s) to be checked

    Raises
    ------
    ResourceError
        If any of the resources does not exist or is empty
    """
    invalid = [str(f) for f in args if not valid_file(f)]

    if len(invalid) > 0:
        raise ResourceError(
            "{}:\n{}".format(message, ", ".join(invalid))
        )
    else:
        return True


def create_prefix_folders(prefix):
    """
    Create a directory tree contained in a prefix.

    prefix : str
        Prefix containing directory tree
    """
    dirname = path.dirname(prefix)

    if dirname != "":
        makedirs(dirname)


def makedirs(directories):
    """
    Create directory subtree, some or all of the folders
    may already exist.

    Parameters
    ----------
    directories : str
        Directory subtree to create
    """
    os.makedirs(directories, exist_ok=True)


def insert_dir(prefix, *dirs, rootname_subdir=True):
    """
    Create new path by inserting additional
    directories into the folder tree of prefix
    (but keeping the filename prefix at the end),
    
    Parameters
    ----------
    prefix : str
        Prefix of path that should be extended
    *dirs : str
        Add these directories at the end of path
    rootname_subdir : bool, optional (default: True)
        Given /my/path/prefix,

        * if True, creates structure like
          /my/path/prefix/*dirs/prefix

        * if False, creates structure like
          /my/path/*dirs/prefix

    Returns
    -------
    str
        Extended path
    """
    base_dir, rootname = path.split(prefix)
    if rootname_subdir:
        return path.join(prefix, *dirs, rootname)
    else:
        return path.join(base_dir, *dirs, rootname)


def temp():
    """
    Create a temporary file

    Returns
    -------
    str
        Path of temporary file
    """
    handle, name = tempfile.mkstemp()
    return name


def tempdir():
    """
    Create a temporary directory

    Returns
    -------
    str
        Path of temporary directory
    """
    return tempfile.mkdtemp()


def write_file(file_path, content):
    """
    Writes content to output file

    Parameters
    ----------
    file_path : str
        Path of output file
    content : str
        Content to be written to file
    """
    with open(file_path, "w") as f:
        f.write(content)


def get(url, output_path=None, allow_redirects=False):
    """
    Download external resource

    Parameters
    ----------
    url : str
        URL of resource that should be downloaded
    output_path: str, optional
        Save contents of URL to this file
        (only for text files)
    allow_redirects: bool
        Allow redirects by server or not

    Returns
    -------
    r : requests.models.Response
        Response object, use r.text to access text,
        r.json() to decode json, and r.content for
        raw bytestring

    Raises
    ------
    ResourceError

    """
    try:
        r = requests.get(url, allow_redirects=allow_redirects)

        if r.status_code != requests.codes.ok:
            raise ResourceError(
                "Invalid status code ({}) for URL: {}".format(
                    r.status_code, url
                )
            )

        if output_path is not None:
            try:
                write_file(output_path, r.text)
            except IOError as e:
                raise ResourceError(
                    "Could not save to file: {}".format(output_path)
                ) from e

        return r

    except requests.exceptions.RequestException as e:
        raise ResourceError() from e


def get_urllib(url, output_path):
    """
    Download external resource to file using urllib.
    This function is intended for cases where get()
    implemented using requests can not be used, e.g.
    for download from an FTP server.

    Parameters
    ----------
    url : str
        URL of resource that should be downloaded
    output_path: str, optional
        Save contents of URL to this file
        (only for text files)
    """
    with urllib.request.urlopen(url) as r, open(output_path, 'wb') as f:
        shutil.copyfileobj(r, f)
