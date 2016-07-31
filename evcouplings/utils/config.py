"""
Configuration handling

Authors:
  Thomas A. Hopf
"""

import yaml


class MissingParameterError(Exception):
    """
    Exception for missing parameters
    """


class InvalidParameterError(Exception):
    """
    Exception for invalid parameter settings
    """


def read_config_file(filename):
    """
    Read and parse a configuration file.

    Parameters
    ----------
    filename : str
        Path of configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(filename) as f:
        config = yaml.safe_load(f)

    return config


def check_required(params, keys):
    """
    Verify if required set of parameters is present in configuration

    Parameters
    ----------
    params : dict
        Dictionary with parameters
    keys : list-like
        Set of parameters that has to be present in params

    Returns
    -------
    bool
        True if all parameters present, False otherwise
    """
    missing = [k for k in keys if k not in params]

    if len(missing) > 0:
        raise MissingParameterError(
            "Missing required parameters: {} \nGiven: {}".format(
                ", ".join(missing), params
            )
        )
