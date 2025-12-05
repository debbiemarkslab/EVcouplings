"""
Configuration handling

.. todo::

    switch ruamel.yaml to round trip loading
    to preserver order and comments?

Authors:
  Thomas A. Hopf
"""

from ruamel.yaml import YAML
from ruamel.yaml.error import MarkedYAMLError


class MissingParameterError(Exception):
    """
    Exception for missing parameters
    """


class InvalidParameterError(Exception):
    """
    Exception for invalid parameter settings
    """


def parse_config(config_str):
    """
    Parse a configuration string

    Parameters
    ----------
    config_str : str
        Configuration to be parsed

    Returns
    -------
    dict
        Configuration dictionary
    """
    # uses round-trip loader by default, preserving order
    yaml = YAML(typ='safe', pure=True)
    try:
        return yaml.load(config_str)
    except MarkedYAMLError as e:
        raise InvalidParameterError(
            "Could not parse input configuration. "
            "Formatting mistake in config file? "
            "See MarkedYAMLError above for details."
        ) from e


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
        return parse_config(f)


def write_config_file(out_filename, config):
    """
    Save configuration data structure in YAML file.

    Parameters
    ----------
    out_filename : str
        Filename of output file
    config : dict
        Config data that will be written to file
    """
    yaml = YAML(typ='safe', pure=True)
    yaml.default_flow_style = False
    with open(out_filename, "w") as f:
        f.write(
            yaml.dump(config)
        )


def check_required(params, keys):
    """
    Verify if required set of parameters is present in configuration

    Parameters
    ----------
    params : dict
        Dictionary with parameters
    keys : list-like
        Set of parameters that has to be present in params

    Raises
    ------
    MissingParameterError
    """
    missing = [k for k in keys if k not in params]

    if len(missing) > 0:
        raise MissingParameterError(
            "Missing required parameters: {} \nGiven: {}".format(
                ", ".join(missing), params
            )
        )


def iterate_files(outcfg, subset=None):
    """
    Generator function to iterate a list of file
    items in an outconfig

    Parameters
    ----------
    outcfg : dict(str)
        Configuration to extract file items for iteration from
    subset : list(str)
        List of keys in outcfg to restrict iteration to

    Returns
    -------
    tuple(str, str, int)
        Generator over tuples (file path, entry key, index).
        index will be None if this is a single file entry
        (i.e. ending with _file rather than _files).
    """
    for k, v in outcfg.items():
        # skip items if there is a subset filter and it matches
        if subset is not None and k not in subset:
            continue

        # also skip in case file has a null value
        if v is None:
            continue

        # only look at file entries, so skip everything else
        # if not (k.endswith("_file") or k.endswith("_files")):
        #    continue
        if k.endswith("_file"):
            yield (v, k, None)
        elif k.endswith("_files"):
            for i, f in enumerate(v):
                yield (f, k, i)
        else:
            # skip any other entries
            pass
