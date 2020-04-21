"""
Configuration handling

.. todo::

    switch ruamel.yaml to round trip loading
    to preserver order and comments?

Authors:
  Thomas A. Hopf
"""

import ruamel.yaml as yaml


class MissingParameterError(Exception):
    """
    Exception for missing parameters
    """


class InvalidParameterError(Exception):
    """
    Exception for invalid parameter settings
    """


def parse_config(config_str, preserve_order=False):
    """
    Parse a configuration string

    Parameters
    ----------
    config_str : str
        Configuration to be parsed
    preserve_order : bool, optional (default: True)
        Preserve formatting of input configuration
        string

    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        if preserve_order:
            return yaml.load(config_str, Loader=yaml.RoundTripLoader)
        else:
            return yaml.safe_load(config_str)
    except yaml.parser.ParserError as e:
        raise InvalidParameterError(
            "Could not parse input configuration. "
            "Formatting mistake in config file? "
            "See ParserError above for details."
        ) from e


def read_config_file(filename, preserve_order=False):
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
        return parse_config(f, preserve_order)


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
    if isinstance(config, yaml.comments.CommentedBase):
        dumper = yaml.RoundTripDumper
    else:
        dumper = yaml.Dumper

    with open(out_filename, "w") as f:
        f.write(
            yaml.dump(config, Dumper=dumper, default_flow_style=False)
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
