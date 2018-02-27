import tarfile
import os

from evcouplings.management.dump import get_dumper


def protocol_standard(**kwargs):

    incfg = kwargs

    outcfg = {
        **kwargs
    }

    dumper = get_dumper(incfg)

    tar_location = dumper.write_tar()

    outcfg["archive_file"] = tar_location

    # delete selected output files if requested
    outcfg = delete_outputs(incfg, outcfg)

    return outcfg


PROTOCOLS = {
    "standard": protocol_standard,
}


def run(**kwargs):
    return PROTOCOLS.get(kwargs["protocol"])(**kwargs)


def delete_outputs(config, outcfg):
    """
    Remove pipeline outputs to save memory
    after running the job

    Parameters
    ----------
    config : dict-like
        Input configuration of job. Uses
        config["management"]["delete"] (list of key
        used to index outcfg) to determine
        which files should be added to archive
    outcfg : dict-like
        Output configuration of job

    Returns
    -------
    outcfg_cleaned : dict-like
        Output configuration with selected
        output keys removed.
    """
    # determine keys (corresponding to files) in
    # outcfg that should be stored
    outkeys = config.get("delete", None)

    # if no output keys are requested, nothing to do
    if outkeys is None:
        return outcfg

    # go through all flagged files and delete if existing
    for k in outkeys:
        # skip missing keys or ones not defined
        if k not in outcfg or k is None:
            continue

        # delete list of files
        if k.endswith("files"):
            for f in outcfg[k]:
                try:
                    os.remove(f)
                except OSError:
                    pass
            del outcfg[k]

        # delete individual file
        else:
            try:
                os.remove(outcfg[k])
                del outcfg[k]
            except OSError:
                pass

    return outcfg