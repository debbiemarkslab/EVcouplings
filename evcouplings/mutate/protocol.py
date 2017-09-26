"""
Sequence statistical energy and mutation effect computation
protocols

Authors:
  Thomas A. Hopf
"""

import pandas as pd
import matplotlib.pyplot as plt
from bokeh.io import save, output_file

from evcouplings.couplings.model import CouplingsModel
from evcouplings.mutate.calculations import (
    single_mutant_matrix, predict_mutation_table
)
import evcouplings
from evcouplings.utils.config import (
    check_required, InvalidParameterError
)
from evcouplings.utils.system import (
    create_prefix_folders, verify_resources
)


def standard(**kwargs):
    """
    Protocol:
    Compare ECs for single proteins (or domains)
    to 3D structure information

    Parameters
    ----------
    Mandatory kwargs arguments:
        See list below in code where calling check_required

    Returns
    -------
    outcfg : dict
        Output configuration of the pipeline, including
        the following fields:

        * mutation_matrix_file
        * [mutation_dataset_predicted_file]
    """
    check_required(
        kwargs,
        [
            "prefix", "model_file",
            "mutation_dataset_file",
        ]
    )

    prefix = kwargs["prefix"]

    outcfg = {
        "mutation_matrix_file": prefix + "_single_mutant_matrix.csv",
        "mutation_matrix_plot_files": [],
    }

    # make sure model file exists
    verify_resources(
        "Model parameter file does not exist",
        kwargs["model_file"]
    )

    # make sure output directory exists
    create_prefix_folders(prefix)

    # load couplings object, and create independent model
    c = CouplingsModel(kwargs["model_file"])
    c0 = c.to_independent_model()

    for model, type_ in [(c, "Epistatic"), (c0, "Independent")]:
        # interactive plot using bokeh
        filename = prefix + "_{}_model".format(type_.lower(),)
        output_file(
            filename + ".html", "{} model".format(type_)
        )
        fig = evcouplings.visualize.mutations.plot_mutation_matrix(model, engine="bokeh")
        save(fig)
        outcfg["mutation_matrix_plot_files"].append(filename + ".html")

        # static matplotlib plot
        evcouplings.visualize.mutations.plot_mutation_matrix(model)
        plt.savefig(filename + ".pdf", bbox_inches="tight")
        outcfg["mutation_matrix_plot_files"].append(filename + ".pdf")

    # create single mutation matrix table,
    # add prediction by independent model and
    # save to file
    singles = single_mutant_matrix(
        c, output_column="prediction_epistatic"
    )

    singles = predict_mutation_table(
        c0, singles, "prediction_independent"
    )

    singles.to_csv(outcfg["mutation_matrix_file"], index=False)

    # Pymol scripts
    outcfg["mutations_epistatic_pml_files"] = []
    for model in ["epistatic", "independent"]:
        pml_filename = prefix + "_{}_model.pml".format(model)
        evcouplings.visualize.mutations.mutation_pymol_script(
            singles, pml_filename, effect_column="prediction_" + model
        )
        outcfg["mutations_epistatic_pml_files"].append(pml_filename)

    # predict experimental dataset if given
    dataset_file = kwargs["mutation_dataset_file"]
    if dataset_file is not None:
        verify_resources("Dataset file does not exist", dataset_file)
        data = pd.read_csv(dataset_file, comment="#")

        # add epistatic model prediction
        data_pred = predict_mutation_table(
            c, data, "prediction_epistatic"
        )

        # add independent model prediction
        data_pred = predict_mutation_table(
            c0, data_pred, "prediction_independent"
        )

        outcfg["mutation_dataset_predicted_file"] = prefix + "_dataset_predicted.csv"
        data_pred.to_csv(
            outcfg["mutation_dataset_predicted_file"], index=False
        )

    return outcfg


# list of available mutation protocols
PROTOCOLS = {
    # standard EVmutation protocol
    "standard": standard,
}


def run(**kwargs):
    """
    Run mutation protocol

    Parameters
    ----------
    Mandatory kwargs arguments:
        protocol: EC protocol to run
        prefix: Output prefix for all generated files

    Returns
    -------
    outcfg : dict
        Output configuration of stage
        (see individual protocol for fields)
    """
    check_required(kwargs, ["protocol"])

    if kwargs["protocol"] not in PROTOCOLS:
        raise InvalidParameterError(
            "Invalid protocol selection: " +
            "{}. Valid protocols are: {}".format(
                kwargs["protocol"], ", ".join(PROTOCOLS.keys())
            )
        )

    return PROTOCOLS[kwargs["protocol"]](**kwargs)
