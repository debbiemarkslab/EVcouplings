"""
Create summary statistics / plots for runs from
evcouplings app

Authors:
  Thomas A. Hopf
"""

import pandas as pd
import click
from evcouplings.utils.system import valid_file
from evcouplings.utils.config import read_config_file, InvalidParameterError

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def protein_monomer(prefix, configs):
    """
    Create results summary for run using
    protein_monomer pipeline

    # TODO: lock output files: https://github.com/openstack/pylockfile
    # TODO: make plots (based on extracted score thresholds)
    """
    print(prefix)
    ali_table = pd.DataFrame()

    # go through all config files
    for cfg_file in configs:
        # check if the file exists and has contents
        # since run might not yet have finished or crashed
        if valid_file(cfg_file):
            # job input configuration
            C = read_config_file(cfg_file)
            sub_prefix = C["global"]["prefix"]

            # read final output state of job
            R = read_config_file(sub_prefix + "_final_global_state.outcfg")
            stat_file = R["statistics_file"]

            # read and modify alignmen statistics
            if valid_file(stat_file):
                # get alignment stats for current job
                stat_df = pd.read_csv(stat_file)
                n_eff = R["effective_sequences"]

                if n_eff is not None:
                    stat_df.loc[0, "N_eff"] = n_eff

                stat_df.loc[0, "domain_threshold"] = C["align"]["domain_threshold"]

                ali_table = ali_table.append(stat_df)

    ali_table = ali_table.sort_values(by="domain_threshold")
    ali_table.to_csv(
        prefix + "_alignment_statistics_summary.csv", index=False, float_format="%.3f"
    )


PIPELINE_TO_SUMMARIZER = {
    "protein_monomer": protein_monomer,
}


@click.command(context_settings=CONTEXT_SETTINGS)
# run settings
@click.argument('pipeline', nargs=1, required=True)
@click.argument('prefix', nargs=1, required=True)
@click.argument('configs', nargs=-1)
def run(**kwargs):
    """
    Create summary statistics for evcouplings pipeline runs
    """
    try:
        summarizer = PIPELINE_TO_SUMMARIZER[kwargs["pipeline"]]
    except KeyError:
        raise InvalidParameterError(
            "Not a valid pipeline, valid selections are: {}".format(
                ",".join(PIPELINE_TO_SUMMARIZER.keys())
            )
        )

    summarizer(kwargs["prefix"], kwargs["configs"])


if __name__ == '__main__':
    run()
