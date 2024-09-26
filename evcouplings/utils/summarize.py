"""
Create summary statistics / plots for runs from
evcouplings app

Authors:
  Thomas A. Hopf
"""

# chose backend for command-line usage
import matplotlib
matplotlib.use("Agg")

from collections import defaultdict
import filelock

import pandas as pd
import click
import matplotlib.pyplot as plt

from evcouplings.utils.system import valid_file
from evcouplings.utils.config import read_config_file, InvalidParameterError
from evcouplings.utils.pipeline import FINAL_CONFIG_SUFFIX

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def protein_monomer(prefix, configs):
    """
    Create results summary for run using
    protein_monomer pipeline

    # TODO
    """
    MIN_PROBABILITY = 0.9

    ali_table = pd.DataFrame()
    prefix_to_cfgs = {}
    data = defaultdict(lambda: defaultdict())

    # go through all config files
    for cfg_file in configs:
        # check if the file exists and has contents
        # since run might not yet have finished or crashed
        if valid_file(cfg_file):
            # job input configuration
            C = read_config_file(cfg_file)
            sub_prefix = C["global"]["prefix"]
            domain_threshold = C["align"]["domain_threshold"]
            sub_index = (domain_threshold, sub_prefix)

            final_state_cfg = sub_prefix + FINAL_CONFIG_SUFFIX
            if not valid_file(final_state_cfg):
                continue

            # read final output state of job
            R = read_config_file(final_state_cfg)
            data[sub_index]["identities"] = R["identities_file"]
            data[sub_index]["frequencies"] = R["frequencies_file"]
            data[sub_index]["minimum_column_coverage"] = C["align"]["minimum_column_coverage"]

            stat_file = R["statistics_file"]
            ec_file = R.get("ec_file", "")
            ec_comp_file = R.get("ec_compared_longrange_file", "")

            prefix_to_cfgs[(sub_prefix)] = (C, R)

            # read and modify alignment statistics
            if valid_file(stat_file):
                # get alignment stats for current job
                stat_df = pd.read_csv(stat_file)
                n_eff = R["effective_sequences"]

                if n_eff is not None:
                    stat_df.loc[0, "N_eff"] = n_eff

                stat_df.loc[0, "domain_threshold"] = domain_threshold
                L = stat_df.loc[0, "num_cov"]

                # try to get number of significant ECs in addition
                if valid_file(ec_file):
                    ecs = pd.read_csv(ec_file)
                    min_seq_dist = C["compare"]["min_sequence_distance"]
                    num_sig = len(ecs.query(
                        "abs(i-j) >= @min_seq_dist and probability >= @MIN_PROBABILITY"
                    ))
                    stat_df.loc[0, "num_significant"] = num_sig

                # try to get EC precision in addition
                if valid_file(ec_comp_file):
                    ec_comp = pd.read_csv(ec_comp_file)
                    stat_df.loc[0, "precision"] = ec_comp.iloc[L]["precision"]

                # finally, append to global table
                ali_table = pd.concat([ali_table, stat_df])

    # sort table by sequence search threshold
    ali_table = ali_table.sort_values(by="domain_threshold")

    # when saving files, have to aquire lock to make sure
    # jobs don't start overwriting results

    # make plots and save
    fig = _protein_monomer_plot(ali_table, data)
    plot_file = prefix + "_job_statistics_summary.pdf"
    lock_plot = filelock.FileLock(plot_file)
    with lock_plot:
        fig.savefig(plot_file, bbox_inches="tight")

    # save ali statistics table
    table_file = prefix + "_job_statistics_summary.csv"
    lock_table = filelock.FileLock(table_file)
    with lock_table:
        ali_table.to_csv(
            table_file, index=False, float_format="%.3f"
        )

    return ali_table


def _protein_monomer_plot(ali_table, data):
    """
    # TODO
    """
    import seaborn as sns
    sns.set_palette("Paired", len(ali_table), None)

    FONTSIZE = 16
    # set up plot and grid
    fig = plt.figure(figsize=(15, 15))
    gridsize = ((3, 2))
    ax_cov = plt.subplot2grid(gridsize, (0, 0), colspan=1)
    ax_distr = plt.subplot2grid(gridsize, (0, 1), colspan=1)
    ax_gaps = plt.subplot2grid(gridsize, (1, 0), colspan=2)
    ax_sig = plt.subplot2grid(gridsize, (2, 0), colspan=1)
    ax_comp = plt.subplot2grid(gridsize, (2, 1), colspan=1)

    # 1) Number of sequences, coverage
    l_seqs = ax_cov.plot(
        ali_table.domain_threshold, ali_table.N_eff / ali_table.num_cov,
        "ok-", label="# Sequences"
    )
    ax_cov.set_xlabel("Domain inclusion threshold")
    ax_cov.set_ylabel("# effective sequences / L")
    ax_cov.set_title("Sequences and coverage", fontsize=FONTSIZE)
    ax_cov.legend(loc="lower left")

    ax_cov2 = ax_cov.twinx()
    l_cov = ax_cov2.plot(
        ali_table.domain_threshold, ali_table.num_cov / ali_table.seqlen,
        "o-", label="Coverage", color="#2079b4"
    )
    ax_cov2.set_ylabel("Coverage (% of region)")
    ax_cov2.legend(loc="lower right")
    ax_cov2.set_ylim(0, 1)

    # 2) sequence identity & coverage distributions
    for (domain_threshold, subjob), subdata in sorted(data.items()):
        # sequence identities to query
        if valid_file(subdata["identities"]):
            ids = pd.read_csv(subdata["identities"]).identity_to_query.dropna()
            ax_distr.hist(
                ids, histtype="step", range=(0, 1.0),
                bins=100, density=True, cumulative=True, linewidth=3,
                label=str(domain_threshold)
            )

            ali_table.loc[ali_table.prefix == subjob, "average_identity"] = ids.mean()

        # coverage distribution
        if valid_file(subdata["frequencies"]):
            freqs = pd.read_csv(subdata["frequencies"])
            # print(freqs.head())
            ax_gaps.plot(
                freqs.i, 1 - freqs.loc[:, "-"], "o", linewidth=3,
                label=str(domain_threshold)
            )
            mincov = subdata["minimum_column_coverage"]
            if mincov > 1:
                mincov /= 100

            ax_gaps.axhline(mincov, ls="--", color="k")

    ax_distr.set_xlabel("% sequence identity to query")
    ax_distr.set_title("Sequence identity distribution", fontsize=FONTSIZE)
    ax_distr.set_xlim(0, 1)
    ax_distr.set_ylim(0, 1)
    ax_distr.legend()

    ax_gaps.set_title("Gap statistics", fontsize=FONTSIZE)
    ax_gaps.set_xlabel("Sequence index")
    ax_gaps.set_ylabel("Column coverage (1 - % gaps)")
    ax_gaps.autoscale(enable=True, axis='x', tight=True)
    ax_gaps.set_ylim(0, 1)
    ax_gaps.legend(loc="best")

    # number of significant ECs, EC precision
    if "num_significant" in ali_table.columns:
        ax_sig.plot(
            ali_table.domain_threshold,
            ali_table.num_significant / ali_table.num_cov,
            "ok-"
        )

    ax_sig.set_title("Significant ECs", fontsize=FONTSIZE)
    ax_sig.set_xlabel("Domain inclusion threshold")
    ax_sig.set_ylabel("Fraction of significant ECs (% of L)")

    if "precision" in ali_table.columns:
        ax_comp.plot(ali_table.domain_threshold, ali_table.precision, "ok-")

    ax_comp.set_title("Comparison to 3D (top L ECs)", fontsize=FONTSIZE)
    ax_comp.set_xlabel("Domain inclusion threshold")
    ax_comp.set_ylabel("EC precision")
    ax_comp.set_ylim(0, 1)

    return fig


def protein_complex(prefix, configs):
    """
    Create results summary for run using
    protein_complex pipeline

    """
    # TODO: this is only designed to work with skewnormal threshold
    MIN_PROBABILITY = 0.9

    # number of inter ECs to check for precision
    NUM_INTER = 5

    # TODO: create segments global variable and import
    FIRST_SEGMENT = "A_1"
    SECOND_SEGMENT = "B_1"

    ali_table = pd.DataFrame()
    prefix_to_cfgs = {}
    data = defaultdict(lambda: defaultdict())

    # go through all config files
    for cfg_file in configs:
        # check if the file exists and has contents
        # since run might not yet have finished or crashed
        if valid_file(cfg_file):
                # job input configuration
                C = read_config_file(cfg_file)
                sub_prefix = C["global"]["prefix"]
                sub_index = (sub_prefix)

                final_state_cfg = sub_prefix + FINAL_CONFIG_SUFFIX
                if not valid_file(final_state_cfg):
                    continue

                # read final output state of job
                R = read_config_file(final_state_cfg)
                data[sub_index]["identities"] = R["identities_file"]
                data[sub_index]["frequencies"] = R["frequencies_file"]
                data[sub_index]["minimum_column_coverage"] = C["concatenate"]["minimum_column_coverage"]

                stat_file = R["statistics_file"]
                ec_file = R.get("ec_file", "")
                ec_comp_file = R.get("ec_compared_longrange_file", "")
                concat_stat_file = R.get("concatentation_statistics_file", "")
                first_stat_file = R.get("first_statistics_file","")
                second_stat_file = R.get("second_statistics_file","")

                prefix_to_cfgs[(sub_prefix)] = (C, R)

                # read and modify alignment statistics
                if valid_file(stat_file):
                    # get alignment stats for current job
                    stat_df = pd.read_csv(stat_file)
                    n_eff = R["effective_sequences"]

                    if n_eff is not None:
                        stat_df.loc[0, "N_eff"] = n_eff

                    L = stat_df.loc[0, "num_cov"]

                    # try to get concatenation statistics in addition
                    if valid_file(concat_stat_file):
                        concat_stat_df = pd.read_csv(concat_stat_file)

                        # get and save n sequences per monomer aln
                        n_seqs_1 = concat_stat_df.loc[0, "num_seqs_1"]
                        n_seqs_2 = concat_stat_df.loc[0, "num_seqs_2"]
                        stat_df.loc[0, "first_n_seqs"] = int(n_seqs_1)
                        stat_df.loc[0, "second_n_seqs"] = int(n_seqs_2)

                        # get and save median n paralogs per monomer aln
                        n_paralogs_1 = concat_stat_df.loc[0, "median_num_per_species_1"]
                        n_paralogs_2 = concat_stat_df.loc[0, "median_num_per_species_2"]
                        stat_df.loc[0, "median_num_per_species_1"] = n_paralogs_1
                        stat_df.loc[0, "median_num_per_species_2"] = n_paralogs_2

                    # try to get number of significant ECs in addition
                    if valid_file(ec_file):
                        ecs = pd.read_csv(ec_file)

                        #number of significant monomer Ecs
                        min_seq_dist = C["compare"]["min_sequence_distance"]
                        num_sig = len(ecs.query(
                            "abs(i-j) >= @min_seq_dist and probability >= @MIN_PROBABILITY"
                        ))

                        # number of inter-protein ECs significant
                        num_sig_inter = len(ecs.query(
                            "segment_i != segment_j and probability >= @MIN_PROBABILITY"
                        ))
                        stat_df.loc[0, "num_significant"] = int(num_sig)

                        #rank of top inter contact
                        top_inter_rank = ecs.query("segment_i != segment_j").index[0]
                        stat_df.loc[0, "top_inter_rank"] = int(top_inter_rank)

                    # try to get EC precision in addition
                    if valid_file(ec_comp_file):
                        ec_comp = pd.read_csv(ec_comp_file)
                        ec_comp_1 = ec_comp.query("segment_i == segment_j == @FIRST_SEGMENT")
                        ec_comp_2 = ec_comp.query("segment_i == segment_j == @SECOND_SEGMENT")
                        ec_comp_inter = ec_comp.query("segment_i != segment_j")

                        # use the monomer statistics files to figure out how many sites in each monomer
                        if valid_file(first_stat_file) and valid_file(second_stat_file):
                            stats_1 = pd.read_csv(first_stat_file)
                            L_1 = L = stats_1.loc[0, "num_cov"]

                            stats_2 = pd.read_csv(second_stat_file)
                            L_2 = L = stats_2.loc[0, "num_cov"]

                            # precision of monomer 1
                            stat_df.loc[0, "first_monomer_precision"] = ec_comp_1.iloc[L_1]["segmentwise_precision"]

                            # precicions of monomer 2
                            stat_df.loc[0, "second_monomer_precision"]= ec_comp_2.iloc[L_2]["segmentwise_precision"]

                            # precision of top 5 inter
                            stat_df.loc[0, "inter_precision"] = ec_comp_inter.iloc[NUM_INTER]["segmentwise_precision"]

                    # finally, append to global table
                    ali_table = pd.concat([ali_table, stat_df])

    # save ali statistics table
    table_file = prefix + "_job_statistics_summary.csv"
    lock_table = filelock.FileLock(table_file)
    with lock_table:
        ali_table.to_csv(
            table_file, index=False, float_format="%.3f"
        )

    return ali_table

PIPELINE_TO_SUMMARIZER = {
    "protein_monomer": protein_monomer,
    "protein_complex": protein_complex,
}


@click.command(context_settings=CONTEXT_SETTINGS)
# run settings
@click.argument('pipeline', nargs=1, required=True)
@click.argument('prefix', nargs=1, required=True)
@click.argument('configs', nargs=-1)
def app(**kwargs):
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
    app()
