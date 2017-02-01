"""
Create summary statistics / plots for runs from
evcouplings app

Authors:
  Thomas A. Hopf
"""

import click

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
# run settings
@click.argument('pipeline', nargs=1, required=True)
@click.argument('prefix', nargs=1, required=True)
@click.argument('configs', nargs=-1)
def run(**kwargs):
    """
    Create summary statistics for evcouplings pipeline runs
    """
    # TODO: make sure there is no overwriting/deadlocks...
    # TODO: make options?
    print(" --- SUMMARIZE --- ")
    print(kwargs)


if __name__ == '__main__':
    run()
