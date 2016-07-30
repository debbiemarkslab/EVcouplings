"""
Standalone alignment calculation tool

# TODO: add logging here
# TODO: add exception handling here

Authors:
  Thomas A. Hopf
"""

import argparse


def run():
    """
    Run buildali after parsing config and command line arguments

    # TODO: parameters this takes: config files...
    # and command-line overrides for all of
    # the most important parameters that change frequently

    # TODO: how to name output files (prefix)?
    # should this somehow contain the config file name?

    # TODO: how to get alignment statistics and plots?
    """
    p = argparse.ArgumentParser(description="Calculate multiple sequence sequence alignment")
    p.add_argument("-p", "--sequence_id", default=None, help="ID/Name of sequence")
    p.add_argument("-f", "--sequence_file", default=None, help="Sequence file")
    p.add_argument("-c", "--config_file", default=None, help="Configuration file")
    args = p.parse_args()

    print(vars(args))
    print(args.keyvalues)
    print(args.protein_id)
    return args

if __name__ == "__main__":
    run()

"""
def greet():
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    greet()
"""
