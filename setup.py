from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    readme = f.read()


# for packaging files must be in a package (with init) and listed in package_data
# package-externals can be included with data_files,
# and there is a bug in pattern matching http://bugs.python.org/issue19286
# install unclear for data_files

setup(
    name='evcouplings',

    # Version:
    version='0.1.1',

    description='A Framework for evolutionary couplings analysis',
    long_description=readme,
    long_description_content_type='text/markdown',

    # The project's main homepage.
    url='https://github.com/debbiemarkslab/EVcouplings',

    # Author details
    author='Thomas Hopf, Benjamin Schubert',
    author_email='thomas_hopf@hms.harvard.edu, benjamin_schubert@hms.harvard.edu',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # The license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],

    # What EVcouplings relates to:
    keywords='evolutionary couplings analysis',

    # Specify  packages via find_packages() and exclude the tests and
    # documentation:
    packages=find_packages(),

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    include_package_data=True,
    package_data={
        'evcouplings.fold.cns_templates': ['*.*'],
        'evcouplings.couplings.scoring_models': ['*.*'],
    },

    #package_data is a lie:
    # http://stackoverflow.com/questions/7522250/how-to-include-package-data-with-setuptools-distribute

    # 'package_data' is used to also install non package data files
    # see http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    # example:
    # data_files=data_files,

    # Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # IMPORTANT: script names need to be in lower case ! ! ! (otherwise
    # deinstallation does not work)

    # Note: evcouplings.utils.app depends on the names evcouplings_runcfg
    # and evcouplings_summarize, so any change here must be applied there too!
    entry_points={
        'console_scripts': [
            'evcouplings=evcouplings.utils.app:app',
            'evcouplings_runcfg=evcouplings.utils.pipeline:app',
            'evcouplings_summarize=evcouplings.utils.summarize:app',
            'evcouplings_dbupdate=evcouplings.utils.update_database:app'
        ],
    },

    # Runtime dependencies. (will be installed by pip when EVcouplings is installed)
    #setup_requires=['setuptools>=18.2', 'numpy'],

    install_requires=['setuptools>=18.2', 'numpy',
        'pandas', 'scipy', 'numba', 'ruamel.yaml', 'matplotlib', 'requests',
        'mmtf-python', 'click', 'filelock', 'psutil', 'bokeh', 'jinja2',
        'biopython', 'seaborn', 'billiard', 'sklearn',
    ],

)
