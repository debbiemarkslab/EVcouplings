[![Build Status](https://travis-ci.org/debbiemarkslab/EVcouplings.svg?branch=master)](https://travis-ci.org/debbiemarkslab/EVcouplings)
# EVcouplings

Predict protein structure, function and mutations using evolutionary sequence covariation.

## Installation and setup

### Installing the Python package

If you are simply interested in using EVcouplings as a library, installing the Python package is all you need to do (unless you use functions that depend on external tools). If you want to run the *evcouplings* application (alignment generation, model parameter inference, structure prediction, etc.) you will also need to follow the sections on installing external tools and databases.

#### Requirements

EVcouplings requires a Python >= 3.5 installation. Since it depends on some packages from the scientific Python stack that can be tricky to install using pip (numba, numpy), we recommend using the [Anaconda Python distribution](https://www.continuum.io/downloads).

#### Installation

To install the latest official release of EVcouplings from PyPI, run

    pip install evcouplings

To obtain the latest version of EVcouplings from the github repository, run

    pip install git+https://github.com/debbiemarkslab/EVcouplings.git

and to update to the latest version after previously installing EVcouplings from the repository, run

    pip install -U --no-deps git+https://github.com/debbiemarkslab/EVcouplings.git


### External software tools

*After installation and before running compute jobs, the paths to the respective binaries of the following external tools have to be set in your EVcouplings job configuration file(s).*

#### plmc (required)

Tool for inferring undirected statistical models from sequence variation. Download and install plmc to a directory of your choice from the [plmc github repository](https://github.com/debbiemarkslab/plmc) according to the included documentation.

For compatibility with evcouplings, please compile using

    make all-openmp32


#### jackhmmer (required)

Download and install HMMER from the [HMMER webpage](http://hmmer.org/download.html) to a directory of your choice.

#### HHsuite (optional)

evcouplings uses the hhfilter tool to filter sequence alignments. Installation is only required if you need this functionality.

Download and install HHsuite from the [HHsuite github repository](https://github.com/soedinglab/hh-suite) to a directory of your choice.

#### CNSsolve 1.21 (optional)

evcouplings uses CNSsolve for computing 3D structure models from coupled residue pairs. Installation is only required if you want to run the *fold* stage of the computational pipeline.

Download and unpack a compiled version of [CNSsolve 1.21](http://cns-online.org/v1.21/) to a directory of your choice. No further setup is necessary, since evcouplings takes care of setting the right environment variables internally without relying on the included shell script cns_solve_env
(you will have to put the path to the cns binary in your job config file however, e.g. cns_solve_1.21/intel-x86_64bit-linux/bin/cns).

#### PSIPRED (optional)

evcouplings uses PSIPRED for secondary structure prediction, to generate secondary structure distance and dihedral angle restraints for 3D structure computation.
Installation is only required if you want to run the *fold* stage of the computational pipeline, and do not supply your own secondary structure predictions.

Download and install [PSIPRED](http://bioinfadmin.cs.ucl.ac.uk/downloads/psipred/) according to the instructions in the included README file.

#### maxcluster (optional)

evcouplings uses maxcluster to compare predicted 3D structure models to experimental protein structures, if there are any for the target protein or one
of its homologs. Installation is only required if you want to run the *fold* stage of the computational pipeline.
 
Download [maxcluster](http://www.sbg.bio.ic.ac.uk/~maxcluster/) and place it in a directory of your choice.

### Databases

*After download and before running compute jobs, the paths to the respective databases have to be set in your EVcouplings job configuration file(s).*

#### Automatic database setup
The *evcouplings* application minimally needs a sequence database for alignment generation, and structure mapping information for comparison of evolutionary couplings to 3D structures.

Sequence and structure mapping databases for EVcouplings can be automatically downloaded using the included command line tool *evcouplings_dbupdate*.
 This tool will fetch the UniProt (SwissProt/TrEMBL), UniRef100 and UniRef90 databases, and generate SIFTS-based structure mapping tables.

Please see

    evcouplings_dbupdate --help

for how to download the respective databases. Note that this may take a while, especially the generation of post-processed SIFTS mapping files. 

#### Other sequence databases

You can however use any sequence database of your choice in FASTA format if you prefer to. The database for any particular job needs to be defined in the job configuration file ("databases" section) and set as the input database in the "alignment" section.

#### Structure and mapping databases

Relevant PDB structures for comparison of ECs and 3D structure predictions will be automatically fetched from the web in the new compressed MMTF format on a per-job basis. You can however also pre-download the entire PDB and place the structures in a directory if you want to (and set pdb_mmtf_dir in your job configuration).

Uniprot to PDB index mapping files will be automatically generated by EVcouplings based on the SIFTS database.
You can either generate the files by running *evcouplings_dbupdate* (see above), or by pointing the sifts_mapping_table and sifts_sequence_db configuration parameters to file paths in a valid directory, and if the files do not yet exist, they will be created by fetching and integrating data from the web (this may take a while) when the pipeline is first run.

## Documentation and tutorials

Please refer to the Jupyter notebooks in the [notebooks subdirectory](https://github.com/debbiemarkslab/EVcouplings/tree/master/notebooks) on how to
* edit configuration files
* run jobs
* use EVcouplings as a Python library

Documentation for the source code is available at [readthedocs](http://evcouplings.readthedocs.io/en/latest/).

## License

EVcouplings is available under the MIT license, with the exception of the included CNS input scripts (please see [LICENSE](https://github.com/debbiemarkslab/EVcouplings/tree/master/LICENSE) for details).

## References

If you find EVcouplings useful for your research, please cite the following papers:

Marks D. S., Colwell, L. J., Sheridan, R., Hopf, T.A., Pagnani, A., Zecchina, R., Sander, C. Protein 3D structure computed from evolutionary sequence variation. *PLOS ONE* **6**(12), e28766 (2011)

Hopf T. A., Colwell, L. J., Sheridan, R., Rost, B., Sander, C., Marks, D. S. Three-dimensional structures of membrane proteins from genomic sequencing. *Cell* **149**, 1607-1621 (2012)

Marks, D. S., Hopf, T. A., Sander, C. Protein structure prediction from sequence variation. *Nature Biotechnology* **30**, 1072–1080 (2012)

Hopf, T. A., Ingraham, J. B., Poelwijk, F.J., Schärfe, C.P.I., Springer, M., Sander, C., & Marks, D. S. (2017). Mutation effects predicted from sequence co-variation. *Nature Biotechnology* **35**, 128–135 doi:10.1038/nbt.3769

## Contributors

EVcouplings is developed in the labs of [Debora Marks](http://marks.hms.harvard.edu) and [Chris Sander](http://sanderlab.org/) at Harvard Medical School.

* [Thomas Hopf](mailto:thomas.hopf@gmail.com) (development lead)
* Benjamin Schubert
* Anna Green
* Sophia Mersmann
* Charlotta Schärfe
* Agnes Toth-Petroczy
* John Ingraham
* Rob Sheridan
