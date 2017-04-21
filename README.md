[![Build Status](https://travis-ci.com/debbiemarkslab/EVcouplings.svg?token=mAy5mus6jwBNzyN7K4jr&branch=master)](https://travis-ci.com/debbiemarkslab/EVcouplings)
# EVcouplings

Predict protein structure, function and mutations using evolutionary sequence covariation.

*Please note that this package is in an early stage and under active development. The API might change at any point without prior warning - use at your own risk :)*

## Installation and setup

### Installing the Python package

If you are simply interested in using EVcouplings as a library, installing the Python package is all you need to do (unless you use functions that depend on external tools). If you want to run the *evcouplings* application (alignment generation, model parameter inference, structure prediction, etc.) you will also need to follow the sections on installing external tools and databases.

#### Requirements

EVcouplings requires a Python >= 3.5 installation. Since it depends on some packages from the scientific Python stack that can be tricky to install using pip (numba, numpy), we recommend using the [Anaconda Python distribution](https://www.continuum.io/downloads).

#### Installation

To install the latest version of EVcouplings from the github repository, run

    pip install git+https://<your_github_user_name>@github.com/debbiemarkslab/EVcouplings.git
    
Installation from PyPI will be added at a later point.

#### Update

To update to the latest version after previously installing EVcouplings, run

    pip install -U --no-deps git+https://<your_github_user_name>@github.com/debbiemarkslab/EVcouplings.git



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

#### Sequence databases

Download and unzip at least one of the following sequence databases to any directory:
* [Uniref100](ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz)
* [Uniref90](ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz)

You can however use any sequence database of your choice in FASTA format. The database for any particular job will be set in the job configuration file.

#### Structure and mapping databases

PDB structures for comparison of ECs and 3D structure predictions will be automatically fetched from the web in the new compressed MMTF format. You can however also pre-download these files and place them in a directory if you want to (and set pdb_mmtf_dir in your job configuration).

Uniprot to PDB index mapping files will be automatically generated by EVcouplings based on the SIFTS database. Point the sifts_mapping_table and sifts_sequence_db configuration parameters to file paths in a valid directory, and if the files do not yet exist, they will be created by fetching and integrating data from the web (this may take a while).

## Usage and tutorials

Please refer to the Jupyter notebooks in the [notebooks subdirectory](https://github.com/debbiemarkslab/EVcouplings/tree/master/notebooks) on how to
* edit configuration files
* run jobs
* use EVcouplings as a Python library

## License

(TODO)

## References

If you find EVcouplings useful for your research, please consider citing the following papers:

Marks D. S., Colwell, L. J., Sheridan, R., Hopf, T.A., Pagnani, A., Zecchina, R., Sander, C. Protein 3D structure computed from evolutionary sequence variation. *PLOS ONE* **6**(12), e28766 (2011)

Hopf T. A., Colwell, L. J., Sheridan, R., Rost, B., Sander, C., Marks, D. S. Three-dimensional structures of membrane proteins from genomic sequencing. *Cell* **149**, 1607-1621 (2012)

Marks, D. S., Hopf, T. A., Sander, C. Protein structure prediction from sequence variation. *Nature Biotechnology* **30**, 1072–1080 (2012)

Hopf, T. A., Ingraham, J. B., Poelwijk, F.J., Schärfe, C.P.I., Springer, M., Sander, C., & Marks, D. S. (2017). Mutation effects predicted from sequence co-variation. *Nature Biotechnology* **35**, 128–135 doi:10.1038/nbt.3769

## Contributors

EVcouplings is developed in the labs of [Debora Marks](http://marks.hms.harvard.edu) and [Chris Sander](http://sanderlab.org/) at Harvard Medical School.

* [Thomas Hopf](mailto:thomas.hopf@gmail.com) (development lead)
* Benjamin Schubert
* Charlotta Schärfe
* Agnes Toth-Petroczy
* John Ingraham
* Anna Green
