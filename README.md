![](./molSimplify/icons/logo.png)
[![Pytest](https://github.com/hjkgrp/molSimplify/actions/workflows/pytest.yaml/badge.svg)](https://github.com/hjkgrp/molSimplify/actions/workflows/pytest.yaml)
[![Documentation Status](https://readthedocs.org/projects/molsimplify/badge/?version=latest)](http://molsimplify.readthedocs.io/?badge=latest)
[![Linter](https://github.com/hjkgrp/molSimplify/actions/workflows/python-linter.yaml/badge.svg)](https://github.com/hjkgrp/molSimplify/actions/workflows/python-linter.yaml)

molSimplify is an open source toolkit for the automated, first-principles screening and discovery of new inorganic molecules and intermolecular complexes. molSimplify is developed by the [Kulik Group](http://hjkgrp.mit.edu) in the [Department of Chemical Engineering](http://web.mit.edu/cheme/) at [MIT](http://web.mit.edu). The software can generate a variety of coordination complexes of metals coordinated by ligands in a mono- or multi-dentate fashion. The code can build a coordination complex directly from a central atom or functionalize a more complex structure (e.g. a porphyrin or other metal-ligand complex) by including additional ligands or replacing existing ones. molSimplify also generates inter-molecular complexes for evaluating binding interactions and generating candidate reactants and intermediates for catalyst reaction mechanism screening. molSimplify also ships neural network models that can predict the [metal-ligand bond lengths](https://pubs.rsc.org/en/content/articlehtml/2017/sc/c7sc01247k), [spin-splitting energy](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.7b08750), [frontier orbital energies](https://pubs.acs.org/doi/abs/10.1021/acs.iecr.8b04015), [spin-state dependent reaction energies](https://pubs.acs.org/doi/abs/10.1021/acscatal.9b02165), and [simulation outcomes](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00057) for octahedral transition metal complexes. See the Tutorials at the [Kulik group webpage](http://hjkgrp.mit.edu/molSimplify-tutorials) for a more complete list of jobs molSimplify can do.

## Installation

### via pip, from PyPI

Starting with version `1.7.4` molSimplify is available on [PyPI](https://pypi.org) enabled by the [openbabel-wheel](https://pypi.org/project/openbabel-wheel/) project.

```bash
pip install molSimplify
```

### via pip, from GitHub

To obtain the latest development version or if you plan to modify the code we recommend installation from GitHub.

1. Clone molSimplify source from github and change into the directory.

   ```bash
   git clone https://github.com/hjkgrp/molSimplify.git
   cd molSimplify
   ```
2. Locally install the molSimplify package using pip.
   ```bash
   pip install -e .[dev]
   ```
3. To test your installation, you can run the command below at the root directory of molSimplify. You are good to go if all the tests are passed! Note, some test will be skipped because none of the optional dependencies are installed this way.
   ```bash
   pytest
   ```

### via conda, from GitHub

The easiest way of installing molSimplify including optional dependencies such as [xtb](https://github.com/grimme-lab/xtb) is via the [Conda](https://conda.io/docs/) package management system.
1. Prerequisite: have [Anaconda or miniconda](https://www.anaconda.com/distribution/) installed on your system. **For M1 Macs, please use [Miniforge](https://github.com/conda-forge/miniforge) for Mac OSX arm64.** (We do not recommend simultaneously installing Anaconda and Miniforge - only install Miniforge.)

2. Clone molSimplify source from github and change into the directory.

   ```bash
   git clone https://github.com/hjkgrp/molSimplify.git
   cd molSimplify
   ```

3. Create a new conda environment and specify the desired Python version (we currently recommend 3.8).

   ```bash
   conda create --name molsimp python=3.8
   ```
4. Activate the conda environment you just created and update using one of the provided environment yaml files. In case you are experiencing problems using the full environment file in `devtools/conda-envs/mols.yml` **(some packages might not be available on all architectures such as M1 Macs)** try commenting the lines marked optional or switch to the minimal environment file `devtools/conda-envs/mols_minimal.yml`.
   ```bash
   conda activate molsimp
   conda env update --file devtools/conda-envs/mols.yml
   ```
5. Locally install the molSimplify package using pip.
   ```bash
   pip install -e . --no-deps
   ```
6. To test your installation, you can run the command below at the root directory of molSimplify. You are good to go if all the tests are passed!
   ```bash
   pytest
   ```

### via conda, from Anaconda
Releases of molSimplify are also available on Anaconda on the [conda-forge channel](https://anaconda.org/conda-forge/molsimplify) and the [hjkgroup channel](https://anaconda.org/hjkgroup/molsimplify).

### via docker
We also maintain an active [docker image on dockerhub](https://hub.docker.com/repository/docker/hjkgroup/molsimplify) for plug-and-play use.

For line by line instructions on an installation via docker, please visit [molSimplify installation webpage of Kulik group](http://hjkgrp.mit.edu/content/installing-molsimplify).

## Tutorials

A set of tutorials covering common use cases is available at the [Kulik group webpage](http://hjkgrp.mit.edu/molSimplify-tutorials). Note that the GUI is no longer supported, so users are encouraged to generate structures through the command line or using the Python command [startgen_pythonic](molSimplify/Scripts/generator.py).

## Documentation

Documentation for molSimplify can be found at our [readthedocs page](https://molsimplify.readthedocs.io/en/latest/).

## Citation [![DOI for Citing MDTraj](https://img.shields.io/badge/DOI-10.1002%2Fjcc.24437-blue.svg)](http://dx.doi.org/10.1002/jcc.24437)

molSimplify is research software. If you use it for work that results in a publication, please cite the following reference:

```
@Article {molSimplify,
author = {Ioannidis, Efthymios I. and Gani, Terry Z. H. and Kulik, Heather J.},
title = {molSimplify: A toolkit for automating discovery in inorganic chemistry},
journal = {Journal of Computational Chemistry},
volume = {37},
number = {22},
pages = {2106--2117},
issn = {1096-987X},
url = {http://dx.doi.org/10.1002/jcc.24437},
doi = {10.1002/jcc.24437},
year = {2016},
}

@Article{Nandy2018IECR,
author = {Nandy, Aditya and Duan, Chenru and Janet, Jon Paul and Gugler, Stefan and Kulik, Heather J.},
title = {Strategies and Software for Machine Learning Accelerated Discovery in Transition Metal Chemistry},
journal = {Industrial {\&} Engineering Chemistry Research},
volume = {57},
number = {42},
pages = {13973-13986},
issn = {0888-5885},
url = {https://doi.org/10.1021/acs.iecr.8b04015},
doi = {10.1021/acs.iecr.8b04015},
year = {2018},
}
```

If you use any machine learning (ML) models in molSimplify that results in a publication, please cite the corresponding reference in [this ML model reference page](https://github.com/hjkgrp/molSimplify/blob/master/MLmodel-reference.md).

**Note that we have disabled developers' supports for Python 2.7 and will only release conda builds on Python 3.**
