[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# causal-networkx

Causal-Networkx is a Python graph library that extends [networkx](https://github.com/networkx/networkx) to implement causal graphical structures.

## Why?
---
Representation of causal inference models in Python are severely lacking. Moreover, sampling from causal models is non-trivial. However, sampling from simulations is a requirement to benchmark different structural learning, causal ID, or other causal related algorithms.

This package aims at serving as a framework for representing causal models and sampling from causal models.

``causalscm`` interfaces with other popular Python packages, such as ``networkx`` for graphical representations.

# Documentation
---
Documentation is hosted on `readthedocs`.

# Installation
---

## Dependencies

causal-networkx requires:

    * Python (>=3.8)
    * NumPy
    * SciPy
    * Networkx

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install causal-networkx is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U causal-networkx

or `conda`:

    TBD

To install the package from github, clone the repository and then `cd` into the directory:

    pip install -e .