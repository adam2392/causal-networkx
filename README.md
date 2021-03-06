[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CircleCI](https://circleci.com/gh/adam2392/causal-networkx/tree/main.svg?style=svg)](https://circleci.com/gh/adam2392/causal-networkx/tree/main)
[![Main](https://github.com/adam2392/causal-networkx/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/adam2392/causal-networkx/actions/workflows/main.yml)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/adam2392/causal-networkx/branch/main/graph/badge.svg?token=H1reh7Qwf4)](https://codecov.io/gh/adam2392/causal-networkx)

# causal-networkx

Causal-Networkx is a Python graph library that extends [networkx](https://github.com/networkx/networkx) to implement causal graphical structures.

Causal-networkx does not directly subclass networkx graphs because they inherently do not support mixed edge graphs. Moreover, there are certain "graph algorithms" in networkx that would not work with mixed edge graphs. However, for the purposes of causal inference, only certain graph semantics are needed. Thus we have a lightweight library for causal graph representations that leverage the robustness and efficiency of networkx.

## Why?

Representation of causal inference models in Python are severely lacking. Moreover, sampling from causal models is non-trivial. However, sampling from simulations is a requirement to benchmark different structural learning, causal ID, or other causal related algorithms.

This package aims at serving as a framework for representing causal models and sampling from causal models.

`causal-networkx` interfaces with other popular Python packages, such as `networkx` for graphical representations.

# Documentation

See the [development version documentation](https://adam2392.github.io/causal-networkx/dev/index.html).

Or see [stable version documentation](https://adam2392.github.io/causal-networkx/stable/index.html)

# Installation

Installation is best done via `pip` or `conda`. For developers, they can also install from source using `pip`. See [installation page](TBD) for full details.

## Dependencies

Minimally, causal-networkx requires:

    * Python (>=3.8)
    * NumPy
    * SciPy
    * Networkx
    * Pandas

For extra functionality, see the `extra-requirements.txt` for additional
packages that one might install.

## User Installation

If you already have a working installation of numpy, scipy and networkx, the easiest way to install causal-networkx is using `pip`:

    # doesn't work until we make an official release :p
    pip install -U causal-networkx

To install the package from github, clone the repository and then `cd` into the directory:

    pip install -e .

# Current Limitations and Current Roadmap

Currently, selection bias representation is not implemented in the graphs and corresponding algorithms. However, I believe it is technically feasible based on the design of how we use networkx.

The roadmap currently is to integrate the next phases of causal ID, estimation, refutation and experimental design with the py-why organization and its package "dowhy".