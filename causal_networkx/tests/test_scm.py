import pytest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.stats import multiscale_graphcorr

from causal_networkx.cgm import CausalGraph
from causal_networkx.scm import StructuralCausalModel

seed = 12345
rng = np.random.RandomState(seed=seed)


def test_scm_errors():
    """Test SCM class and errors it should raise."""
    func_uxy = rng.uniform
    func_x = lambda u_xy: 2 * u_xy
    func_y = lambda x, u_xy: x ^ u_xy

    # initialize the SCM with incorrect exogenous
    with pytest.raises(ValueError, match="Endogenous functions"):
        StructuralCausalModel(
            exogenous={
                "u_x": func_uxy,
            },
            endogenous={"x": func_x, "y": func_y},
        )

    # initialize the SCM with incorrect endogenous
    with pytest.raises(ValueError, match="Endogenous functions"):
        StructuralCausalModel(
            exogenous={
                "u_xy": func_uxy,
            },
            endogenous={"z": func_x, "y": func_y},
        )


def test_scm_induced_graph():
    """Test inducing a causal graph."""
    func_uxy = rng.uniform
    func_uz = rng.uniform
    func_x = lambda u_xy: 2 * u_xy
    func_y = lambda x, u_xy: x
    func_z = lambda u_z: u_z**2

    scm = StructuralCausalModel(
        exogenous={
            "u_xy": func_uxy,
            "u_z": func_uz,
        },
        endogenous={"x": func_x, "y": func_y, "z": func_z},
    )
    G = scm.get_causal_graph()

    expected_c_comps = [("x", "y")]

    assert isinstance(G, CausalGraph)
    assert G.nodes == set("x", "y", "z")
    assert G.c_components == expected_c_comps


def test_scm_sampling():
    """Test sampling from an SCM."""
    func_uxy = rng.uniform
    func_uz = rng.uniform
    func_x = lambda u_xy: 2 * u_xy
    func_y = lambda x, u_xy: x
    func_z = lambda u_z: u_z**2

    scm = StructuralCausalModel(
        exogenous={
            "u_xy": func_uxy,
        },
        endogenous={"x": func_x, "y": func_y},
    )
    sample_df = scm.sample()
    assert sample_df.shape == (1000, 3)
    assert_array_equal(sample_df["y"].values, sample_df["x"].values)

    # test that sampling preserves conditional independence
    scm = StructuralCausalModel(
        exogenous={
            "u_xy": func_uxy,
            "u_z": func_uz,
        },
        endogenous={"x": func_x, "y": func_y, "z": func_z},
    )
    sample_df = scm.sample(n=10)
    x = sample_df["x"].values
    y = sample_df["y"].values
    z = sample_df["z"].values

    # independent variables
    _, pvalue, _ = multiscale_graphcorr(x, z)
    assert pvalue > 0.05
    _, pvalue, _ = multiscale_graphcorr(y, z)
    assert pvalue > 0.05

    # dependent variables
    _, pvalue, _ = multiscale_graphcorr(x, y)
    assert pvalue < 0.05
