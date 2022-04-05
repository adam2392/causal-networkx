import pytest

import numpy as np
import pandas as pd
import networkx as nx

from causal_networkx.scm import StructuralCausalModel
from causal_networkx.discovery import FCI
from causal_networkx.ci import g_square_binary, g_square_discrete, Oracle


class Test_FCI:
    def setup(self):
        seed = 12345
        rng = np.random.RandomState(seed=seed)

        func_uz = lambda: rng.negative_binomial(n=1, p=0.25)
        func_uxy = lambda: rng.binomial(n=1, p=0.4)
        func_x = lambda u_xy: 2 * u_xy
        func_y = lambda x, u_xy, z: x * u_xy + z
        func_z = lambda u_z: u_z

        # construct the SCM and the corresponding causal graph
        scm = StructuralCausalModel(
            exogenous={
                "u_xy": func_uxy,
                "u_z": func_uz,
            },
            endogenous={"x": func_x, "y": func_y, "z": func_z},
        )
        G = scm.get_causal_graph()
        oracle = Oracle(G)

        self.scm = scm
        self.G = G
        self.ci_estimator = oracle.ci_test

    def test_fci(self):
        sample = self.scm.sample(n=1000)
        G = self.G
        ci_estimator = self.ci_estimator
        print(self.scm.symbolic_runtime)
        print(sample.head)
        fci = FCI(ci_estimator=ci_estimator)
        fci.fit(sample)
