import pandas as pd
import pytest

from causal_networkx import ADMG
from causal_networkx.ci import Oracle
from causal_networkx.discovery import PsiFCI


@pytest.mark.skip(reason="Doesnt work yet")
def test_psifci_oracle():
    """Test the Psi-FCI algorithm with an Oracle.

    From Figure 1a and 2, test the Psi-FCI algorithm.
    Reference: https://causalai.net/r67.pdf
    """
    edges = [
        ("Z", "X"),
        ("Z", "Y"),
        ("X", "Y"),
        ("X", "W"),
        ("Y", "W"),
    ]
    bidirected_edges = [("X", "W")]
    oracle_graph = ADMG(incoming_graph_data=edges, incoming_latent_data=bidirected_edges)
    oracle = Oracle(oracle_graph)

    # observational + interventional structure learning algorithms require
    # multiple dataframes per intervention set
    df = oracle_graph.dummy_sample()

    intervention_set = {
        ("X",): df,
        (None,): df,
    }

    fcialg = PsiFCI(ci_estimator=oracle)
    fcialg.fit(intervention_set)


def _intervention_errors():
    edges = [
        ("Z", "X"),
        ("Z", "Y"),
        ("X", "Y"),
        ("X", "W"),
        ("Y", "W"),
    ]
    bidirected_edges = [("X", "W")]
    oracle_graph = ADMG(incoming_graph_data=edges, incoming_latent_data=bidirected_edges)
    oracle = Oracle(oracle_graph)

    fcialg = PsiFCI(ci_estimator=oracle)
    df = {(None,): oracle_graph.dummy_sample(), ("X",): pd.DataFrame()}

    with pytest.raises(RuntimeError, match="All dataset distributions should have"):
        fcialg.fit(df)
