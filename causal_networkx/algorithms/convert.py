from causal_networkx import ADMG, CPDAG, DAG, PAG
from causal_networkx.ci import Oracle
from causal_networkx.discovery import FCI, PC


def dag2cpdag(graph: DAG) -> CPDAG:
    """Convert a DAG to a completed partially directed acyclic graph.

    Runs the PC algorithm with an Oracle to obtain the CPDAG.
    This is useful to obtain the best possible CPDAG one can obtain
    for a certain graph using data.

    Parameters
    ----------
    graph : DAG
        The causal DAG.

    Returns
    -------
    cpdag : CPDAG
        The oracle CPDAG.
    """
    oracle = Oracle(graph)
    dummy_X = graph.dummy_sample()

    # run PC algorithm with oracle to obtain the CPDAG
    pcalg = PC(oracle.ci_test)
    pcalg.fit(dummy_X)
    cpdag = pcalg.graph_
    return cpdag  # type: ignore


def admg2pag(graph: ADMG) -> PAG:
    """Convert an ADMG to a PAG.

    Runs the FCI algorithm with an Oracle to obtain the PAG.
    This is useful to obtain the best possible PAG one can obtain
    for a certain graph using data.

    Parameters
    ----------
    graph : ADMG
        The causal ADMG.

    Returns
    -------
    pag : PAG
        The oracle PAG.
    """
    oracle = Oracle(graph)
    dummy_X = graph.dummy_sample()

    # run FCI algorithm with oracle to obtain the PAG
    fci = FCI(oracle.ci_test)
    fci.fit(dummy_X)
    pag = fci.graph_
    return pag  # type: ignore


def is_markov_equivalent(graph, other_graph) -> bool:
    """Check markov equivalence of two graphs.

    For DAGs, two graphs are Markov equivalent iff. they have the same skeleton
    and same v-structures.
    
    For ADMGs, two graphs are Markov equivalent iff. they are MEC in the DAG-sense
    and if whenever there is the same discriminating path for some node in both
    graphs, the node is a collider on that path in one graph iff. it is a collider
    on that path in the other graph.

    Parameters
    ----------
    graph : _type_
        _description_
    other_graph : _type_
        _description_

    Returns
    -------
    is_mec : bool
        If the two graphs are markov equivalent.
    """
    # See: https://graphical-models.readthedocs.io/en/latest/_modules/graphical_models/classes/mags/ancestral_graph.html#AncestralGraph.markov_blanket_of  # noqa
    # first check skeleton

    # second check v-structures

    # third check discriminating triples if needed
    pass