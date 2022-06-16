import networkx as nx

from causal_networkx import ADMG, CPDAG, DAG, PAG


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
    from causal_networkx.ci import Oracle
    from causal_networkx.discovery import PC

    oracle = Oracle(graph)
    dummy_X = graph.dummy_sample()

    # run PC algorithm with oracle to obtain the CPDAG
    pcalg = PC(oracle)
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
    from causal_networkx.ci import Oracle
    from causal_networkx.discovery import FCI

    oracle = Oracle(graph)
    dummy_X = graph.dummy_sample()

    # run FCI algorithm with oracle to obtain the PAG
    fci = FCI(oracle)
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
    graph : instance of DAG
        Causal graph.
    other_graph : instance of DAG
        Another causal graph to compare to.

    Returns
    -------
    is_mec : bool
        If the two graphs are markov equivalent.
    """
    from causal_networkx.algorithms.dag import compute_v_structures

    # See: https://graphical-models.readthedocs.io/en/latest/_modules/graphical_models/classes/mags/ancestral_graph.html#AncestralGraph.markov_blanket_of  # noqa
    # first check skeleton
    first_skel = graph.to_adjacency_graph()
    second_skel = other_graph.to_adjacency_graph()
    same_skeleton = nx.is_isomorphic(first_skel, second_skel)
    if not same_skeleton:
        return False

    # second check v-structures
    first_vstructs = compute_v_structures(graph)
    second_vstructs = compute_v_structures(other_graph)
    same_vstructures = first_vstructs == second_vstructs
    if not same_vstructures:
        return False

    # TODO: implement discriminating paths
    # third check discriminating triples if needed
    # self_discriminating_paths = self.discriminating_paths()
    # other_discriminating_paths = other.discriminating_paths()
    # shared_disc_paths = set(self_discriminating_paths.keys()) & set(other_discriminating_paths)
    # same_discriminating = all(
    #     self_discriminating_paths[path] == other_discriminating_paths[path]
    #     for path in shared_disc_paths
    # )

    # fourth if interventional, then must check that
    # interventional distributions are consistent
    return True
