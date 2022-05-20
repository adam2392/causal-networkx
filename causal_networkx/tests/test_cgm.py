import networkx as nx
import numpy as np
import pytest

from causal_networkx.algorithms import d_separated
from causal_networkx.cgm import ADMG, PAG


class TestGraph:
    def setup_method(self):
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        self.Graph = ADMG
        incoming_latent_data = [(0, 1)]

        # build dict-of-dict-of-dict K3
        ed1, ed2 = ({}, {})
        incoming_graph_data = {0: {1: ed1, 2: ed2}}
        self.G = self.Graph(incoming_graph_data, incoming_latent_data)


class TestNetworkxGraph(TestGraph):
    """Test CausalGraph relevant networkx properties."""

    def test_data_input(self):
        G = self.Graph({1: [2], 2: [3]}, name="test")
        assert G.name == "test"
        assert G.has_edge(1, 2)
        assert G.has_edge(2, 3)

        with pytest.raises(RuntimeError, match="Causal DAG must be acyclic"):
            self.Graph({1: [2], 2: [1]}, name="test")

    def test_getitem(self):
        G = self.G
        assert G[0] == {1: {}, 2: {}}
        with pytest.raises(KeyError):
            G.__getitem__("j")
        with pytest.raises(TypeError):
            G.__getitem__(["A"])

    def test_add_node(self):
        G = self.Graph()
        G.add_node(0)
        assert 0 in G
        assert 0 in G.dag
        assert 0 not in G.c_component_graph
        # test add attributes
        G.add_node(1, c="red")
        G.add_node(2, c="blue")
        G.add_node(3, c="red")
        assert G.nodes[1]["c"] == "red"
        assert G.nodes[2]["c"] == "blue"
        assert G.nodes[3]["c"] == "red"
        # test updating attributes
        G.add_node(1, c="blue")
        G.add_node(2, c="red")
        G.add_node(3, c="blue")
        assert G.nodes[1]["c"] == "blue"
        assert G.nodes[2]["c"] == "red"
        assert G.nodes[3]["c"] == "blue"

    def test_add_nodes_from(self):
        G = self.Graph()
        G.add_nodes_from([0, 1, 2])
        for i in [0, 1, 2]:
            assert i in G
            assert i in G.dag
            assert i not in G.c_component_graph
        # test add attributes
        G.add_nodes_from([0, 1, 2], c="red")
        assert G.nodes[0]["c"] == "red"
        assert G.nodes[2]["c"] == "red"
        # test that attribute dicts are not the same
        assert G.nodes[0] is not G.nodes[1]
        # test updating attributes
        G.add_nodes_from([0, 1, 2], c="blue")
        assert G.nodes[0]["c"] == "blue"
        assert G.nodes[2]["c"] == "blue"
        assert G.nodes[0] is not G.nodes[1]
        # test tuple input
        H = self.Graph()
        H.add_nodes_from(G.nodes(data=True))
        assert H.nodes[0]["c"] == "blue"
        assert H.nodes[2]["c"] == "blue"
        assert H.nodes[0] is not H.nodes[1]
        # specific overrides general
        H.add_nodes_from([0, (1, {"c": "green"}), (3, {"c": "cyan"})], c="red")
        assert H.nodes[0]["c"] == "red"
        assert H.nodes[1]["c"] == "green"
        assert H.nodes[2]["c"] == "blue"
        assert H.nodes[3]["c"] == "cyan"

    def test_remove_node(self):
        G = self.G.copy()
        G.remove_node(0)
        assert 0 not in G
        with pytest.raises(nx.NetworkXError):
            G.remove_node(-1)

    def test_remove_nodes_from(self):
        G = self.G.copy()
        G.remove_nodes_from([0, 1])
        assert list(G.nodes) == [2]
        G.remove_nodes_from([-1])  # silent fail

    def test_add_edge(self):
        G = self.Graph()
        G.add_edge(0, 1)
        assert G.has_edge(0, 1)
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G.has_edge(0, 1)

    def test_add_edges_from(self):
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 2, {"weight": 3})])
        assert G.has_edge(0, 1)
        assert G.has_edge(0, 2)
        assert G[0][2]["weight"] == 3

        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 2, {"weight": 3}), (1, 2, {"data": 4})], data=2)
        assert G.has_edge(0, 1)
        assert G.has_edge(0, 2)
        assert G[0][2]["weight"] == 3
        assert G[0][2]["data"] == 2
        assert G.has_edge(1, 2)
        assert G[1][2]["data"] == 4

        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0,)])  # too few in tuple
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0, 1, 2, 3)])  # too many in tuple
        with pytest.raises(TypeError):
            G.add_edges_from([0])  # not a tuple

    def test_remove_edge(self):
        G = self.G.copy()
        G.remove_edge(0, 1)
        assert not G.has_edge(0, 1)
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)

    def test_remove_edges_from(self):
        G = self.G.copy()
        G.remove_edges_from([(0, 1)])
        assert not G.has_edge(0, 1)
        G.remove_edges_from([(0, 0)])  # silent fail

    def test_clear(self):
        G = self.G.copy()
        G.dag.graph["name"] = "K3"
        G.clear()
        assert list(G.nodes) == []
        assert all(graph.graph == {} for graph in G._graphs)

    def test_clear_edges(self):
        G = self.G.copy()
        G.dag.graph["name"] = "K3"
        nodes = list(G.nodes)
        G.clear_edges()
        assert list(G.nodes) == nodes
        assert G.dag.adj == {0: {}, 1: {}, 2: {}}
        assert list(G.edges) == []
        assert G.dag.graph["name"] == "K3"

    def test_get_edge_data(self):
        G = self.G.copy()
        assert G.get_edge_data(0, 1) == {}
        assert G[0][1] == {}
        assert G.get_edge_data(10, 20) is None
        assert G.get_edge_data(-1, 0) is None
        assert G.get_edge_data(-1, 0, default=1) == 1

    def test_contains(self):
        G = self.G
        assert 1 in G
        assert 4 not in G
        assert "b" not in G
        assert [] not in G  # no exception for nonhashable
        assert {1: 1} not in G  # no exception for nonhashable

    def test_order(self):
        G = self.G
        assert len(G) == 3
        assert G.order() == 3
        assert G.number_of_nodes() == 3

    def test_nodes(self):
        G = self.G
        assert sorted(G.nodes(data=True)) == [(0, {}), (1, {}), (2, {})]

    def test_none_node(self):
        G = self.Graph()
        with pytest.raises(ValueError):
            G.add_node(None)
        with pytest.raises(ValueError):
            G.add_nodes_from([None])
        with pytest.raises(ValueError):
            G.add_edge(0, None)
        with pytest.raises(ValueError):
            G.add_edges_from([(0, None)])

    def test_has_node(self):
        G = self.G
        assert G.has_node(1)
        assert not G.has_node(4)
        assert not G.has_node([])  # no exception for nonhashable
        assert not G.has_node({1: 1})  # no exception for nonhashable

    def test_has_edge(self):
        G = self.G
        assert G.has_edge(0, 1)
        assert not G.has_edge(0, -1)

    def test_size(self):
        G = self.G

        # size stores all edges
        assert G.size() == 3
        assert G.number_of_edges() == 2
        assert G.number_of_bidirected_edges() == 1

    def test_name(self):
        G = self.Graph(name="")
        assert G.name == ""
        G = self.Graph(name="test")
        assert G.name == "test"

    def test_str_unnamed(self):
        G = self.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        G.add_bidirected_edge(1, 2)
        assert str(G) == f"{type(G).__name__} with 3 nodes, 2 edges and 1 bidirected edges"

    def test_str_named(self):
        G = self.Graph(name="foo")
        G.add_edges_from([(1, 2), (2, 3)])
        G.add_bidirected_edge(1, 2)
        assert (
            str(G) == f"{type(G).__name__} named 'foo' with 3 nodes, 2 edges and 1 bidirected edges"
        )

    def add_attributes(self, G):
        """Test adding edges with attributes to graph."""
        G.dag.graph["foo"] = []
        G.nodes[0]["foo"] = []
        G.remove_edge(1, 2)
        ll = []
        G.add_edge(1, 2, foo=ll)
        G.add_edge(2, 1, foo=ll)


class TestCausalGraph(TestGraph):
    """Test relevant causal graph properties."""

    def test_hash(self):
        """Test hashing a causal graph."""
        G = self.G
        current_hash = hash(G)
        assert G._current_hash is None

        G.add_bidirected_edge("1", "2")
        new_hash = hash(G)
        assert current_hash != new_hash

        G.remove_bidirected_edge("1", "2")
        assert current_hash == hash(G)

    def test_full_graph(self):
        """Test computing a full graph from causal graph."""
        G = self.G
        # the current hash should match after computing full graphs
        current_hash = hash(G)
        G.compute_full_graph()
        assert current_hash == G._current_hash
        G.compute_full_graph()
        assert current_hash == G._current_hash

        # after adding a new edge, the hash should change and
        # be different
        G.add_bidirected_edge("1", "2")
        new_hash = hash(G)
        assert new_hash != G._current_hash

        # once the hash is computed, it should be the same again
        G.compute_full_graph()
        assert new_hash == G._current_hash

        # removing the bidirected edge should result in the same
        # hash again
        G.remove_bidirected_edge("1", "2")
        assert current_hash != G._current_hash
        G.compute_full_graph()
        assert current_hash == G._current_hash

        # different orders of edges shouldn't matter
        G_copy = G.copy()
        G.add_bidirected_edge("1", "2")
        G.add_bidirected_edge("2", "3")
        G_hash = hash(G)
        G_copy.add_bidirected_edge("2", "3")
        G_copy.add_bidirected_edge("1", "2")
        copy_hash = hash(G_copy)
        assert G_hash == copy_hash

    def test_bidirected_edge(self):
        """Test bidirected edge functions."""
        # add bidirected edge to an isolated node
        G = self.G
        G.add_bidirected_edge(1, 5)
        assert G.has_bidirected_edge(1, 5)
        assert G.has_bidirected_edge(5, 1)
        G.remove_bidirected_edge(1, 5, remove_isolate=False)
        assert 5 in G
        assert nx.is_isolate(G, 5)
        assert not G.has_bidirected_edge(1, 5)
        assert not G.has_bidirected_edge(5, 1)

        G.add_bidirected_edge(1, 5)
        G.remove_bidirected_edge(1, 5)
        print(G.nodes)
        assert 5 not in G

    def test_d_separation(self):
        G = self.G.copy()
        # add collider on 0
        G.add_edge(3, 0)

        # normal d-separation statements should hold
        assert not d_separated(G, 1, 2, set())
        assert d_separated(G, 1, 2, 0)

        # when we add an edge from 0 -> 1
        # there is no d-separation statement
        assert not d_separated(G, 3, 1, set())
        assert not d_separated(G, 3, 1, 0)

        # test collider works on bidirected edge
        # 1 <-> 0
        G.remove_edge(0, 1)
        assert d_separated(G, 3, 1, set())
        assert not d_separated(G, 3, 1, 0)

    # def test_add_multiple_edges(self):
    #     G = self.G
    # since there is a directed edge from

    def test_children_and_parents(self):
        """Test working with children and parents."""
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        G = self.G.copy()

        # basic parent/children semantics
        assert [1, 2] == list(G.children(0))
        assert [] == list(G.parents(0))
        assert [] == list(G.children(1))
        assert [0] == list(G.parents(1))

        # a lone bidirected edge is not a child or a parent
        G.add_bidirected_edge(2, 3)
        assert [] == list(G.parents(3))
        assert [] == list(G.children(3))

    def test_export_dot(self):
        """Test exporting to DOT format."""
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        G = self.G.copy()

        # make sure output handles a string for a node
        G.add_edge(0, "1-0")

        dot_graph = G.to_dot_graph()

        # make sure the output adheres to the DOT format
        assert dot_graph.startswith("strict digraph {")
        assert dot_graph.endswith("}")
        for node in G.nodes:
            assert f"{node};\n" in dot_graph
        for u, v in G.edges:
            if isinstance(u, str):
                u = f'"{u}"'
            if isinstance(v, str):
                v = f'"{v}"'
            assert f"{u} -> {v};\n" in dot_graph
        for u, v in G.bidirected_edges:
            assert f"{u} <-> {v};\n" in dot_graph

    def test_do_intervention(self):
        """Test do interventions with causal graph."""
        pass

    def test_soft_intervention(self):
        """Test soft interventions with causal graph."""
        pass

    def test_c_components(self):
        """Test working with c-components in causal graph."""
        pass


class TestPAG(TestCausalGraph):
    def setup_method(self):
        # setup the causal graph in previous method
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        super().setup_method()
        self.Graph = PAG
        self.PAG = PAG(self.G.dag)

        # Create a PAG: 2 <- 0 <-> 1 o-o 4
        # handle the bidirected edge from 0 to 1
        self.PAG.remove_edge(0, 1)
        self.PAG.add_bidirected_edge(0, 1)

        # also setup a PAG with uncertain edges
        self.PAG.add_circle_edge(1, 4, bidirected=True)

    def test_neighbors(self):
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        G = self.PAG

        assert G.neighbors(2) == [0]
        assert G.neighbors(0) == [2, 1]
        assert G.neighbors(1) == [0, 4]
        assert G.neighbors(4) == [1]

    def test_wrong_construction(self):
        # PAGs only allow one type of edge between any two nodes
        edge_list = [
            ("x4", "x1"),
            ("x2", "x5"),
        ]
        latent_edge_list = [("x1", "x2"), ("x4", "x5"), ("x4", "x1")]
        with pytest.raises(RuntimeError, match="There are multiple edges"):
            PAG(edge_list, incoming_latent_data=latent_edge_list)

    def test_hash_with_circles(self):
        # 2 <- 0 <-> 1 o-o 4
        G = self.PAG
        current_hash = hash(G)
        assert G._current_hash is None

        G.add_circle_edge(2, 3, bidirected=True)
        new_hash = hash(G)
        assert current_hash != new_hash

        G.remove_circle_edge(2, 3, bidirected=True)
        assert current_hash == hash(G)

    def test_add_circle_edge(self):
        G = self.PAG
        assert not G.has_edge(1, 3)

        # if we try to add a circle edge to a new node
        # where there is no arrow already without specifying
        # bidirected, then an error will be raised
        with pytest.raises(RuntimeError, match="There is no directed"):
            G.add_circle_edge(1, 3)
        G.add_circle_edge(1, 3, bidirected=True)
        assert not G.has_edge(1, 3)
        assert G.has_circle_edge(1, 3)

    def test_adding_edge_errors(self):
        """Test that adding edges in PAG result in certain errors."""
        # 2 <- 0 <-> 1 o-o 4
        G = self.PAG

        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 2"):
            G.add_circle_edge(0, 2)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 1"):
            G.add_circle_edge(0, 1)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 1"):
            G.add_circle_edges_from([(0, 1)])
        with pytest.raises(RuntimeError, match="There is already an existing edge between 1 and 4"):
            G.add_edge(1, 4)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 1"):
            G.add_edges_from([(0, 1)])
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 2"):
            G.add_bidirected_edge(0, 2)
        with pytest.raises(RuntimeError, match="There is already an existing edge between 0 and 2"):
            G.add_bidirected_edges_from([(0, 2)])
        with pytest.raises(RuntimeError, match="There is already an existing edge between 1 and 4"):
            G.add_bidirected_edges_from([(1, 4)])
        with pytest.raises(RuntimeError, match="There is an existing 0 -> 2"):
            # adding an edge from 2 -> 0, will result in an error
            G.add_edge(2, 0)

        # adding a single circle edge is fine
        G.add_circle_edge(2, 0)

    def test_remove_circle_edge(self):
        G = self.PAG
        assert G.has_circle_edge(1, 4)
        G.remove_circle_edge(1, 4)
        assert not G.has_circle_edge(1, 4)

    def test_orient_circle_edge(self):
        G = self.PAG
        G.orient_circle_edge(1, 4, "arrow")
        assert G.has_edge(1, 4)
        assert not G.has_circle_edge(1, 4)

        with pytest.raises(ValueError, match="edge_type must be"):
            G.orient_circle_edge(1, 4, "circl")
        assert G.has_edge(1, 4)
        assert not G.has_circle_edge(1, 4)

    def test_m_separation(self):
        G = self.PAG
        assert not d_separated(G, 0, 4, set())
        assert not d_separated(G, 0, 4, 1)

        # check various cases
        G.add_edge(4, 3)
        assert not d_separated(G, 3, 1, set())
        assert d_separated(G, 3, 1, 4)

        # check what happens in the other direction
        G.remove_edge(4, 3)
        G.add_edge(3, 4)
        assert not d_separated(G, 3, 1, set())
        assert not d_separated(G, 3, 1, 4)

    def test_children_and_parents(self):
        """Test working with children and parents."""
        # 2 <- 0 <-> 1 o-o 4
        G = self.PAG.copy()

        # basic parent/children semantics
        assert [2] == list(G.children(0))
        assert [] == list(G.parents(0))
        assert [] == list(G.children(1))
        assert [] == list(G.parents(1))
        assert [] == list(G.parents(4))
        assert [] == list(G.children(4))

        # o-o edges do not constitute possible parent/children
        assert [] == list(G.possible_children(1)) == list(G.possible_parents(1))
        assert [] == list(G.possible_children(4)) == list(G.possible_parents(4))

        # when the parental relationship between 2 and 0
        # is made uncertain, the parents/children sets reflect
        G.add_circle_edge(2, 0)
        assert [] == list(G.children(0))
        assert [] == list(G.parents(2))

        # 2 and 0 now have possible children/parents relationship
        assert [0] == list(G.possible_parents(2))
        assert [2] == list(G.possible_children(0))

    def test_export_numpy(self):
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        G = self.PAG.copy()

        numpy_graph = G.to_numpy_array()
        expected_arr = np.zeros((3, 3))
        expected_arr[0, 1] = 2
        expected_arr[0, 2] = 2
        expected_arr[1, 0] = 1
