import networkx as nx
import pytest

from causal_networkx.cgm import CausalGraph


class TestGraph:
    def setup_method(self):
        # start every graph with the confounded graph
        # 0 -> 1, 0 -> 2 with 1 <--> 0
        self.Graph = CausalGraph
        incoming_latent_data = [("0", "1")]

        # build dict-of-dict-of-dict K3
        ed1, ed2 = ({}, {})
        incoming_graph_data = {0: {1: ed1, 2: ed2}}
        self.G = self.Graph(incoming_graph_data, incoming_latent_data)


class TestNetworkxGraph(TestGraph):
    """Test CausalGraph relevant networkx properties."""

    def test_data_input(self):
        G = self.Graph({1: [2], 2: [1]}, name="test")
        assert G.name == "test"
        assert G.has_edge(1, 2)
        assert G.has_edge(2, 1)

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
        assert G.dag.graph == {}

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
        G.add_bidirected_edge("1", "2")
        G.remove_bidirected_edge("1", "2", remove_isolate=False)
        assert "2" in G
        assert nx.is_isolate(G, "2")

        G.add_bidirected_edge("1", "2")
        G.remove_bidirected_edge("1", "2")
        assert "2" not in G

    def test_children_and_parents(self):
        """Test working with children and parents."""
        pass

    def test_do_intervention(self):
        """Test do interventions with causal graph."""
        pass

    def test_soft_intervention(self):
        """Test soft interventions with causal graph."""
        pass

    def test_c_components(self):
        """Test working with c-components in causal graph."""
        pass


# class TestEdgeSubgraph:
#     """Unit tests for the :meth:`Graph.edge_subgraph` method."""

#     def setup_method(self):
#         # Create a path graph on five nodes.
#         G = nx.path_graph(5)
#         # Add some node, edge, and graph attributes.
#         for i in range(5):
#             G.nodes[i]["name"] = f"node{i}"
#         G.edges[0, 1]["name"] = "edge01"
#         G.edges[3, 4]["name"] = "edge34"
#         G.graph["name"] = "graph"
#         # Get the subgraph induced by the first and last edges.
#         self.G = G
#         self.H = G.edge_subgraph([(0, 1), (3, 4)])

#     def test_correct_nodes(self):
#         """Tests that the subgraph has the correct nodes."""
#         assert [0, 1, 3, 4] == sorted(self.H.nodes())

#     def test_correct_edges(self):
#         """Tests that the subgraph has the correct edges."""
#         assert [(0, 1, "edge01"), (3, 4, "edge34")] == sorted(self.H.edges(data="name"))

#     def test_add_node(self):
#         """Tests that adding a node to the original graph does not
#         affect the nodes of the subgraph.

#         """
#         self.G.add_node(5)
#         assert [0, 1, 3, 4] == sorted(self.H.nodes())

#     def test_remove_node(self):
#         """Tests that removing a node in the original graph does
#         affect the nodes of the subgraph.

#         """
#         self.G.remove_node(0)
#         assert [1, 3, 4] == sorted(self.H.nodes())

#     def test_node_attr_dict(self):
#         """Tests that the node attribute dictionary of the two graphs is
#         the same object.

#         """
#         for v in self.H:
#             assert self.G.nodes[v] == self.H.nodes[v]
#         # Making a change to G should make a change in H and vice versa.
#         self.G.nodes[0]["name"] = "foo"
#         assert self.G.nodes[0] == self.H.nodes[0]
#         self.H.nodes[1]["name"] = "bar"
#         assert self.G.nodes[1] == self.H.nodes[1]

#     def test_edge_attr_dict(self):
#         """Tests that the edge attribute dictionary of the two graphs is
#         the same object.

#         """
#         for u, v in self.H.edges():
#             assert self.G.edges[u, v] == self.H.edges[u, v]
#         # Making a change to G should make a change in H and vice versa.
#         self.G.edges[0, 1]["name"] = "foo"
#         assert self.G.edges[0, 1]["name"] == self.H.edges[0, 1]["name"]
#         self.H.edges[3, 4]["name"] = "bar"
#         assert self.G.edges[3, 4]["name"] == self.H.edges[3, 4]["name"]

#     def test_graph_attr_dict(self):
#         """Tests that the graph attribute dictionary of the two graphs
#         is the same object.

#         """
#         assert self.G.graph is self.H.graph
