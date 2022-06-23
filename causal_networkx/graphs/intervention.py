from typing import Dict, List

from .cpdag import CPDAG
from .pag import PAG


class InterventionGraphMixin:
    _fnodes: Dict

    @property
    def f_nodes(self) -> List:
        return list(self._fnodes.keys())

    def add_fnode_for(self, nodes):
        """Add a F-node that points to the various nodes.

        The F-node will have a directed edge pointing to each node in 'nodes'.

        Parameters
        ----------
        nodes : list of node
            The nodes to add an F-node to. The node must already exist in the graph, else
            an error will be raised.

        Notes
        -----
        Each F-node corresponds to an existing node in the data. The ``node`` to be added
        as an F-node will be named "F_<existing node name>". So for example, if we want
        to add an F-node for node "10", then the F-node name would be "F_10". Thus one should
        rename nodes in their graph if "F_10" is an actual node in the graph already.

        One can remove F-nodes by simply calling ``remove_node``.
        """
        num_fnodes = len(self.f_nodes)
        f_node = f"F_{num_fnodes}"
        if f_node in self.nodes:
            raise RuntimeError(f"The F-node {f_node} is already in the graph. Please rename nodes.")

        for node in nodes:
            if node not in self.nodes:
                raise ValueError(f"{node} is not in the graph, so we cannot add an F-node to it.")
            self.add_node(f_node)
            self.add_edge(f_node, node)

        # keep track of F-nodes internally
        self._fnodes[f_node] = None

    def remove_node(self, n):
        super().remove_node(n)
        self._fnodes.pop(n)


class PsiCPDAG(CPDAG, InterventionGraphMixin):
    def __init__(self, incoming_graph_data=None, incoming_uncertain_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, incoming_uncertain_data, **attr)

        self._fnodes = dict()


class PsiPAG(PAG, InterventionGraphMixin):
    def __init__(
        self,
        incoming_graph_data=None,
        incoming_latent_data=None,
        incoming_uncertain_data=None,
        incoming_selection_data=None,
        **attr,
    ) -> None:
        super().__init__(
            incoming_graph_data,
            incoming_latent_data,
            incoming_uncertain_data,
            incoming_selection_data,
            **attr,
        )

        self._fnodes = dict()
