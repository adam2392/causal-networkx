from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    """Meta enumeration."""

    def __contains__(cls, item):
        """Check that item is contained in the Enumeration."""
        try:
            cls(item)
        except ValueError:
            return False
        return True


class EdgeType(Enum, metaclass=MetaEnum):
    """Enumeration of different causal edges.

    Categories
    ----------
    arrow : str
        Signifies ">", or "<" edge. That is a normal
        directed edge.
    circle : str
        Signifies "o" endpoint. That is an uncertain edge,
        meaning it could be a tail, or an arrow.
    bidirected : str
        Signifies a bidirected edge.

    Notes
    -----
    The possible edges between two nodes thus are:

    ->, <-, <->, o->, <-o, o-o

    """

    circle_to_circle = "circle_to_circle"  # o-o
    circle_to_directed = "circle_to_directed"  # o->
    circle_to_undirected = "circle_to_undirected"  # o--
    directed_to_circle = "directed_to_circle"  # <-o
    undirected_to_circle = "undirected_to_circle"  # --o
    directed = "directed"  # -->
    undirected = "undirected"  # ---
    bidirected = "bidirected"  # <->


class EndPoint(Enum, metaclass=MetaEnum):
    """Enumeration of different causal edge endpoints.

    Categories
    ----------
    arrow : str
        Signifies arrowhead (">") endpoint. That is a normal
        directed edge (``->``), bidirected arrow (``<->``),
        or circle with directed edge (``o->``).
    circle : str
        Signifies a cirlce ("o") endpoint. That is an uncertain edge,
        which is either circle with directed edge (``o->``),
        circle with undirected edge (``o-``), or
        circle with circle edge (``o-o``).
    tail : str
        Signifies a tail ("-") endpoint. That is either
        a directed edge (``->``), or an undirected edge (``-``), or
        circle with circle edge (``-o``).

    Notes
    -----
    The possible edges between two nodes thus are:

    ->, <-, <->, o->, <-o, o-o

    In general, among all possible causal graphs, arrowheads depict
    non-descendant relationships. In DAGs, arrowheads depict direct
    causal relationships (i.e. parents/children). In ADMGs, arrowheads
    can come from directed edges, or bidirected edges
    """

    arrow = "arrow"
    circle = "circle"
    tail = "tail"


# Say we are given an adjacency 2D matrix, then
# we would have the following entries corresponding to different
# endpoints. An edge is then fully defined by its two endpoints, which
# are the ijth and jith component in the array.
#
# For example, for a PAG with selection bias:
# - adj[i,j] = 0 iff no edge btw i,j iff adj[j,i] = 0
# - adj[i,j] = 1 iff i *-> j with possibly adj[j,i] = 1, 2, 3
# - adj[i,j] = 2 iff i *-- j with possibly adj[j,i] = 1, 2, 3
# - adj[i,j] = 3 iff i *-o j with possibly adj[j,i] = 1, 2, 3
MIXED_EDGE_TO_VALUE_MAPPING = {
    None: 0,
    EndPoint.arrow.value: 1,
    EndPoint.tail.value: 2,
    EndPoint.circle.value: 3,
}
VALUE_TO_MIXED_EDGE_MAPPING = {val: key for key, val in MIXED_EDGE_TO_VALUE_MAPPING.items()}

# map pairs of endpoints to their corresponding edges
ENDPOINT_TO_EDGE_MAPPING = {
    (EndPoint.arrow.value, EndPoint.arrow.value): EdgeType.bidirected.value,
    (EndPoint.tail.value, EndPoint.arrow.value): EdgeType.directed.value,
    (EndPoint.arrow.value, EndPoint.tail.value): EdgeType.directed.value,
    (EndPoint.tail.value, EndPoint.tail.value): EdgeType.undirected.value,
    (EndPoint.circle.value, EndPoint.circle.value): EdgeType.circle_to_circle.value,
    (EndPoint.circle.value, EndPoint.arrow.value): EdgeType.circle_to_directed.value,
    (EndPoint.circle.value, EndPoint.tail.value): EdgeType.circle_to_undirected.value,
    (EndPoint.arrow.value, EndPoint.circle.value): EdgeType.directed_to_circle.value,
    (EndPoint.tail.value, EndPoint.circle.value): EdgeType.undirected_to_circle.value,
}

GRAPH_TYPES = ["dag", "cpdag", "admg", "pag"]
