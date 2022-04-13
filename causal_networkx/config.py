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

    arrow = "arrow"
    circle = "circle"
    bidirected = "bidirected"
