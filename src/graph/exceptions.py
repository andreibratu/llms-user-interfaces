import networkx as nx


class GraphException(Exception):

    def __init__(
        self,
        message: str,
        code: int,
        *args: object,
        graph: nx.Graph = None,
    ) -> None:
        super().__init__(*args)
        self._message = message
        self._code = code
        self._graph = graph
