
from typing import Dict, Set
from langgraph.graph import StateGraph, START, END


class EdgeDescriptor(Dict[str, Set[str]]):
    """
    Describe the edges between nodes in the graph before compilation.
    """
    def __init__(self, start: str):
        """
        Parameters:
            start (str): The name of the nodes that starts the graph.
            end (str): The name of the nodes that ends the graph.
        """
        super().__init__()
        self.add(START, start)

    def add(self, start: str, end: str):
        """
        Add an edge between two nodes in the graph.

        Parameters:
            start (str): The name of the node the edge stems from.
            end (str): The name of the node the edge goes to.
        """
        if start not in self.keys():
            self[start] = set()
        self[start].add(end)

    def attach_edges(self, graph: StateGraph):
        """Attach the edges to the graph. The nodes must exist prior to this."""
        for start, ends in self.items():
            for end in ends:
                if end == "__end__":
                    graph.add_edge(start, END)
                else:
                    graph.add_edge(start, end)
