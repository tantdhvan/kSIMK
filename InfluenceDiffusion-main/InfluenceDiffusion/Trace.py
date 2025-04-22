from __future__ import annotations
from typing import Union, Set, List, Tuple, Dict
import numpy as np
from itertools import chain

from .Graph import Graph
from .utils import multiple_union

__all__ = ["PseudoTraces", "Traces", "Trace"]


class PseudoTraces(dict):
    """A collection of pseudo traces.

    Attributes
    ----------
    pseudo_traces : Dict[int, List[Tuple[Set, Set]]]
        A dictionary mapping trace indices to lists of pseudo traces.
    """

    def __init__(self, pseudo_traces: Dict[int, List[Tuple[Set, Set]]]) -> None:
        for trace in chain.from_iterable(pseudo_traces.values()):
            assert len(trace) == 2, "Each pseudo trace should have length 2"
            assert isinstance(trace[0], set) and isinstance(trace[1], set), "Both elements of trace must be sets"
        super().__init__(pseudo_traces)


class Traces(list):
    """A list of traces associated with a graph.

    Attributes
    ----------
    graph : Graph
        The graph associated with the traces.
    """

    def __new__(cls, graph: Graph, traces: List[Union[Trace, Tuple[Set]]]) -> Traces:
        return super(Traces, cls).__new__(cls, traces)

    def __init__(self, graph: Graph, traces: List[Union[Trace, Tuple[Set]]]) -> None:
        traces = [trace if isinstance(trace, Trace) else Trace(graph, trace) for trace in traces]
        super().__init__(traces)
        self.graph = graph

    def append(self, obj: Union[Trace, Tuple[Set]]) -> None:
        """Append a new trace or tuple of sets to the list of traces.

        Parameters
        ----------
        obj : Union[Trace, Tuple[Set]]
            The trace or tuple of sets to append.
        """
        if isinstance(obj, Trace):
            super().append(obj)
        else:
            assert isinstance(obj, tuple) and all(isinstance(elem, set) for elem in obj)
            super().append(Trace(self.graph, obj))


class Trace(tuple):
    """A trace consisting of sets of activated vertices over time.

    Attributes
    ----------
    graph : Graph
        The graph associated with the trace.
    check_feasibility : bool
        Whether to check the feasibility of the trace during initialization.
    """

    def __new__(cls, graph: Graph, trace: Tuple[Set], check_feasibility: bool = True) -> Trace:
        return super(Trace, cls).__new__(cls, trace)

    def __init__(self, graph: Graph, trace: Tuple[Set], check_feasibility: bool = True) -> None:
        self.graph = graph
        self.check_feasibility = check_feasibility
        
        if check_feasibility:
            self.make_feasibility_check()

        self.__cumulative_trace: Tuple[Set] = None
        self.__failed_vertices: Set = None
        self.__activated_vertices: Set = None
        self.__activation_time_dict: Dict[int, int] = None

    def make_feasibility_check(self) -> None:
        """Check if each vertex has at least one active parent at each time step."""
        for t, (set_t, set_tp1) in enumerate(zip(self[:-1], self[1:])):
            for vertex in set_tp1:
                parents = self.graph.get_parents(vertex)
                assert len(set_t.intersection(parents)) != 0, \
                    f"Vertex {vertex} has no parents in newly active set at time {t}"

    @property
    def length(self) -> int:
        """Return the number of time steps in the trace."""
        return len(self) - 1

    def cum_union(self) -> Tuple[Set]:
        """Calculate the cumulative union of activated vertices over time.

        Returns
        -------
        Tuple[Set]
            A tuple containing sets of activated vertices at each time step.
        """
        if self.__cumulative_trace is None:
            cumulative_trace = [self[0]]
            for new_activated_vertices in self[1:]:
                cur_activated_vertices = cumulative_trace[-1].union(new_activated_vertices)
                cumulative_trace.append(cur_activated_vertices)
            self.__cumulative_trace = tuple(cumulative_trace)
        return self.__cumulative_trace

    def get_num_activated_vertices(self) -> int:
        """Get the total number of activated vertices.

        Returns
        -------
        int
            The number of activated vertices.
        """
        return len(self.get_all_activated_vertices())

    def get_all_activated_vertices(self) -> Set:
        """Get all activated vertices throughout the trace.

        Returns
        -------
        Set
            A set of all activated vertices.
        """
        if self.__activated_vertices is None:
            if self.__cumulative_trace is not None:
                self.__activated_vertices = self.__cumulative_trace[-1]
            else:
                self.__activated_vertices = multiple_union(self)
        return self.__activated_vertices

    def get_all_activated_vertices_no_seed(self) -> Set:
        """Get all activated vertices excluding the seed vertices.

        Returns
        -------
        Set
            A set of all activated vertices except the initial seed.
        """
        return self.get_all_activated_vertices().difference(self[0])

    def get_all_failed_vertices(self) -> Set:
        """Get all failed vertices that were activated.

        Returns
        -------
        Set
            A set of failed vertices.
        """
        if self.__failed_vertices is None:
            failed_vertices = set()
            all_activated_vertices = self.get_all_activated_vertices()
            for vertex in all_activated_vertices:
                failed_vertices.update(self.graph.get_children(vertex))
            self.__failed_vertices = failed_vertices.difference(all_activated_vertices)
        return self.__failed_vertices

    def get_all_failed_and_activated_vertices(self) -> Set:
        """Get all vertices that are either failed or activated.

        Returns
        -------
        Set
            A set of failed and activated vertices.
        """
        return self.get_all_failed_vertices().union(self.get_all_activated_vertices())

    def get_all_failed_and_activated_vertices_no_seed(self) -> Set:
        """Get all vertices that are failed or activated, excluding the seed.

        Returns
        -------
        Set
            A set of failed and activated vertices except the initial seed.
        """
        return self.get_all_failed_vertices().union(self.get_all_activated_vertices_no_seed())

    def get_active_parents_at_time(self, vertex: int, time: int) -> Set:
        """Get all active parents of a vertex after a certain time.

        Parameters
        ----------
        vertex : int
            The vertex to check.
        time : int
            The time step to check.

        Returns
        -------
        Set
            A set of active parents at the specified time.
        """
        assert int(time) == time and time >= -1, "time should be an integer >= -1"
        if time == -1:
            return set()
        return self.cum_union()[time].intersection(self.graph.get_parents(vertex))

    def get_active_parents_mask_at_time(self, vertex: int, time: int) -> np.ndarray:
        """Get a mask of active parents for a vertex at a certain time.

        Parameters
        ----------
        vertex : int
            The vertex to check.
        time : int
            The time step to check.

        Returns
        -------
        np.ndarray
            A boolean mask of active parents.
        """
        active_parents_at_time = self.get_active_parents_at_time(vertex, time)
        return self.graph.get_edges_mask_from_set_to_vertex(active_parents_at_time, vertex)

    def get_active_parents_edge_indices_at_time(self, vertex: int, time: int) -> np.ndarray:
        """Get the edge indices of active parents for a vertex at a certain time.

        Parameters
        ----------
        vertex : int
            The vertex to check.
        time : int
            The time step to check.

        Returns
        -------
        np.ndarray
            The indices of edges connected to active parents.
        """
        return np.arange(self.graph.count_edges())[self.get_active_parents_mask_at_time(vertex, time)]

    def get_newly_active_parents_at_time(self, vertex: int, time: int) -> Set:
        """Get newly activated parents of a vertex at a certain time.

        Parameters
        ----------
        vertex : int
            The vertex to check.
        time : int
            The time step to check.

        Returns
        -------
        Set
            A set of newly activated parents at the specified time.
        """
        assert int(time) == time and time >= -1, "time should be an integer >= -1"
        if time == -1:
            return set()
        return self[time].intersection(self.graph.get_parents(vertex))

    def get_newly_active_parents_mask_at_time(self, vertex: int, time: int) -> np.ndarray:
        """Get a mask of newly active parents for a vertex at a certain time.

        Parameters
        ----------
        vertex : int
            The vertex to check.
        time : int
            The time step to check.

        Returns
        -------
        np.ndarray
            A boolean mask of newly active parents.
        """
        newly_active_parents_at_time = self.get_newly_active_parents_at_time(vertex, time)
        return self.graph.get_edges_mask_from_set_to_vertex(newly_active_parents_at_time, vertex)

    def get_newly_active_parents_edge_indices_at_time(self, vertex: int, time: int) -> np.ndarray:
        """Get the edge indices of newly active parents for a vertex at a certain time.

        Parameters
        ----------
        vertex : int
            The vertex to check.
        time : int
            The time step to check.

        Returns
        -------
        np.ndarray
            The indices of edges connected to newly active parents.
        """
        return np.arange(self.graph.count_edges())[self.get_newly_active_parents_mask_at_time(vertex, time)]

    def get_vertex_activation_time(self, vertex: int) -> int:
        """Get the time step at which a vertex was activated.

        Parameters
        ----------
        vertex : int
            The vertex to check.

        Returns
        -------
        int
            The activation time of the vertex.
        """
        assert vertex in self.get_all_activated_vertices(), "vertex not activated"
        return self.get_activation_time_dict()[vertex]

    def get_activation_time_dict(self) -> Dict[int, int]:
        """Get a dictionary mapping each activated vertex to its activation time.

        Returns
        -------
        Dict[int, int]
            A dictionary of vertices and their activation times.
        """
        if self.__activation_time_dict is None:
            time_array = np.empty((0, 2), dtype=int)
            for time, node_set in enumerate(self):
                new_nodes_time_array = np.vstack([list(node_set), time * np.ones(len(node_set), dtype=int)]).T
                time_array = np.append(time_array, new_nodes_time_array, axis=0)
            self.__activation_time_dict = dict(time_array)
        return self.__activation_time_dict

    def is_edge_between_subseq_active_nodes(self, edge: Tuple[int, int]) -> bool:
        """Check if an edge connects two subsequent active nodes.

        Parameters
        ----------
        edge : Tuple[int, int]
            The edge defined by a source and a sink vertex.

        Returns
        -------
        bool
            True if the edge connects active nodes at subsequent times, False otherwise.
        """
        assert self.graph.is_edge_in_graph(edge), "no edge in graph"
        if not set(edge).issubset(self.get_all_activated_vertices()):
            return False

        activation_time_dict = self.get_activation_time_dict()
        source, sink = edge
        source_time, sink_time = activation_time_dict[source], activation_time_dict[sink]
        return sink_time - source_time == 1

    def is_edge_from_unseen_to_failed_node(self, edge: Tuple[int, int]) -> bool:
        """Check if an edge goes from an unseen node to a failed node.

        Parameters
        ----------
        edge : Tuple[int, int]
            The edge defined by a source and a sink vertex.

        Returns
        -------
        bool
            True if the edge goes from an unseen node to a failed node, False otherwise.
        """
        assert self.graph.is_edge_in_graph(edge), "no edge in graph"
        source, sink = edge
        return sink in self.get_all_failed_vertices() and source not in self.get_all_activated_vertices()

    def get_attempted_edges_mask(self) -> np.ndarray:
        """Get a mask indicating which edges were attempted for activation.

        Returns
        -------
        np.ndarray
            A boolean mask for attempted edges.
        """
        attempted_edges = self.get_edges_with_activation_attempt()
        return np.array([(edge in attempted_edges) for edge in self.graph.get_edges()], dtype=bool)

    def was_attempt_through_edge(self, edge: Tuple[int, int]) -> bool:
        """Check if activation was attempted through a given edge.

        Parameters
        ----------
        edge : Tuple[int, int]
            The edge defined by a source and a sink vertex.

        Returns
        -------
        bool
            True if activation was attempted through the edge, False otherwise.
        """
        assert self.graph.is_edge_in_graph(edge)
        activation_time_dict = self.get_activation_time_dict()
        source, sink = edge
        if source not in activation_time_dict:
            return False

        source_time = activation_time_dict[source]
        return (source in self.get_active_parents_at_time(sink, source_time) and
                sink not in self[source_time])

    def get_edges_with_activation_attempt(self) -> Set[Tuple[int, int]]:
        """Get edges where activation attempts occurred.

        Returns
        -------
        Set[Tuple[int, int]]
            A set of edges where activation attempts were made.
        """
        attempt_edges_mask = np.array([self.was_attempt_through_edge(edge) for edge in self.graph.edge_array])
        return self.graph.edge_array[attempt_edges_mask]

    def get_edges_between_subseq_active_nodes(self) -> Set[Tuple[int, int]]:
        """Get edges that connect subsequent active nodes.

        Returns
        -------
        Set[Tuple[int, int]]
            A set of edges between subsequent active nodes.
        """
        active_sources_mask = np.isin(self.graph.edge_array[:, 0], list(self.get_all_activated_vertices()))
        return {tuple(edge) for edge in self.graph.edge_array[active_sources_mask]
                if self.is_edge_between_subseq_active_nodes(edge)}

    def get_edges_from_unseen_to_failed_nodes(self) -> Set[Tuple[int, int]]:
        """Get edges that go from unseen nodes to failed nodes.

        Returns
        -------
        Set[Tuple[int, int]]
            A set of edges from unseen to failed nodes.
        """
        failed_sink_mask = np.isin(self.graph.edge_array[:, 1], list(self.get_all_failed_vertices()))
        return {tuple(edge) for edge in self.graph.edge_array[failed_sink_mask]
                if self.is_edge_from_unseen_to_failed_node(edge)}
