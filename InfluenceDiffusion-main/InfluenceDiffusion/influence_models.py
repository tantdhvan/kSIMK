import numpy as np
from copy import deepcopy
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Dict, Set, Union, List
from joblib import Parallel, delayed

from Graph import Graph
from Trace import Trace, Traces


class InfluenceModel:
    """Base class for influence models on a graph.

    Parameters
    ----------
    g : Graph
        The graph on which the influence model operates.
    check_init : bool, optional
        Whether to check the parameters upon initialization (default is True).
    random_state : int, optional
        Seed for random number generation (default is None).
    n_jobs : int, optional
        Number of parallel jobs for simulations (default is None).

    Attributes
    ----------
    g : Graph
        The graph on which the influence model operates.
    vertices : List[int]
        The list of vertices in the graph.
    random_state : int
        Seed for random number generation.
    n_jobs : int
        Number of parallel jobs for simulations.
    seed_set : Set[int]
        Set of initial active nodes.
    _cur_influence_nodes : Set[int]
        Nodes currently influencing others.
    all_influenced_nodes : Set[int]
        All nodes that have been influenced.
    _propagation_trace : List[Set[int]]
        Trace of the propagation process.
    """

    def __init__(self, g: Graph, check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        self.g = g
        self.vertices = list(self.g.get_vertices())
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._edge_2_index = {edge: idx for idx, edge in enumerate(g.edges)}
        if check_init:
            self.check_param_init_correctness()

    def check_param_init_correctness(self) -> None:
        """Check the correctness of the model's parameters.

        Raises
        ------
        NotImplementedError
            This method should be implemented in derived classes.
        """
        raise NotImplementedError()

    def _init_simulation_rvs(self, simulation_rvs=None) -> None:
        """Initialize simulation random variables.

        Parameters
        ----------
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).
        
        Raises
        ------
        NotImplementedError
            This method should be implemented in derived classes.
        """
        raise NotImplementedError()

    def _generate_simulation_rvs(self, n_runs: int = 1) -> List:
        """Generate random variables for simulation.

        Parameters
        ----------
        n_runs : int, optional
            Number of simulation runs (default is 1).

        Returns
        -------
        List
            Generated random variables for the simulation.
        
        Raises
        ------
        NotImplementedError
            This method should be implemented in derived classes.
        """
        raise NotImplementedError()

    def _pre_simulation_init(self, seed_set: Set[int], simulation_rvs=None) -> None:
        """Prepare for simulation by initializing parameters.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Raises
        ------
        AssertionError
            If the seed set is empty.
        """
        assert len(seed_set) > 0, "Seed set should contain at least one vertex"
        self.seed_set = set(seed_set)
        self._cur_influence_nodes = deepcopy(self.seed_set)
        self.all_influenced_nodes = deepcopy(self.seed_set)
        self._propagation_trace: List[Set[int]] = [self.seed_set]
        self._init_simulation_rvs(simulation_rvs)

    def _make_edge_influence_attempt(self, v: int, v_adj: int) -> bool:
        """Attempt to influence adjacent vertex.

        Parameters
        ----------
        v : int
            The current influencing vertex.
        v_adj : int
            The adjacent vertex to be influenced.

        Returns
        -------
        bool
            True if the adjacent vertex is influenced, otherwise False.
        
        Raises
        ------
        NotImplementedError
            This method should be implemented in derived classes.
        """
        raise NotImplementedError()

    def _simulate_trace(self, out_trace_type: bool = True) -> Union[Trace, List[Set[int]]]:
        """Simulate the influence spread.

        Parameters
        ----------
        out_trace_type : bool, optional
            Whether to return a Trace object (default is True).

        Returns
        -------
        Union[Trace, List[Set[int]]]
            The trace of influenced nodes or a Trace object.
        """
        while self._cur_influence_nodes:
            self._make_step()
        return Trace(self.g, tuple(self._propagation_trace)) if out_trace_type else self._propagation_trace

    def sample_trace(self, seed_set: Set[int],
                     out_trace_type: bool = True,
                     simulation_rvs=None) -> Union[Trace, List[Set[int]]]:
        """Sample a single trace from the model.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        out_trace_type : bool, optional
            Whether to return a Trace object (default is True).
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Returns
        -------
        Union[Trace, List[Set[int]]]
            The sampled trace of influenced nodes or a Trace object.
        """
        self._pre_simulation_init(seed_set, simulation_rvs=simulation_rvs)
        return self._simulate_trace(out_trace_type=out_trace_type)

    def make_simulation(self, seed_set: Set[int], simulation_rvs=None) -> Union[Trace, List[Set[int]]]:
        """Run a simulation and return the propagation trace.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Returns
        -------
        np.ndarray
            The propagation trace as an array.
        """
        self._pre_simulation_init(seed_set, simulation_rvs=simulation_rvs)
        return self._simulate_trace()

    def sample_traces(self, n_traces: int = 100,
                      seed_size_range: List[int] = None,
                      out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample multiple traces.

        Parameters
        ----------
        n_traces : int, optional
            Number of traces to sample (default is 100).
        seed_size_range : List[int], optional
            Range of seed sizes to sample from (default is None).
        out_trace_type : bool, optional
            Whether to return a Traces object (default is True).

        Returns
        -------
        Union[Traces, List[List[Set[int]]]]
            The sampled traces or a Traces object.
        """
        seed_sets = self._sample_seeds(n_seeds=n_traces, seed_size_range=seed_size_range)
        return self.sample_traces_from_seeds(seed_sets, out_trace_type=out_trace_type)

    def sample_traces_from_seeds(self, seed_sets: List[Set[int]],
                                 out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample traces from a list of seed sets.

        Parameters
        ----------
        seed_sets : List[Set[int]]
            List of seed sets for sampling.
        out_trace_type : bool, optional
            Whether to return a Traces object (default is True).

        Returns
        -------
        Union[Traces, List[List[Set[int]]]]
            The sampled traces or a Traces object.
        """
        simulation_rvs_over_runs = self._generate_simulation_rvs(n_runs=len(seed_sets))
        traces = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda seed_set, simulation_rvs:
                    self.sample_trace(seed_set=seed_set, out_trace_type=False, simulation_rvs=simulation_rvs))
            (seed_set, thresholds)
            for seed_set, thresholds in zip(seed_sets, simulation_rvs_over_runs)
        )
        return Traces(self.g, traces) if out_trace_type else traces

    def _make_step(self) -> None:
        """Make a simulation step by activating new nodes.

        This method updates the current influence nodes and the propagation trace.
        """
        new_influence_nodes = set()
        for v in self._cur_influence_nodes:
            for v_adj in self.g.get_children(v).difference(self.all_influenced_nodes):
                influence_res = self._make_edge_influence_attempt(v, v_adj)
                if influence_res:
                    new_influence_nodes.add(v_adj)

        self._cur_influence_nodes = new_influence_nodes
        self.all_influenced_nodes.update(new_influence_nodes)

        # Append to the propagation trace if there are new nodes influenced
        if new_influence_nodes:
            self._propagation_trace.append(new_influence_nodes)
class kInfluenceModel(InfluenceModel):
    """Base class for k-topic influence models."""

    def __init__(self, g: Graph, n_topics: int, check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        """
        Initialize the k-Topic Influence Model.

        Parameters
        ----------
        g : Graph
            The graph on which the influence model operates.
        n_topics : int
            The number of topics.
        check_init : bool, optional
            Whether to check the parameters upon initialization (default is True).
        random_state : int, optional
            Seed for random number generation (default is None).
        n_jobs : int, optional
            Number of parallel jobs for simulations (default is None).
        """
        super().__init__(g, check_init, random_state, n_jobs)
        self.n_topics = n_topics  # Store the number of topics
        self.vertex_2_topic = {v: None for v in self.vertices}  # Track topic for each vertex

    def _pre_simulation_init(self, seed_set: Set[int], seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> None:
        """Prepare for simulation by initializing parameters for k-topic influence."""
        super()._pre_simulation_init(seed_set, simulation_rvs=simulation_rvs)

        # Use the seed_set_with_topics to assign topics to the seed nodes
        for v in seed_set:
            if v in seed_set_with_topics:
                self.vertex_2_topic[v] = seed_set_with_topics[v]
            else:
                raise ValueError(f"Seed node {v} is missing a predefined topic assignment in seed_set_with_topics")

    def sample_trace(self, seed_set: Set[int], seed_set_with_topics: Dict[int, int], out_trace_type: bool = True, simulation_rvs=None) -> Union[Trace, List[Set[int]]]:
        """Sample a single trace from the model with predefined topics for seed nodes.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        seed_set_with_topics : Dict[int, int]
            Dictionary mapping each node in the seed set to its predefined topic.
        out_trace_type : bool, optional
            Whether to return a Trace object (default is True).
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Returns
        -------
        Union[Trace, List[Set[int]]]
            The sampled trace of influenced nodes or a Trace object.
        """
        self._pre_simulation_init(seed_set, seed_set_with_topics, simulation_rvs=simulation_rvs)
        return self._simulate_trace(out_trace_type=out_trace_type)

    def make_simulation(self, seed_set: Set[int], seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> Union[Trace, List[Set[int]]]:
        """Run a simulation and return the propagation trace.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        seed_set_with_topics : Dict[int, int]
            Dictionary mapping each node in the seed set to its predefined topic.
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Returns
        -------
        np.ndarray
            The propagation trace as an array.
        """
        self._pre_simulation_init(seed_set, seed_set_with_topics, simulation_rvs=simulation_rvs)
        return self._simulate_trace()

    def sample_traces(self, n_traces: int = 100, seed_size_range: List[int] = None, seed_set_with_topics: Dict[int, int] = None, out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample multiple traces.

        Parameters
        ----------
        n_traces : int, optional
            Number of traces to sample (default is 100).
        seed_size_range : List[int], optional
            Range of seed sizes to sample from (default is None).
        seed_set_with_topics : Dict[int, int], optional
            Dictionary mapping each node in the seed set to its predefined topic.
        out_trace_type : bool, optional
            Whether to return a Traces object (default is True).

        Returns
        -------
        Union[Traces, List[List[Set[int]]]]
            The sampled traces or a Traces object.
        """
        seed_sets = self._sample_seeds(n_seeds=n_traces, seed_size_range=seed_size_range)
        return self.sample_traces_from_seeds(seed_sets, seed_set_with_topics, out_trace_type=out_trace_type)

    def sample_traces_from_seeds(self, seed_sets: List[Set[int]], seed_set_with_topics: Dict[int, int], out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample traces from a list of seed sets.

        Parameters
        ----------
        seed_sets : List[Set[int]]
            List of seed sets for sampling.
        seed_set_with_topics : Dict[int, int]
            Dictionary mapping each node in the seed set to its predefined topic.
        out_trace_type : bool, optional
            Whether to return a Traces object (default is True).

        Returns
        -------
        Union[Traces, List[List[Set[int]]]]
            The sampled traces or a Traces object.
        """
        simulation_rvs_over_runs = self._generate_simulation_rvs(n_runs=len(seed_sets))
        traces = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda seed_set, simulation_rvs:
                    self.sample_trace(seed_set=seed_set, seed_set_with_topics=seed_set_with_topics, out_trace_type=False, simulation_rvs=simulation_rvs))
            (seed_set, thresholds)
            for seed_set, thresholds in zip(seed_sets, simulation_rvs_over_runs)
        )
        return Traces(self.g, traces) if out_trace_type else traces

class kICM(kInfluenceModel):
    """k-Topic Independent Cascade Model (kICM) for influence spread."""

    def __init__(self, g: Graph, n_topics: int, check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        """
        Initialize the k-Topic Independent Cascade Model (kICM).
        
        Parameters
        ----------
        g : Graph
            The graph on which the kICM operates.
        n_topics : int
            The number of topics.
        check_init : bool, optional
            Whether to check the parameters upon initialization (default is True).
        random_state : int, optional
            Seed for random number generation (default is None).
        n_jobs : int, optional
            Number of parallel jobs for simulations (default is None).
        """
        super().__init__(g, n_topics, check_init, random_state, n_jobs)
    def check_param_init_correctness(self) -> None:
        """Check if the graph edge weights are valid for the ICM.

        Raises
        ------
        AssertionError
            If any edge weight is not in the range [0, 1].
        """
        assert np.all((0 <= self.g.weights) & (self.g.weights <= 1))
    def _generate_simulation_rvs(self, n_runs: int = 1) -> np.ndarray:
        """Generate random activations for edges (ICM logic)."""
        if self.random_state:
            np.random.seed(self.random_state)
        edge_activations = np.random.rand(n_runs, self.g.count_edges()) <= self.g.weights
        return edge_activations[0] if n_runs == 1 else edge_activations

    def _make_edge_influence_attempt(self, v: int, v_adj: int) -> bool:
        """Attempt to influence an adjacent vertex based on edge activation."""
        # Check if the adjacent vertex has been influenced yet
        if self.vertex_2_topic[v_adj] is None:  # If not influenced
            edge_idx = self._edge_2_index.get((v, v_adj))
            # Check if the edge is activated based on the pre-generated random values
            if self.edge_activations[self._edge_2_index[(v, v_adj)]]:
                self.vertex_2_topic[v_adj] = self.vertex_2_topic[v]  # Assign topic of v to v_adj
                return True
        return False

    def _pre_simulation_init(self, seed_set: Set[int], seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> None:
        """Prepare for simulation by initializing parameters for k-topic independent cascade."""
        super()._pre_simulation_init(seed_set, seed_set_with_topics, simulation_rvs=simulation_rvs)

        # Generate edge activation probabilities for each edge
        self.edge_activations = self._generate_simulation_rvs(n_runs=1)

    def _simulate_trace(self, out_trace_type: bool = True) -> Union[Trace, List[Set[int]]]:
        """Simulate the influence spread for k-topic Independent Cascade."""
        while self._cur_influence_nodes:
            self._make_step()
        return Trace(self.g, tuple(self._propagation_trace)) if out_trace_type else self._propagation_trace

    def _make_step(self) -> None:
        """Make a simulation step by activating new nodes based on Independent Cascade model."""
        new_influence_nodes = set()
        for v in self._cur_influence_nodes:
            for v_adj in self.g.get_children(v).difference(self.all_influenced_nodes):
                influence_res = self._make_edge_influence_attempt(v, v_adj)
                if influence_res:
                    new_influence_nodes.add(v_adj)

        self._cur_influence_nodes = new_influence_nodes
        self.all_influenced_nodes.update(new_influence_nodes)

        # Append to the propagation trace if there are new nodes influenced
        if new_influence_nodes:
            self._propagation_trace.append(new_influence_nodes)

    def sample_trace(self, seed_set: Set[int], seed_set_with_topics: Dict[int, int], out_trace_type: bool = True, simulation_rvs=None) -> Union[Trace, List[Set[int]]]:
        """Sample a single trace from the kICM model with predefined topics for seed nodes.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        seed_set_with_topics : Dict[int, int]
            Dictionary mapping each node in the seed set to its predefined topic.
        out_trace_type : bool, optional
            Whether to return a Trace object (default is True).
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Returns
        -------
        Union[Trace, List[Set[int]]]
            The sampled trace of influenced nodes or a Trace object.
        """
        self._pre_simulation_init(seed_set, seed_set_with_topics, simulation_rvs=simulation_rvs)
        return self._simulate_trace(out_trace_type=out_trace_type)

    def make_simulation(self, seed_set: Set[int], seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> Union[Trace, List[Set[int]]]:
        """Run a simulation and return the propagation trace.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        seed_set_with_topics : Dict[int, int]
            Dictionary mapping each node in the seed set to its predefined topic.
        simulation_rvs : np.ndarray, optional
            Predefined random variables for simulation (default is None).

        Returns
        -------
        np.ndarray
            The propagation trace as an array.
        """
        self._pre_simulation_init(seed_set, seed_set_with_topics, simulation_rvs=simulation_rvs)
        return self._simulate_trace()

    def sample_traces(self, n_traces: int = 100, seed_size_range: List[int] = None, seed_set_with_topics: Dict[int, int] = None, out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample multiple traces.

        Parameters
        ----------
        n_traces : int, optional
            Number of traces to sample (default is 100).
        seed_size_range : List[int], optional
            Range of seed sizes to sample from (default is None).
        seed_set_with_topics : Dict[int, int], optional
            Dictionary mapping each node in the seed set to its predefined topic.
        out_trace_type : bool, optional
            Whether to return a Traces object (default is True).

        Returns
        -------
        Union[Traces, List[List[Set[int]]]]
            The sampled traces or a Traces object.
        """
        seed_sets = self._sample_seeds(n_seeds=n_traces, seed_size_range=seed_size_range)
        return self.sample_traces_from_seeds(seed_sets, seed_set_with_topics, out_trace_type=out_trace_type)

    def sample_traces_from_seeds(self, seed_sets: List[Set[int]], seed_set_with_topics: Dict[int, int], out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample traces from a list of seed sets.

        Parameters
        ----------
        seed_sets : List[Set[int]]
            List of seed sets for sampling.
        seed_set_with_topics : Dict[int, int]
            Dictionary mapping each node in the seed set to its predefined topic.
        out_trace_type : bool, optional
            Whether to return a Traces object (default is True).

        Returns
        -------
        Union[Traces, List[List[Set[int]]]]
            The sampled traces or a Traces object.
        """
        simulation_rvs_over_runs = self._generate_simulation_rvs(n_runs=len(seed_sets))
        traces = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda seed_set, simulation_rvs:
                    self.sample_trace(seed_set=seed_set, seed_set_with_topics=seed_set_with_topics, out_trace_type=False, simulation_rvs=simulation_rvs))
            (seed_set, thresholds)
            for seed_set, thresholds in zip(seed_sets, simulation_rvs_over_runs)
        )
        return Traces(self.g, traces) if out_trace_type else traces
    def _init_simulation_rvs(self, simulation_rvs=None) -> None:
        """Initialize edge activations for the ICM.

        Parameters
        ----------
        simulation_rvs : np.ndarray, optional
            Predefined edge activations (default is None).

        Raises
        ------
        AssertionError
            If the length of simulation_rvs does not match the number of edges.
        """
        if simulation_rvs is None:
            simulation_rvs = self._generate_simulation_rvs()
        else:
            assert len(simulation_rvs) == self.g.count_edges(), \
                "Length of edge_activations must match the number of edges in the graph"
        self.edge_activations = simulation_rvs