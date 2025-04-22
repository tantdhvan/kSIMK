import numpy as np
from copy import deepcopy
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Dict, Set, Union, List
from joblib import Parallel, delayed

from .Graph import Graph
from .Trace import Trace, Traces


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

    def _sample_seeds(self, n_seeds: int, seed_size_range: List[int] = None) -> List[Set[int]]:
        """Sample seed sets from the graph.

        Parameters
        ----------
        n_seeds : int
            Number of seed sets to sample.
        seed_size_range : List[int], optional
            Range of seed sizes to sample from (default is None).

        Returns
        -------
        List[Set[int]]
            List of sampled seed sets.

        Raises
        ------
        AssertionError
            If values in seed_size_range are out of bounds.
        """
        if seed_size_range is None:
            seed_size_range = range(1, self.g.count_vertices() + 1)
        else:
            assert max(seed_size_range) <= self.g.count_vertices() and min(seed_size_range) > 0, \
                "Values of `seed_size_range should be between 1 and the number of vertices in the graph"

        if self.random_state:
            np.random.seed(self.random_state)
        seed_sizes = np.random.choice(seed_size_range, size=n_seeds, replace=True)
        seed_sets = [set(np.random.choice(self.vertices, size=size, replace=False)) for size in seed_sizes]
        return seed_sets

    def estimate_spread(self, seed_set: Set[int], n_runs: int = 100, with_std: bool = False) -> Union[float, tuple]:
        """Estimate the spread of influence from a seed set.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        n_runs : int, optional
            Number of simulation runs (default is 100).
        with_std : bool, optional
            Whether to return standard deviation along with the mean (default is False).

        Returns
        -------
        Union[float, tuple]
            Mean spread, or a tuple of (mean spread, standard deviation) if with_std is True.
        """
        traces = self.sample_traces_from_seeds([seed_set] * n_runs, out_trace_type=True)
        spread_over_runs = [len(trace.get_all_activated_vertices()) for trace in traces]
        mean_spread, std_spread = np.mean(spread_over_runs), np.std(spread_over_runs)
        return (mean_spread, std_spread) if with_std else mean_spread

    def find_optimal_seed_greedily(self, seed_size: int, n_runs_per_node: int = 100) -> List[int]:
        """Find the optimal seed set using a greedy approach.

        Parameters
        ----------
        seed_size : int
            Desired size of the seed set.
        n_runs_per_node : int, optional
            Number of runs to estimate the spread for each candidate node (default is 100).

        Returns
        -------
        List[int]
            List of vertices in the optimal seed set.
        """
        seed_set = []
        unseen_seeds = list(range(len(self.vertices)))

        for _ in range(seed_size):
            spread_over_seeds = [
                self.estimate_spread(seed_set=set(seed_set + [vertex]), n_runs=n_runs_per_node)
                for vertex in unseen_seeds
            ]
            best_node_idx = np.argmax(spread_over_seeds)
            best_node = unseen_seeds[best_node_idx]
            seed_set.append(best_node)
            unseen_seeds.pop(best_node_idx)
        return seed_set


class LTM(InfluenceModel):
    """Linear Threshold Model (LTM) for influence spread.

    Parameters
    ----------
    g : Graph
        The graph on which the LTM operates.
    threshold_generator : rv_frozen, optional
        Distribution for generating thresholds (default is uniform distribution).
    check_init : bool, optional
        Whether to check the parameters upon initialization (default is True).
    random_state : int, optional
        Seed for random number generation (default is None).
    n_jobs : int, optional
        Number of parallel jobs for simulations (default is None).

    Attributes
    ----------
    vertex_2_threshold : Dict[int, float]
        Mapping of vertices to their corresponding thresholds.
    vertex_2_influence_counter : Dict[int, float]
        Mapping of vertices to their current influence counter.
    """

    def __init__(self, g: Graph, threshold_generator=stats.uniform(0, 1),
                 check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        self.threshold_generator = threshold_generator
        super().__init__(g, check_init=check_init, random_state=random_state, n_jobs=n_jobs)

    def _generate_simulation_rvs(self, n_runs: int = 1) -> np.ndarray:
        """Generate random thresholds for simulation.

        Parameters
        ----------
        n_runs : int, optional
            Number of simulation runs (default is 1).

        Returns
        -------
        np.ndarray
            Generated thresholds for each vertex.
        """
        thresholds = self.threshold_generator.rvs(size=(n_runs, len(self.vertices)),
                                                  random_state=self.random_state)
        return thresholds[0] if n_runs == 1 else thresholds

    def _init_simulation_rvs(self, simulation_rvs=None) -> None:
        """Initialize thresholds for vertices.

        Parameters
        ----------
        simulation_rvs : np.ndarray, optional
            Predefined thresholds for vertices (default is None).

        Raises
        ------
        AssertionError
            If the length of simulation_rvs does not match the number of vertices.
        """
        if simulation_rvs is None:
            simulation_rvs = self._generate_simulation_rvs()
        else:
            assert len(simulation_rvs) == len(self.vertices), \
                "Length of simulation_rvs should match the number of nodes in the graph for LTM"
        self.vertex_2_threshold = dict(zip(self.vertices, simulation_rvs))

    def _pre_simulation_init(self, seed_set: Set[int], simulation_rvs=None) -> None:
        """Prepare for simulation by initializing parameters specific to LTM.

        Parameters
        ----------
        seed_set : Set[int]
            Set of initial active nodes.
        simulation_rvs : np.ndarray, optional
            Predefined thresholds for vertices (default is None).
        """
        super()._pre_simulation_init(seed_set, simulation_rvs=simulation_rvs)
        self.vertex_2_influence_counter = {vertex: self.vertex_2_threshold[vertex] if vertex in seed_set else 0.0
                                           for vertex in self.vertices}

    def _make_edge_influence_attempt(self, v: int, v_adj: int) -> bool:
        """Attempt to influence an adjacent vertex based on its threshold and received influence.

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
        """
        self.vertex_2_influence_counter[v_adj] += self.g.get_edge_data(v, v_adj)["weight"]
        return self.vertex_2_threshold[v_adj] <= self.vertex_2_influence_counter[v_adj]

    def check_param_init_correctness(self, eps: float = 1e-6) -> None:
        """Validate parameters of the LTM model.

        Parameters
        ----------
        eps : float, optional
            Tolerance for validating indegree constraints (default is 1e-6).

        Raises
        ------
        AssertionError
            If the parameters are not valid.
        """
        assert isinstance(self.threshold_generator, rv_frozen), "Only scipy distributions are allowed"
        dist_min, dist_max = self.threshold_generator.support()
        indegrees = np.array([self.g.get_indegree(vertex, weighted=True) for vertex in self.g.get_vertices()])
        assert np.all((indegrees >= dist_min - eps) & (indegrees <= dist_max + eps))


class GLTM(LTM):
    """General Linear Threshold Model (GLTM).

    Parameters
    ----------
    g : Graph
        The graph on which the GLTM operates.
    threshold_distribs : Dict[int, rv_frozen]
        Mapping of vertices to their corresponding threshold distributions.
    check_init : bool, optional
        Whether to check the parameters upon initialization (default is True).
    random_state : int, optional
        Seed for random number generation (default is None).
    n_jobs : int, optional
        Number of parallel jobs for simulations (default is None).

    Attributes
    ----------
    threshold_distribs : Dict[int, rv_frozen]
        Mapping of vertices to their corresponding threshold distributions.
    """

    def __init__(self, g: Graph, threshold_distribs: Dict[int, rv_frozen],
                 check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        self.threshold_distribs = threshold_distribs
        super().__init__(g, check_init=check_init, random_state=random_state, n_jobs=n_jobs)

    def _generate_simulation_rvs(self, n_runs: int = 1) -> np.ndarray:
        """Generate random thresholds for each vertex using their specific distributions.

        Parameters
        ----------
        n_runs : int, optional
            Number of simulation runs (default is 1).

        Returns
        -------
        np.ndarray
            Generated thresholds for each vertex.
        """
        if self.random_state:
            np.random.seed(self.random_state)
        thresholds = np.random.rand(n_runs, len(self.vertices))
        for idx, vertex in enumerate(self.vertices):
            inv_cdf = self.threshold_distribs[vertex].ppf
            thresholds[:, idx] = inv_cdf(thresholds[:, idx])
        return thresholds[0] if n_runs == 1 else thresholds

    def check_param_init_correctness(self, eps: float = 1e-6) -> None:
        """Validate parameters of the GLT model.

        Parameters
        ----------
        eps : float, optional
            Tolerance for validating indegree constraints (default is 1e-6).

        Raises
        ------
        AssertionError
            If the parameters are not valid.
        """
        assert isinstance(self.threshold_distribs, dict), \
            "Threshold distributions should be a dict with vertices as keys and rv_frozen objects as values"
        indegrees = self.g.get_indegrees_dict(weighted=True)
        for vertex, threshold_dist in self.threshold_distribs.items():
            assert isinstance(threshold_dist, rv_frozen), "Only scipy distributions are allowed"
            dist_min, dist_max = threshold_dist.support()
            assert (indegrees[vertex] >= dist_min - eps) and (indegrees[vertex] <= dist_max + eps), \
                f"Indegree of vertex {vertex} is out of the distribution support range."

        
class ICM(InfluenceModel):
    """Independent Cascade Model (ICM) for influence spread.

    Parameters
    ----------
    g : Graph
        The graph on which the ICM operates.
    check_init : bool, optional
        Whether to check the parameters upon initialization (default is True).
    random_state : int, optional
        Seed for random number generation (default is None).
    n_jobs : int, optional
        Number of parallel jobs for simulations (default is None).
    """

    def check_param_init_correctness(self) -> None:
        """Check if the graph edge weights are valid for the ICM.

        Raises
        ------
        AssertionError
            If any edge weight is not in the range [0, 1].
        """
        assert np.all((0 <= self.g.weights) & (self.g.weights <= 1))

    def _make_edge_influence_attempt(self, v: int, v_adj: int) -> bool:
        """Attempt to influence an adjacent vertex based on edge activation.

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
        """
        return self.edge_activations[self._edge_2_index[(v, v_adj)]]

    def _generate_simulation_rvs(self, n_runs: int = 1) -> List:
        """Generate random activations for edges.

        Parameters
        ----------
        n_runs : int, optional
            Number of simulation runs (default is 1).

        Returns
        -------
        List[np.ndarray]
            Random activation results for each edge.
        """
        if self.random_state:
            np.random.seed(self.random_state)
        edge_activations = np.random.rand(n_runs, self.g.count_edges()) <= self.g.weights
        return edge_activations[0] if n_runs == 1 else edge_activations

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
