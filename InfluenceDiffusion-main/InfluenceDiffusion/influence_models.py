import numpy as np
from copy import deepcopy
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from typing import Dict, Set, Union, List
from joblib import Parallel, delayed

from graph import Graph


class kInfluenceModel:
    """Base class for k-topic influence models."""

    def __init__(self, g: Graph, n_topics: int, check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        self.g = g
        self.vertices = list(self.g.get_vertices())
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._edge_2_index = {edge: idx for idx, edge in enumerate(g.edges)}
        self.n_topics = n_topics
        if check_init:
            self.check_param_init_correctness()

    def check_param_init_correctness(self) -> None:
        """Check the correctness of the model's parameters."""
        raise NotImplementedError

    def _init_simulation_rvs(self, simulation_rvs=None) -> None:
        """Initialize simulation random variables."""
        raise NotImplementedError

    def _generate_simulation_rvs(self, n_runs: int = 1) -> np.ndarray:
        """Generate random activations for edges (ICM logic) for each topic."""
        if self.random_state:
            np.random.seed(self.random_state)
        random_vals = np.random.rand(n_runs, self.g.count_edges(), self.n_topics)
        edge_activations = random_vals <= self.g.weights[:, np.newaxis]
        return edge_activations[0] if n_runs == 1 else edge_activations

    def _pre_simulation_init(self, seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> None:
        """Prepare for simulation by initializing parameters."""
        assert len(seed_set_with_topics) > 0, "Seed set should contain at least one vertex with a topic"
        self.seed_set_with_topics = deepcopy(seed_set_with_topics)
        self._cur_influence_nodes = deepcopy(seed_set_with_topics)
        self.all_influenced_nodes = deepcopy(seed_set_with_topics)
        self._propagation_trace: List[Dict[int, int]] = [self._cur_influence_nodes]
        self._init_simulation_rvs(simulation_rvs)

    def _make_edge_influence_attempt(self, v: int, v_adj: int, topic: int) -> bool:
        """Attempt to influence adjacent vertex."""
        if self.edge_activations[self._edge_2_index[(v, v_adj)],topic]:
            return True
        return False

    def _simulate_trace(self) -> List[Dict[int, int]]:
        """Simulate the influence spread."""
        while self._cur_influence_nodes:
            self._make_step()
        return self._propagation_trace

    def sample_trace(self, seed_set_with_topics,                  
                     simulation_rvs=None) -> List[Dict[int, int]]:
        """Sample a single trace from the model."""
        self._pre_simulation_init(seed_set_with_topics, simulation_rvs=simulation_rvs)
        return self._simulate_trace()

    def make_simulation(self, seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> List[Dict[int, int]]:
        """Run a simulation and return the propagation trace."""
        self._pre_simulation_init(seed_set_with_topics, simulation_rvs=simulation_rvs)
        return self._simulate_trace()

    '''
    def sample_traces(self, n_traces: int = 100,
                      seed_size_range: List[int] = None,
                      out_trace_type: bool = True) -> Union[Traces, List[List[Set[int]]]]:
        """Sample multiple traces."""
        seed_sets = self._sample_seeds(n_seeds=n_traces, seed_size_range=seed_size_range)
        return self.sample_traces_from_seeds(seed_sets, out_trace_type=out_trace_type)
    
    def sample_traces_from_seeds(self, seed_sets: List[Set[int]],
                                 out_trace_type: bool = False) -> Union[Traces, List[List[Set[int]]]]:
        """Sample traces from a list of seed sets."""
        simulation_rvs_over_runs = self._generate_simulation_rvs(n_runs=len(seed_sets))
        traces = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda seed_set, simulation_rvs:
                    self.sample_trace(seed_set_with_topics=seed_set, out_trace_type=False, simulation_rvs=simulation_rvs))
            (seed_set, thresholds)
            for seed_set, thresholds in zip(seed_sets, simulation_rvs_over_runs)
        )
        return traces
    '''
    def _make_step(self) -> None:

        """Make a simulation step by activating new nodes."""
        new_influence_nodes = {}
        assert isinstance(self._cur_influence_nodes, dict), "self._cur_influence_nodes should always be a dictionary"
        for v, topic in self._cur_influence_nodes.items():
            for v_adj in self.g.get_children(v).difference(self.all_influenced_nodes.keys()):
                influence_res = self._make_edge_influence_attempt(v, v_adj, topic)  # Now we pass the topic
                if influence_res:
                    new_influence_nodes[v_adj] = topic
        self._cur_influence_nodes=deepcopy(new_influence_nodes)
        self.all_influenced_nodes.update(new_influence_nodes)

        if new_influence_nodes:
            self._propagation_trace.append(new_influence_nodes)


class kICM(kInfluenceModel):
    """k-Topic Independent Cascade Model (kICM) for influence spread."""

    def __init__(self, g: Graph, n_topics: int, check_init: bool = True, random_state: int = None, n_jobs: int = None) -> None:
        super().__init__(g, n_topics, check_init, random_state, n_jobs)
    def check_param_init_correctness(self) -> None:
        """Check if the graph edge weights are valid for the ICM."""
        assert self.g.weights.shape == (self.g.count_edges(), self.n_topics)
        assert np.all((0 <= self.g.weights) & (self.g.weights <= 1))

    def _generate_simulation_rvs(self, n_runs: int = 1) -> np.ndarray:
        """Generate random activations for edges (kICM logic)."""
        if self.random_state:
            np.random.seed(self.random_state)
        edge_activations = np.random.rand(n_runs, self.g.count_edges(), self.n_topics) <= self.g.weights[:, np.newaxis]

        return edge_activations[0] if n_runs == 1 else edge_activations
    '''
    def _make_edge_influence_attempt(self, v: int, v_adj: int, topic: int) -> bool:
        """Attempt to influence an adjacent vertex based on edge activation."""
        if self.vertex_2_topic[v_adj] is None:  # If the adjacent vertex is not influenced
            edge_idx = self._edge_2_index.get((v, v_adj))
            if self.edge_activations[self._edge_2_index[(v, v_adj)],topic]:
                self.vertex_2_topic[v_adj] = topic  # Assign the topic of the influencing node to the adjacent node
                return True
        return False
    '''
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
        countTrue=np.sum(self.edge_activations, axis=0)
        countFalse=self.edge_activations.shape[0]-countTrue
    def _pre_simulation_init(self, seed_set_with_topics: Dict[int, int], simulation_rvs=None) -> None:
        """Prepare for simulation by initializing parameters for k-topic independent cascade."""
        super()._pre_simulation_init(seed_set_with_topics, simulation_rvs=simulation_rvs)
        self.edge_activations = self._generate_simulation_rvs(n_runs=1)
