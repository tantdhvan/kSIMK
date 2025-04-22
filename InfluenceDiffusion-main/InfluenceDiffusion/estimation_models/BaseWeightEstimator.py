from typing import List, Union, Dict
import os
import pickle
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import uniform
from scipy.stats._distn_infrastructure import rv_frozen

from ..Trace import Traces, PseudoTraces
from ..Graph import Graph
from ..utils import multiple_union

__all__ = ["BaseWeightEstimator", "BaseGLTWEightEstimator", "BaseICWEightEstimator"]


class BaseWeightEstimator:
    """Base class for weight estimation in influence models.

    Attributes
    ----------
    graph : Graph
        The graph representing the network.
    n_jobs : Optional[int]
        Number of jobs for parallel processing.
    informative_vertices : set
        Set of informative vertices in the graph.
    weights_ : np.ndarray
        Estimated weights for edges.
    _vertex_2_num_traces_was_informative : Dict[int, int]
        Mapping of vertices to the number of informative traces they participated in.
    _failed_vertices_masks : Dict[int, np.ndarray]
        Masks for failed vertices.
    _vertex_2_active_parent_mask_tm1 : Dict[int, np.ndarray]
        Masks for active parents at time t-1.
    _vertex_2_active_parent_mask_t : Dict[int, np.ndarray]
        Masks for active parents at time t.
    """

    def __init__(self, graph: Graph, n_jobs: int = None):
        self.graph = graph
        self.n_jobs = n_jobs
        self.informative_vertices = set()
        self.weights_ = np.array([])

    def _precompute_num_traces_vertices_participated(self) -> None:
        """Precompute the number of traces each vertex participated in."""
        self._vertex_2_num_traces_was_informative = {}
        for vertex in self.informative_vertices:
            num_failed_traces = 0 if vertex not in self._failed_vertices_masks \
                else self._failed_vertices_masks[vertex].shape[0]
            num_activated_traces = 0 if vertex not in self._vertex_2_active_parent_mask_tm1 \
                else self._vertex_2_active_parent_mask_tm1[vertex].shape[0]
            self._vertex_2_num_traces_was_informative[vertex] = num_activated_traces + num_failed_traces

    def _precompute_failed_vertices_parents_masks_from_traces(self, traces: Traces) -> None:
        """Compute masks for failed vertices based on traces.

        Parameters
        ----------
        traces : Traces
            The traces to analyze.
        """
        failed_vertices_masks = {vertex: [] for vertex in self.informative_vertices}
        for trace in traces:
            active_vertices = np.array(list(trace.get_all_activated_vertices()))
            failed_vertices = np.array(list(trace.get_all_failed_vertices()))
            for vertex in failed_vertices:
                parents = self.graph.get_parents(vertex, out_type=np.array)
                active_parents_mask = np.in1d(parents, active_vertices)
                failed_vertices_masks[vertex].append(active_parents_mask)
            
        self._failed_vertices_masks = {vertex: np.vstack(masks) for vertex, masks in failed_vertices_masks.items()
                                       if len(masks) > 0}

    def _precompute_active_vertices_parents_masks_from_traces(self, traces: Traces) -> None:
        """Compute active vertices' parents masks from traces.

        Parameters
        ----------
        traces : Traces
            The traces to analyze.
        """
        vertex_2_active_parent_mask_tm1 = {vertex: [] for vertex in self.informative_vertices}
        vertex_2_active_parent_mask_t = {vertex: [] for vertex in self.informative_vertices}

        for trace in traces:
            cum_trace_list = [set()] + list(trace.cum_union())
            for new_vertices_tp1, vertices_t, vertices_tm1 in zip(trace[1:], cum_trace_list[1:-1], cum_trace_list[:-2]):
                for vertex in new_vertices_tp1:
                    parents = self.graph.get_parents(vertex, out_type=np.array)
                    active_parents_mask_t = np.in1d(parents, np.array(list(vertices_t)))
                    active_parents_mask_tm1 = np.in1d(parents, np.array(list(vertices_tm1)))
                    vertex_2_active_parent_mask_tm1[vertex].append(active_parents_mask_tm1)
                    vertex_2_active_parent_mask_t[vertex].append(active_parents_mask_t)

        self._vertex_2_active_parent_mask_tm1 = {vertex: np.vstack(masks)
                                                 for vertex, masks in vertex_2_active_parent_mask_tm1.items()
                                                 if len(masks) > 0}
        self._vertex_2_active_parent_mask_t = {vertex: np.vstack(masks)
                                               for vertex, masks in vertex_2_active_parent_mask_t.items()
                                               if len(masks) > 0}

    def _load_masks(self, masks_path: str) -> None:
        """Load masks from specified file paths.

        Parameters
        ----------
        masks_path : str
            Path to the directory containing mask files.
        """
        with open(os.path.join(masks_path, "activated_masks_tm1.pkl"), "rb") as f:
            self._vertex_2_active_parent_mask_tm1 = pickle.load(f)
        with open(os.path.join(masks_path, "activated_masks_t.pkl"), "rb") as f:
            self._vertex_2_active_parent_mask_t = pickle.load(f)
        with open(os.path.join(masks_path, "failed_masks.pkl"), "rb") as f:
            self._failed_vertices_masks = pickle.load(f)

    def _precompute_all_informative_vertices(self, traces: Union[Traces, PseudoTraces]) -> None:
        """Precompute all informative vertices from traces.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        """
        if isinstance(traces, Traces):
            self.informative_vertices = multiple_union(
                [trace.get_all_failed_and_activated_vertices_no_seed() for trace in traces])
        elif isinstance(traces, PseudoTraces):
            self.informative_vertices = set(traces.keys())
        else:
            raise NotImplementedError("traces should be of either Traces or PseudoTraces types")

    def _precompute_vertices_parents_masks_from_pseudo_traces(self, traces: PseudoTraces) -> None:
        """Compute parents masks from pseudo traces.

        Parameters
        ----------
        traces : PseudoTraces
            The pseudo traces to analyze.
        """
        vertex_2_active_parent_mask_tm1 = {vertex: [] for vertex in self.informative_vertices}
        vertex_2_active_parent_mask_t = {vertex: [] for vertex in self.informative_vertices}
        failed_vertices_masks = {vertex: [] for vertex in self.informative_vertices}

        for vertex, vertex_pseudo_traces in traces.items():
            for vertices_tm1, new_vertices_t in vertex_pseudo_traces:
                parents = self.graph.get_parents(vertex, out_type=np.array)
                active_parents_mask_tm1 = np.isin(parents, np.array(list(vertices_tm1)))
                if new_vertices_t:
                    vertices_t = vertices_tm1 | new_vertices_t
                    active_parents_mask_t = np.isin(parents, np.array(list(vertices_t)))
                    vertex_2_active_parent_mask_tm1[vertex].append(active_parents_mask_tm1)
                    vertex_2_active_parent_mask_t[vertex].append(active_parents_mask_t)
                else:
                    failed_vertices_masks[vertex].append(active_parents_mask_tm1)

        self._failed_vertices_masks = {vertex: np.vstack(masks)
                                       for vertex, masks in failed_vertices_masks.items() if len(masks) > 0}
        self._vertex_2_active_parent_mask_tm1 = {vertex: np.vstack(masks)
                                                 for vertex, masks in vertex_2_active_parent_mask_tm1.items()
                                                 if len(masks) > 0}
        self._vertex_2_active_parent_mask_t = {vertex: np.vstack(masks)
                                               for vertex, masks in vertex_2_active_parent_mask_t.items()
                                               if len(masks) > 0}

    def _precompute_vertices_parents_masks(self, traces: Union[Traces, PseudoTraces]) -> None:
        """Precompute vertices' parents masks based on traces.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        """
        if isinstance(traces, Traces):
            self._precompute_failed_vertices_parents_masks_from_traces(traces)
            self._precompute_active_vertices_parents_masks_from_traces(traces)
        elif isinstance(traces, PseudoTraces):
            self._precompute_vertices_parents_masks_from_pseudo_traces(traces)
        else:
            raise NotImplementedError("traces can only be of type Traces or PseudoTraces")

    def _preprocess_traces(self, traces: Union[Traces, PseudoTraces], masks_path: str = None) -> None:
        """Preprocess traces by validating and computing masks.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to preprocess.
        masks_path : Optional[str]
            Path to load masks from, if applicable.
        """
        traces = self._validate_traces(traces)
        self._precompute_all_informative_vertices(traces)
        if masks_path is None:
            self._precompute_vertices_parents_masks(traces)
        else:
            self._load_masks(masks_path)
        self._precompute_num_traces_vertices_participated()

    def _validate_traces(self, traces) -> Union[Traces, PseudoTraces]:
        """Validate and convert traces to Traces or PseudoTraces.

        Parameters
        ----------
        traces : any
            The traces to validate.

        Returns
        -------
        Union[Traces, PseudoTraces]
            The validated traces.
        """
        try:
            return Traces(self.graph, traces)
        except:
            return PseudoTraces(traces)

    def _check_init_weight_correctness(self, init_weights: Union[List, np.ndarray]) -> None:
        """Check the correctness of initial weights.

        Parameters
        ----------
        init_weights : Union[List, np.ndarray]
            Initial weights to validate.
        
        Raises
        ------
        AssertionError
            If the length of the weights does not match the number of edges.
        """
        assert len(init_weights) == self.graph.count_edges()

    def _init_weights(self, init_weights: Union[List, np.ndarray]) -> None:
        """Initialize weights based on provided initial weights or generate random weights.

        Parameters
        ----------
        init_weights : Optional[Union[List, np.ndarray]]
            Initial weights to set. If None, random weights are generated.
        """
        if init_weights is not None:
            self._check_init_weight_correctness(init_weights)
            self.weights_ = np.array(init_weights)
        else:
            self.weights_ = self._generate_random_weights()

    def _generate_random_weights(self) -> np.ndarray:
        """Generate random weights for edges.

        Returns
        -------
        np.ndarray
            Randomly generated weights.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _optimize_vertex_parent_params(self, vertex: int, **optim_kwargs) -> np.ndarray:
        """Optimize parameters for a specific vertex.

        Parameters
        ----------
        vertex : int
            The vertex to optimize parameters for.
        optim_kwargs : dict
            Additional optimization parameters.

        Returns
        -------
        np.ndarray
            Optimized parameters for the vertex.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _optimize_informative_vertices_parent_params(self, verbose: bool = False, **optim_kwargs) -> List[np.ndarray]:
        """Optimize parameters for all informative vertices in parallel.

        Parameters
        ----------
        verbose : bool, optional
            If True, progress messages are printed.
        optim_kwargs : dict
            Additional optimization parameters.

        Returns
        -------
        List[np.ndarray]
            List of optimized parameters for each informative vertex.
        """
        informative_vertices_params = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
            delayed(self._optimize_vertex_parent_params)(vertex, **optim_kwargs)
            for vertex in self.informative_vertices)
        return informative_vertices_params

    def _set_informative_vertices_parent_params(self, informative_vertices_params: List[np.ndarray]) -> None:
        """Set the optimized parameters for informative vertices.

        Parameters
        ----------
        informative_vertices_params : List[np.ndarray]
            List of optimized parameters for each informative vertex.
        """
        for vertex, parent_weights in zip(self.informative_vertices, informative_vertices_params):
            self.weights_[self.graph.get_parents_mask(vertex)] = parent_weights

    def _pre_fit(self, traces: Union[Traces, PseudoTraces],
                 init_weights: Union[List, np.ndarray] = None,
                 masks_path: str = None) -> None:
        """Prepare for fitting by preprocessing traces and initializing weights.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        init_weights : Optional[Union[List, np.ndarray]]
            Initial weights to set.
        masks_path : Optional[str]
            Path to load masks from.
        """
        self._preprocess_traces(traces, masks_path=masks_path)
        self._init_weights(init_weights)

    def fit(self, traces: Union[Traces, PseudoTraces], 
            init_weights: Union[List, np.ndarray] = None,
            verbose: bool = False,
            masks_path: str = None,
            **optim_kwargs) -> np.ndarray:
        """Fit the model to the given traces.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        init_weights : Optional[Union[List, np.ndarray]]
            Initial weights to set.
        verbose : bool, optional
            If True, progress messages are printed.
        masks_path : Optional[str]
            Path to load masks from.
        optim_kwargs : dict
            Additional optimization parameters.

        Returns
        -------
        np.ndarray
            The final estimated weights.
        """
        self._pre_fit(traces, init_weights=init_weights, masks_path=masks_path)
        informative_vertices_params = self._optimize_informative_vertices_parent_params(verbose=verbose, **optim_kwargs)
        self._set_informative_vertices_parent_params(informative_vertices_params)
        return self.weights_


class BaseGLTWEightEstimator(BaseWeightEstimator):
    """Base class for GLTW weight estimation.

    Attributes
    ----------
    vertex_2_distrib : Dict[int, scipy.stats.rv_frozen]
        Distribution mapping for vertices.
    """

    def __init__(self, graph: Graph, n_jobs: int = None,
                 vertex_2_distrib: Dict[int, rv_frozen] = None):
        super().__init__(graph, n_jobs)
        if vertex_2_distrib is None:
            vertex_2_distrib = {v: uniform(0, 1) for v in graph.get_vertices()}
        self.vertex_2_distrib = vertex_2_distrib

    def _generate_random_weights(self) -> np.ndarray:
        """Generate random weights based on vertex distributions.

        Returns
        -------
        np.ndarray
            Randomly generated weights for edges.
        """
        weights = np.zeros(self.graph.count_edges())
        for vertex in self.informative_vertices:
            parent_mask = self.graph.get_parents_mask(vertex)
            num_parents = self.graph.get_indegree(vertex, weighted=False)
            support_ub = self.vertex_2_distrib[vertex].support()[1]
            if support_ub == np.inf:
                weights[parent_mask] = np.random.exponential(num_parents)
            else:
                weights[parent_mask] = np.random.rand(num_parents) * support_ub / num_parents
        return weights

    def _check_init_weight_correctness(self, init_weights: Union[List, np.ndarray], eps: float = 1e-6) -> None:
        """Check the correctness of initial weights for GLTM.

        Parameters
        ----------
        init_weights : Union[List, np.ndarray]
            Initial weights to validate.
        eps : float, optional
            Tolerance for weight validation.

        Raises
        ------
        AssertionError
            If initial weights do not meet the criteria.
        """
        super()._check_init_weight_correctness(init_weights)
        indegrees = np.array([init_weights[self.graph.get_parents_mask(vertex)].sum() 
                              for vertex in self.vertex_2_distrib])
        vertex_supports = np.vstack([distrib.support() for vertex, distrib in self.vertex_2_distrib.items()])
        assert np.all((indegrees >= vertex_supports[:, 0] - eps) & (indegrees <= vertex_supports[:, 1] + eps))


class BaseICWEightEstimator(BaseWeightEstimator):
    """Base class for Independent Cascade Model weight estimation."""

    def _generate_random_weights(self) -> np.ndarray:
        """Generate random weights for edges uniformly distributed between 0 and 1.

        Returns
        -------
        np.ndarray
            Randomly generated weights for edges.
        """
        return np.random.rand(self.graph.count_edges())
        
    def _check_init_weight_correctness(self, init_weights: Union[List, np.ndarray]) -> None:
        """Check the correctness of initial weights for the Independent Cascade Model.

        Parameters
        ----------
        init_weights : Union[List, np.ndarray]
            Initial weights to validate.

        Raises
        ------
        AssertionError
            If initial weights are not within the range [0, 1].
        """
        assert np.all((init_weights >= 0) & (init_weights <= 1))
