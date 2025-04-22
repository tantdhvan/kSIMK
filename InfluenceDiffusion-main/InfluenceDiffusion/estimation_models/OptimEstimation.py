import numpy as np
from typing import List, Union, Dict, Tuple
from scipy.optimize import LinearConstraint, minimize
from scipy.stats._distn_infrastructure import rv_frozen

from ..Graph import Graph
from ..Trace import Traces, PseudoTraces
from .BaseWeightEstimator import BaseGLTWEightEstimator

__all__ = ["GLTGridSearchEstimator", "GLTWeightEstimator"]


class GLTWeightEstimator(BaseGLTWEightEstimator):
    """GLTW Weight Estimator for modeling influence in networks.

    Attributes
    ----------
    vertex_2_distrib : Dict[int, rv_frozen]
        Distribution mapping for vertices.
    """

    def _compute_failed_vertex_ll(self, weights: np.ndarray, vertex: int) -> float:
        """Compute the log-likelihood for failed vertices.

        Parameters
        ----------
        weights : np.ndarray
            The weights for the graph edges.
        vertex : int
            The vertex index.

        Returns
        -------
        float
            The log-likelihood for the failed vertex.
        """
        if vertex not in self._failed_vertices_masks:
            return 0.0
        cdf = self.vertex_2_distrib[vertex].cdf
        failed_vertices_indeg = self._failed_vertices_masks[vertex] @ weights
        failure_probs = 1.0 - cdf(failed_vertices_indeg)
        ll = np.log(np.clip(failure_probs, 1e-8, None)).sum()
        return ll

    def _compute_activated_vertex_ll(self, weights: np.ndarray, vertex: int) -> float:
        """Compute the log-likelihood for activated vertices.

        Parameters
        ----------
        weights : np.ndarray
            The weights for the graph edges.
        vertex : int
            The vertex index.

        Returns
        -------
        float
            The log-likelihood for the activated vertex.
        """
        if vertex not in self._vertex_2_active_parent_mask_tm1:
            return 0.0
        cdf = self.vertex_2_distrib[vertex].cdf
        activated_vertices_indeg_tm1 = self._vertex_2_active_parent_mask_tm1[vertex] @ weights
        activated_vertices_indeg_t = self._vertex_2_active_parent_mask_t[vertex] @ weights
        activation_probs = cdf(activated_vertices_indeg_t) - cdf(activated_vertices_indeg_tm1)
        ll = np.log(np.clip(activation_probs, 1e-8, None)).sum()
        return ll

    def _compute_vertex_nll(self, weights: np.ndarray, vertex: int) -> float:
        """Compute the negative log-likelihood for a vertex.

        Parameters
        ----------
        weights : np.ndarray
            The weights for the graph edges.
        vertex : int
            The vertex index.

        Returns
        -------
        float
            The negative log-likelihood for the vertex.
        """
        ll = self._compute_activated_vertex_ll(weights, vertex) + self._compute_failed_vertex_ll(weights, vertex)
        return -ll

    def _compute_total_nll(self, weights: np.ndarray) -> float:
        """Compute the total negative log-likelihood for informative vertices.

        Parameters
        ----------
        weights : np.ndarray
            The weights for the graph edges.

        Returns
        -------
        float
            The total negative log-likelihood.
        """
        return np.sum([
            self._compute_vertex_nll(weights[self.graph.get_parents_mask(vertex)], vertex)
            for vertex in self.informative_vertices
        ])

    def _compute_normalized_vertex_nll(self, weights: np.ndarray, vertex: int) -> float:
        """Compute the normalized negative log-likelihood for a vertex.

        Parameters
        ----------
        weights : np.ndarray
            The weights for the graph edges.
        vertex : int
            The vertex index.

        Returns
        -------
        float
            The normalized negative log-likelihood for the vertex.
        """
        return self._compute_vertex_nll(weights, vertex) / self._vertex_2_num_traces_was_informative[vertex]

    def _make_parent_weight_constraints(self, vertex: int, eps: float = 1e-8) -> LinearConstraint:
        """Create constraints for parent weights of a vertex.

        Parameters
        ----------
        vertex : int
            The vertex index.
        eps : float, optional
            Small value for numerical stability.

        Returns
        -------
        LinearConstraint
            The linear constraints for optimization.
        """
        indeg_max = self.vertex_2_distrib[vertex].support()[1]
        num_parents = self.graph.get_indegree(vertex, weighted=False)

        mat_constrain = np.vstack([np.eye(num_parents), np.ones(num_parents)])
        lb = eps * np.ones(num_parents + 1)
        ub = (indeg_max - eps) * np.ones(num_parents + 1)
        return LinearConstraint(A=mat_constrain, lb=lb, ub=ub)

    def _precompute_all_constraints(self) -> None:
        """Precompute weight constraints for all informative vertices."""
        self._weight_constraints = {
            vertex: self._make_parent_weight_constraints(vertex)
            for vertex in self.informative_vertices
        }

    def _optimize_vertex_parent_params(self, vertex: int, optimization_kwargs: Dict = None) -> Tuple[np.ndarray, rv_frozen]:
        """Optimize parameters for a specific vertex.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.
        optimization_kwargs : Dict, optional
            Additional optimization parameters.

        Returns
        -------
        Tuple[np.ndarray, rv_frozen]
            Optimized weights and the vertex distribution.
        """
        optimizer_output = minimize(
            lambda weights: self._compute_normalized_vertex_nll(weights, vertex=vertex),
            x0=self.weights_[self.graph.get_parents_mask(vertex)],
            method='SLSQP',
            constraints=self._weight_constraints[vertex],
            options=optimization_kwargs
        )
        return optimizer_output.x, self.vertex_2_distrib[vertex]

    def _set_informative_vertices_parent_params(self, informative_vertices_params: List[Tuple[np.ndarray, rv_frozen]]) -> None:
        """Set the optimized parameters for informative vertices.

        Parameters
        ----------
        informative_vertices_params : List[Tuple[np.ndarray, rv_frozen]]
            List of optimized parameters for each informative vertex.
        """
        for vertex, (parent_weights, vertex_distrib) in zip(self.informative_vertices, informative_vertices_params):
            self.weights_[self.graph.get_parents_mask(vertex)] = parent_weights
            self.vertex_2_distrib[vertex] = vertex_distrib

    def _pre_fit(self, traces: Union[Traces, PseudoTraces], 
                 init_weights: Union[List[float], np.array] = None, 
                 masks_path: str = None) -> None:
        """Prepare for fitting by preprocessing traces and initializing constraints.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        init_weights : Optional[List[float]]
            Initial weights to set.
        masks_path : Optional[str]
            Path to load masks from.
        """
        self._preprocess_traces(traces, masks_path=masks_path)
        self._precompute_all_constraints()
        self._init_weights(init_weights)


class GLTGridSearchEstimator(GLTWeightEstimator):
    """GLT Grid Search Estimator for hyperparameter tuning.

    Attributes
    ----------
    vertex_2_distrib_grid : Dict[int, List[rv_frozen]]
        Distribution grid for each vertex.
    """

    def __init__(self, graph: Graph,
                 distribs_grid: Union[Dict[int, List[rv_frozen]], List[rv_frozen]],
                 n_jobs: int = 1) -> None:
        """Initialize the GLT Grid Search Estimator.

        Parameters
        ----------
        graph : Graph
            The graph structure.
        distribs_grid : Union[Dict[int, List[rv_frozen]], List[rv_frozen]]
            The grid of distributions for each vertex.
        n_jobs : int, optional
            Number of jobs for parallel processing.
        """
        super().__init__(graph=graph, vertex_2_distrib=None, n_jobs=n_jobs)
        self._validate_grid(distribs_grid)

    def _validate_grid(self, distribs_grid: Union[Dict[int, List[rv_frozen]], List[rv_frozen]]) -> None:
        """Validate the distribution grid.

        Parameters
        ----------
        distribs_grid : Union[Dict[int, List[rv_frozen]], List[rv_frozen]]
            The grid of distributions to validate.

        Raises
        ------
        AssertionError
            If the distribution grid does not match the graph vertices.
        """
        if isinstance(distribs_grid, (List, Tuple)):
            self.vertex_2_distrib_grid = {vertex: distribs_grid for vertex in self.graph.get_vertices()}
        elif isinstance(distribs_grid, Dict):
            assert set(distribs_grid.keys()) == self.graph.get_vertices(), \
                "If `distribs_grid` is a dict, its keys should be the vertices of the graph."
            self.vertex_2_distrib_grid = distribs_grid
        else:
            raise NotImplementedError("`distribs_grid` should be either a list, tuple, or a dict.")

        assert all([
            all([isinstance(distrib, rv_frozen) for distrib in vertex_distribs])
            for vertex_distribs in self.vertex_2_distrib_grid.values()
        ]), "All elements of the grid should be `rv_frozen`."

    def _optimize_vertex_parent_params(self, vertex: int, optimization_kwargs: Dict = None) -> Tuple[np.ndarray, rv_frozen]:
        """Optimize parameters for a specific vertex using grid search.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.
        optimization_kwargs : Dict, optional
            Additional optimization parameters.

        Returns
        -------
        Tuple[np.ndarray, rv_frozen]
            Best weights and the best distribution for the vertex.
        """
        best_weights = None
        best_nll = np.inf
        best_distrib = None

        for distrib in self.vertex_2_distrib_grid[vertex]:
            self.vertex_2_distrib[vertex] = distrib
            optimizer_output = minimize(
                lambda weights: self._compute_normalized_vertex_nll(weights, vertex=vertex),
                x0=self.weights_[self.graph.get_parents_mask(vertex)],
                method='SLSQP',
                constraints=self._weight_constraints[vertex],
                options=optimization_kwargs
            )
            nll = optimizer_output.fun
            if nll < best_nll:
                best_weights = optimizer_output.x
                best_distrib = distrib
                best_nll = nll

        return best_weights, best_distrib
