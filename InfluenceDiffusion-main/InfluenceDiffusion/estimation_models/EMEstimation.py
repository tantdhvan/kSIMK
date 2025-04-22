import numpy as np
from typing import Union

from .BaseWeightEstimator import BaseWeightEstimator, BaseGLTWEightEstimator, BaseICWEightEstimator
from ..Trace import Traces, PseudoTraces
from ..utils import invert_non_zeros
from ..Graph import Graph

__all__ = ["BaseWeightEstimatorEM", "ICWeightEstimatorEM", "LTWeightEstimatorEM"]


class BaseWeightEstimatorEM(BaseWeightEstimator):
    """Base class for EM weight estimators.

    This class implements the EM algorithm for optimizing parent weights of vertices in a graph.

    Methods
    -------
    _em_step(vertex, vertex_parent_weights)
        Performs a single EM step for the specified vertex.
    _optimize_vertex_parent_params(vertex, max_iter=50, tol=1e-3)
        Optimizes the parent weights for the specified vertex using the EM algorithm.
    """

    def _em_step(self, vertex: int, vertex_parent_weights: np.array) -> np.array:
        """Perform a single EM step for the specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to perform the EM step.
        vertex_parent_weights : np.array
            The current weights of the parent vertices.

        Returns
        -------
        np.array
            The updated weights for the parent vertices.
        """
        raise NotImplementedError()

    def _optimize_vertex_parent_params(self, vertex: int, max_iter: int = 50, tol: float = 1e-3) -> np.array:
        """Optimize the parent weights for the specified vertex using the EM algorithm.

        Parameters
        ----------
        vertex : int
            The vertex for which to optimize parent weights.
        max_iter : int, optional
            The maximum number of iterations (default is 50).
        tol : float, optional
            The tolerance for convergence (default is 1e-3).

        Returns
        -------
        np.array
            The optimized weights for the parent vertices.
        """
        parent_weights = self.weights_[self.graph.get_parents_mask(vertex)]

        for _ in range(max_iter):
            new_parent_weights = self._em_step(vertex=vertex, vertex_parent_weights=parent_weights)
            parent_weight_change = np.linalg.norm(new_parent_weights - parent_weights)
            parent_weights = new_parent_weights
            if parent_weight_change < tol:
                break
        return parent_weights


class ICWeightEstimatorEM(BaseICWEightEstimator, BaseWeightEstimatorEM):
    """Independent Cascade Weight Estimator using EM algorithm.

    This class extends the base EM weight estimator for the Independent Cascade model.

    Methods
    -------
    _precompute_num_traces_parents_activated()
        Precomputes the number of traces where parents were activated for each vertex.
    _em_step(vertex, vertex_parent_weights)
        Performs a single EM step for the specified vertex.
    _preprocess_traces(traces: Union[Traces, PseudoTraces], masks_path: str = None)
        Preprocesses the input traces and prepares necessary data for EM steps.
    _generate_random_weights()
        Generates random initial weights for the parent vertices.
    """

    def _precompute_num_traces_parents_activated(self) -> None:
        """Precompute the number of traces where parents were activated for each vertex.

        This method updates the internal state to keep track of how many traces
        activated the parent vertices of the informative vertices.
        """
        self._vertex_2_num_traces_parents_activated = {}
        for vertex in self.informative_vertices:
            n_traces_parents_activated = np.zeros(self.graph.get_indegree(vertex))
            if vertex in self._vertex_2_active_parent_mask_t:
                n_traces_parents_activated += self._vertex_2_active_parent_mask_t[vertex].sum(0)
            if vertex in self._failed_vertices_masks:
                n_traces_parents_activated += self._failed_vertices_masks[vertex].sum(0)
            self._vertex_2_num_traces_parents_activated[vertex] = n_traces_parents_activated

    def _em_step(self, vertex: int, vertex_parent_weights: np.array) -> np.array:
        """Perform a single EM step for the specified vertex.

        This method computes the updated weights based on the current parent weights.

        Parameters
        ----------
        vertex : int
            The vertex for which to perform the EM step.
        vertex_parent_weights : np.array
            The current weights of the parent vertices.

        Returns
        -------
        np.array
            The updated weights for the parent vertices.
        """
        if vertex not in self._vertex_2_active_parent_mask_t:
            return np.zeros_like(vertex_parent_weights)

        active_parents_mask_t = self._vertex_2_active_parent_mask_t[vertex]
        active_parents_mask_tm1 = self._vertex_2_active_parent_mask_tm1[vertex]
        new_active_parents_mask = active_parents_mask_t & (~active_parents_mask_tm1)

        no_activation_probs = new_active_parents_mask * (1. - vertex_parent_weights)
        probs_vertex_activated = 1. - np.prod(no_activation_probs, axis=1, where=no_activation_probs > 0, keepdims=True)
        inv_probs_vertex_activated = invert_non_zeros(probs_vertex_activated)
        upd_weights = vertex_parent_weights * (new_active_parents_mask * inv_probs_vertex_activated).sum(0)

        n_traces_parents_activated = self._vertex_2_num_traces_parents_activated[vertex]
        seen_parents_mask = n_traces_parents_activated != 0
        upd_weights[seen_parents_mask] /= n_traces_parents_activated[seen_parents_mask]
        return np.clip(upd_weights, 0, 1)

    def _preprocess_traces(self, traces: Union[Traces, PseudoTraces], masks_path: str = None) -> None:
        """Preprocesses the input traces and prepares necessary data for EM steps.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The input traces or pseudotraces to analyze.
        masks_path : str, optional
            The path to load masks from (default is None).
        """
        super()._preprocess_traces(traces=traces, masks_path=masks_path)
        self._precompute_num_traces_parents_activated()

    def _generate_random_weights(self) -> np.array:
        """Generates random initial weights for the parent vertices.

        Returns
        -------
        np.array
            An array of random weights for the parent vertices.
        """
        weights = np.zeros(self.graph.count_edges())
        informative_edge_mask = np.isin(self.graph.edge_array[:, 1], list(self.informative_vertices))
        weights[informative_edge_mask] = np.random.rand(informative_edge_mask.sum())
        return weights


class LTWeightEstimatorEM(BaseGLTWEightEstimator, BaseWeightEstimatorEM):
    """Linear Threshold Weight Estimator using EM algorithm.

    This class extends the base EM weight estimator for the Linear Threshold model.

    Methods
    -------
    _em_step(vertex, vertex_parent_weights)
        Performs a single EM step for the specified vertex.
    _generate_random_weights()
        Generates random initial weights for the parent vertices.
    """

    def __init__(self, graph: Graph, n_jobs: int = None):
        """Initialize the Linear Threshold Weight Estimator.

        Parameters
        ----------
        graph : Graph
            The graph representing the relationships between vertices.
        n_jobs : int, optional
            The number of jobs for parallel processing (default is None).
        """
        super().__init__(graph, n_jobs=n_jobs)

    def _em_step(self, vertex: int, vertex_parent_weights: np.array) -> np.array:
        """Perform a single EM step for the specified vertex.

        Parameters
        ----------
        vertex : int
            The vertex for which to perform the EM step.
        vertex_parent_weights : np.array
            The current weights of the parent vertices.

        Returns
        -------
        np.array
            The updated weights for the parent vertices.
        """
        if vertex in self._vertex_2_active_parent_mask_t:
            active_parents_mask_t = self._vertex_2_active_parent_mask_t[vertex]
            active_parents_mask_tm1 = self._vertex_2_active_parent_mask_tm1[vertex]
            new_active_parents_mask = active_parents_mask_t & (~active_parents_mask_tm1)
            probs_vertex_activated = (new_active_parents_mask * vertex_parent_weights).sum(1).reshape(-1, 1)
            inv_probs_vertex_activated = invert_non_zeros(probs_vertex_activated)
            H_uvs = vertex_parent_weights * (new_active_parents_mask * inv_probs_vertex_activated).sum(0)
        else:
            H_uvs = np.zeros_like(vertex_parent_weights)

        if vertex in self._failed_vertices_masks:
            active_parents_T = self._failed_vertices_masks[vertex]
            probs_vertex_failed = 1. - (vertex_parent_weights * active_parents_T).sum(1).reshape(-1, 1)
            inv_probs_vertex_failed = invert_non_zeros(probs_vertex_failed)
            H_empty_v = (1. - vertex_parent_weights.sum()) * inv_probs_vertex_failed.sum()
            H_uvs += vertex_parent_weights * ((~active_parents_T) * inv_probs_vertex_failed).sum(0)
        else:
            H_empty_v = 0.

        upd_weights = H_uvs / (H_empty_v + H_uvs.sum())
        return upd_weights

    def _generate_random_weights(self) -> np.array:
        """Generates random initial weights for the parent vertices.

        Returns
        -------
        np.array
            An array of random weights for the parent vertices.
        """
        weights = np.zeros(self.graph.count_edges())
        for vertex in self.informative_vertices:
            parent_mask = self.graph.get_parents_mask(vertex)
            num_parents = self.graph.get_indegree(vertex, weighted=False)
            weights[parent_mask] = np.random.rand(num_parents) / num_parents
        return weights
