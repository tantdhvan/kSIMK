import numpy as np

from .Graph import Graph
from .utils import random_vector_inside_simplex, random_vector_on_simplex

__all__ = ["make_weighted_cascade_weights", "make_random_weights_with_indeg_constraint",
           "make_random_weights_with_fixed_indeg"]


def make_weighted_cascade_weights(g: Graph):
    vertex_2_indegree = g.get_vertex_2_indegree_dict(weighted=False)
    weights = 1. / np.array([vertex_2_indegree[sink] for sink in g.edge_array[:, 1]])
    return weights


def make_random_weights_with_indeg_constraint(g: Graph, indeg_ub=1, random_state=None):
    if random_state:
        np.random.seed(random_state)
    weights = np.zeros(g.count_edges())
    for v in g.get_sinks():
        parent_mask = g.get_parents_mask(v)
        weights[parent_mask] = random_vector_inside_simplex(parent_mask.sum(), ub=indeg_ub)
    return weights


def make_random_weights_with_fixed_indeg(g: Graph, indeg_ub=1, random_state=None):
    if random_state:
        np.random.seed(random_state)
    weights = np.zeros(g.count_edges())
    for v in g.get_sinks():
        parent_mask = g.get_parents_mask(v)
        weights[parent_mask] = random_vector_on_simplex(parent_mask.sum(), ub=indeg_ub)
    return weights
