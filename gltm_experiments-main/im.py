import numpy as np
import pandas as pd
import os
from scipy.stats import uniform, beta
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from copy import deepcopy
from itertools import product
from sklearn.metrics import mean_absolute_error
from multiprocessing import cpu_count
from collections import Counter
import networkx as nx

os.chdir("C:/Users/orlab/Documents/GitHub/kSIMK/gltm_experiments-main")

from InfluenceDiffusion.Graph import Graph
from InfluenceDiffusion.influence_models import GLTM, LTM, ICM

from InfluenceDiffusion.estimation_models.OptimEstimation import GLTGridSearchEstimator
from InfluenceDiffusion.estimation_models.EMEstimation import ICWeightEstimatorEM, LTWeightEstimatorEM
from InfluenceDiffusion.weight_samplers import make_random_weights_with_indeg_constraint,\
                                               make_random_weights_with_fixed_indeg, \
                                               make_weighted_cascade_weights

from benchmark_estimators import propagated_trace_number_weight_estimator, \
                                 propagated_trace_proportion_weight_estimator


from utils.trace_utils import make_report_traces, trace_train_test_split
from utils.utils import make_set_intersection_table, plot_distribution
from utils.model_evaluation_utils import make_heatmap

def influence_spread(model, seed_set):
    trace_steps = model.sample_trace(seed_set, out_trace_type=False)
    total_influenced = len(set.union(*trace_steps))  # Loại bỏ trùng lặp
    return total_influenced
def estimate_influence(model, seed_set, num_simulations=100):
    from concurrent.futures import ThreadPoolExecutor

    def single_simulation(_):
        return influence_spread(model, seed_set)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(single_simulation, range(num_simulations)))

    return sum(results) / num_simulations

n_nodes = 2000
p = 0.01
random_state = 1
max_seed_size = 5
indeg_ub = 1
fixed_indeg = False
n_train_traces = 2000

np.random.seed(random_state)

true_distrib_dict = {v: beta(1, int(np.random.randint(1, 6))) for v in range(n_nodes)}

g_nx = nx.erdos_renyi_graph(n_nodes, p=p, directed=True)
g = Graph(g_nx.edges)
if fixed_indeg:
    true_weights = make_random_weights_with_fixed_indeg(g, indeg_ub=indeg_ub, random_state=None)
else:
    true_weights = make_random_weights_with_indeg_constraint(g, indeg_ub=indeg_ub, random_state=None)
g.set_weights(true_weights)

modelGLTM = GLTM(g, true_distrib_dict, random_state=None)
modelLTM =LTM(g)
modelICM=ICM(g)

seed_set=[0]
kq = estimate_influence(modelICM, seed_set, num_simulations=1000)
print("Tổng số nút bị ảnh hưởng theo GLTM:", kq)