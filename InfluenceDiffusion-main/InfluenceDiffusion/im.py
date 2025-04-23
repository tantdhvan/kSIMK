import os
from copy import deepcopy
from itertools import product
from multiprocessing import cpu_count, Pool
import networkx as nx
import pickle

os.chdir("/work/tan.trandinh/im_py/gltm_experiments-main")

from influence_models import kICM
from InfluenceDiffusion.Graph import Graph
from InfluenceDiffusion.weight_samplers import make_random_weights_with_indeg_constraint

def create_graph(n_nodes, p, directed=True):
    indeg_ub = 1
    g_nx = nx.erdos_renyi_graph(n_nodes, p, directed=directed)
    g = Graph(g_nx.edges)
    true_weights = make_random_weights_with_indeg_constraint(g, indeg_ub=indeg_ub, random_state=None)
    g.set_weights(true_weights)
    return g
def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
def get_or_create_graph(filename, n_nodes=500, p=0.01):
    if os.path.exists(filename):
        print("Tải đồ thị từ file...")
        return load_graph(filename)
    else:
        print("File không tồn tại, tạo mới đồ thị và lưu vào file...")
        g = create_graph(n_nodes, p)
        save_graph(g, filename)
        return g
def single_simulation(model, seed_set,seed_set_with_topics):
    trace_steps = model.sample_trace(seed_set,seed_set_with_topics, out_trace_type=False)
    total_influenced = len(set.union(*trace_steps))
    return total_influenced

def estimate_influence(model, seed_set,seed_set_with_topics, num_simulations=1000):
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(single_simulation, [(model, seed_set,seed_set_with_topics)] * num_simulations)

    return sum(results) / num_simulations

filename= 'graph.pkl'

g = get_or_create_graph(filename)
k=3
modelICM = kICM(g,n_topics=k)
seed_set = []
seed_set_with_topics: dict[int, int] = {}
current_f=0
for i in range(k):
    max_node=-1
    max_topic=-1
    max_f=0
    for node in g.nodes:
        for topic in range(k):
            tmp_seed=deepcopy(seed_set)
            tmp_seed_set_with_topics=deepcopy(seed_set_with_topics)
            tmp_seed.append(node)
            tmp_seed_set_with_topics[node]=topic
            tmp_f = estimate_influence(modelICM, tmp_seed,tmp_seed_set_with_topics)
            if tmp_f>max_f:
                max_f=tmp_f
                max_node=node
                max_topic=topic
    if max_node==-1 or max_topic==-1:
        break
    current_f=max_f
    seed_set.append(max_node)
    seed_set_with_topics[max_node]=max_topic
    print(max_node,' ',max_f)
print(seed_set)
print(current_f)