import os
from copy import deepcopy
from itertools import product
from multiprocessing import cpu_count, Pool
import networkx as nx
import pickle
import random
import numpy as np
import time
from influence_models import kICM
from graph import Graph
from weight_samplers import make_random_weights_with_indeg_constraint


def create_graph(n_nodes, p,n_topics=3, directed=True):
    indeg_ub = 1
    g_nx = nx.erdos_renyi_graph(n_nodes, p, directed=directed)

    g = Graph(g_nx.edges)
    topic_weights={}
    true_weights=[]
    for topic in range(n_topics):
        topic_weights[topic] = make_random_weights_with_indeg_constraint(g, indeg_ub=indeg_ub, random_state=None)
    true_weights = np.zeros((g.count_edges(), n_topics))
    for i, edge in enumerate(g.get_edges()):
        for topic in range(n_topics):
            true_weights[i][topic] = float(topic_weights[topic][i])

    # Cập nhật trọng số cho các cạnh trong đồ thị
    g.set_weights(true_weights)
    print("Đồ thị tạo ra:", true_weights)
    return g

def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
def get_or_create_graph(filename, n_nodes, p):
    if os.path.exists(filename):
        print("Tải đồ thị từ file...")
        return load_graph(filename)
    else:
        print("File không tồn tại, tạo mới đồ thị và lưu vào file...")
        g = create_graph(n_nodes, p)
        save_graph(g, filename)
        return g
def single_simulation(model,seed_set_with_topics):
    trace_steps = model.sample_trace(seed_set_with_topics)
    total_influenced = len(set.union(*trace_steps))
    return total_influenced
def estimate_influence(model,seed_set_with_topics:dict[int,int]):

    assert isinstance(seed_set_with_topics, dict), "estimate_influence: seed_set_with_topics should always be a dictionary after update."
    num_simulations=1000
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(single_simulation, [(model,seed_set_with_topics)] * num_simulations)

    return sum(results) / num_simulations
def greedy_influence(model,b,n_topics=3):
    start_time=time.time()
    seed_set = []
    seed_set_weights = [0.0,0.0,0.0]
    seed_set_with_topics: dict[int, int] = {}
    current_f=0
    count_f=0
    while True:
        max_node=-1
        max_topic=-1
        max_f=0
        max_delta=-1
        for node in g.nodes:
            if node in seed_set:
                continue
            for topic in range(n_topics):
                if(seed_set_weights[topic] + model.g.nodes[node]['weight'] > b):
                    continue
                tmp_seed_set_with_topics=deepcopy(seed_set_with_topics)
                tmp_seed_set_with_topics[node]=topic

                tmp_f = estimate_influence(model,tmp_seed_set_with_topics)
                count_f=count_f+1
                tmp_delta = (tmp_f - current_f)/model.g.nodes[node]['weight']
                if tmp_delta>max_delta:
                    max_f=tmp_f
                    max_node=node
                    max_topic=topic
                    max_delta=tmp_delta
        if max_node==-1 or max_topic==-1:
            break
        current_f=max_f
        seed_set.append(max_node)
        seed_set_with_topics[max_node]=max_topic
        seed_set_weights[max_topic] += model.g.nodes[node]['weight']
    end_time=time.time()
    return current_f,count_f,round(end_time-start_time,1)
def get_Emax(model,node):
    tmp_seed_set_with_topics={node:0}
    tmp_f = estimate_influence(model,tmp_seed_set_with_topics)
    return tmp_f
def get_threshold(M, epsilon, b, weight, n_topics):
    threshold = {}
    j = -1
    while True:
        j = j + 1
        tmp_threshold = (1 + epsilon) ** j
        if tmp_threshold > M * b * n_topics/weight:
            break  # kiểm tra kết thúc trước
        if tmp_threshold >= M:
            threshold[j] = tmp_threshold  # lưu nếu thỏa điều kiện
    return threshold    
def streaming(model,node_weights,b,epsilon,alpha,n_topics=3):
    start_time=time.time()
    M=0.0
    seed_sets={}
    seed_sets_with_topics={}
    seed_sets_weights={}
    seed_sets_current_f={}
    seed_topics={}
    hash_f={}
    count_f=0
    f_emax=0
    node_max=-1
    topic_emax=-1
    for node in g.nodes:
        tmp_f=get_Emax(model,node)
        count_f=count_f+1
        for topic in range(n_topics):           
            tmp_seed={(node,topic)}
            hash_key=hash(frozenset(tmp_seed))
            hash_f[hash_key]=tmp_f
            if(tmp_f/node_weights[node][topic]>f_emax):
                f_emax=tmp_f/node_weights[node][topic]
                node_max=node
                topic_emax=topic
                M=tmp_f
    threshold=get_threshold(M,epsilon,b,node_weights[node_max][topic_emax],n_topics)
    for j in threshold.keys():
        seed_sets[j]=[]
        seed_sets_with_topics[j]={}
        seed_sets_current_f[j]=0.0
        seed_sets_weights[j]={}
        seed_topics[j]=[]
        for topic in range(n_topics):
            seed_sets_weights[j][topic]=0.0
    
    for node in g.nodes:
        for j, threshold_j in sorted(threshold.items()):              
            for topic in range(n_topics):
                if(seed_sets_weights[j][topic] + node_weights[node][topic] > b):
                    continue
                tmp_seed=deepcopy(seed_sets[j])
                tmp_seed_set_with_topics=deepcopy(seed_sets_with_topics[j])
                tmp_seed_topics=deepcopy(seed_topics[j])

                tmp_seed.append(node)
                tmp_seed_set_with_topics[node]=topic
                tmp_seed_topics.append((node,topic))
                tmp_hash=hash(frozenset(tmp_seed_topics))
                if tmp_hash in hash_f:
                    tmp_f=hash_f[tmp_hash]
                else:
                    tmp_f = estimate_influence(model, tmp_seed,tmp_seed_set_with_topics)
                    count_f=count_f+1
                    hash_f[tmp_hash]=tmp_f

                tmp_delta = (tmp_f - seed_sets_current_f[j])/node_weights[node][topic]
                if tmp_delta > threshold_j*alpha/b:
                    seed_sets[j]=tmp_seed
                    seed_sets_with_topics[j]=tmp_seed_set_with_topics
                    seed_sets_weights[j][topic]=seed_sets_weights[j][topic]+node_weights[node][topic]
                    seed_sets_current_f[j]=tmp_f
                    seed_topics[j]=tmp_seed_topics
    
    for node in g.nodes:
        for j,threshold_j in sorted(threshold.items()):
            if node in seed_sets[j]:
                continue
            for topic in range(n_topics):
                if(seed_sets_weights[j][topic] + node_weights[node][topic] > b):
                    continue
                tmp_seed=deepcopy(seed_sets[j])
                tmp_seed_set_with_topics=deepcopy(seed_sets_with_topics[j])
                tmp_seed_topics=deepcopy(seed_topics[j])

                tmp_seed.append(node)
                tmp_seed_set_with_topics[node]=topic
                tmp_seed_topics.append((node,topic))


                tmp_hash=frozenset(tmp_seed_topics)
                if tmp_hash in hash_f:
                    tmp_f=hash_f[tmp_hash]
                else:
                    tmp_f = estimate_influence(model, tmp_seed,tmp_seed_set_with_topics)
                    count_f=count_f+1
                    hash_f[tmp_hash]=tmp_f

                if tmp_f>seed_sets_current_f[j]:
                    seed_sets[j]=tmp_seed
                    seed_sets_with_topics[j]=tmp_seed_set_with_topics
                    seed_sets_weights[j][topic]=seed_sets_weights[j][topic]+node_weights[node][topic]
                    seed_sets_current_f[j]=tmp_f
                    seed_topics[j]=tmp_seed_topics
    end_time=time.time()
    return max(max(seed_sets_current_f.values()),M),count_f,round(end_time-start_time,1)
n_nodes=500
p=0.05
filename= 'graph_'+str(n_nodes)+'_'+str(p)+'.pkl'
g = get_or_create_graph(filename,n_nodes=n_nodes,p=p)
n_topics=3
epsilon=0.2
B=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
alpha=0.2
modelICM = kICM(g,n_topics=n_topics)
for b in B:
    #f_str,count_f_str,time_str=streaming(modelICM,node_weights,b,epsilon,alpha,n_topics=n_topics)
    #print('Streaming,',b,',',f_str,',',count_f_str,',',time_str)
    f_greedy,count_f_greedy,time_greedy=greedy_influence(modelICM,b,n_topics=n_topics)
    print('Greedy,',b,',',f_greedy,',',count_f_greedy,',',time_greedy)

