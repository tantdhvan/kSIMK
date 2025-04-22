import networkx as nx
from InfluenceDiffusion.Graph import Graph
from InfluenceDiffusion.influence_models import LTM

# Bước 1: Đọc đồ thị từ file
def read_graph_from_file(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            u, v, w = line.split()
            G.add_edge(int(u), int(v), weight=float(w))
    return G

# Bước 2: Chuyển đổi đồ thị NetworkX sang đồ thị của InfluenceDiffusion
def convert_to_influencediffusion_graph(G):
    return Graph(edge_list=G.edges(data=True))

# Bước 3: Mô phỏng lan truyền ảnh hưởng từ seed set
def simulate_influence(g, seed_set, num_simulations=1000):
    ltm = LTM(g)
    traces = ltm.sample_traces(seed_set, num_simulations)
    activated_nodes = set(node for trace in traces for node in trace)
    return len(activated_nodes)

# Đọc đồ thị từ file
G = read_graph_from_file('graph.txt')

# Chuyển đổi đồ thị sang định dạng của InfluenceDiffusion
g = convert_to_influencediffusion_graph(G)

# Định nghĩa seed set
seed_set = [1, 2, 3]

# Mô phỏng lan truyền và tính toán độ ảnh hưởng
influence = simulate_influence(g, seed_set)
print(f"Độ ảnh hưởng của seed set {seed_set}: {influence}")
