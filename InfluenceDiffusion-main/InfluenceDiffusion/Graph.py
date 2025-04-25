import numpy as np
from typing import List, Tuple, Union, Dict
import networkx as nx

__all__ = ["Graph"]

class Graph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, weights: Union[List, np.array] = None, n_topics: int = 1, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.edge_array = np.array(self.edges)
        
        # Nếu không có trọng số được cung cấp, tạo một mảng trọng số với kích thước n_topic
        if weights is None:
            self.weights = np.ones((len(self.edges), n_topics))  # Trọng số cho mỗi cạnh là một mảng với n_topic
        else:
            assert len(weights) == self.count_edges(), "Số lượng trọng số không khớp với số lượng cạnh"
            self.weights = np.array(weights)
        
        # Lưu trọng số theo kiểu mảng cho mỗi cạnh
        nx.set_edge_attributes(self, {edge: weight.tolist() for edge, weight in zip(self.edges, self.weights)}, "weight")

    def add_edge(self, u_of_edge, v_of_edge, weights: Union[List, np.array] = None, **attr):
        super().add_edge(u_of_edge, v_of_edge, **attr)
        self.edge_array = np.array(self.edges)
        
        if weights is None:
            # Nếu không cung cấp trọng số, tạo một mảng với tất cả trọng số bằng 1 cho mỗi topic
            self.weights = np.vstack([self.weights, np.ones(self.weights.shape[1])])
        else:
            # Nếu cung cấp trọng số, thêm vào mảng trọng số
            self.weights = np.vstack([self.weights, np.array(weights)])
        
        # Cập nhật trọng số cho các cạnh
        nx.set_edge_attributes(self, {edge: weight.tolist() for edge, weight in zip(self.edges, self.weights)}, "weight")

    def add_edges_from(self, ebunch_to_add, weights: Union[List, np.array] = None, **attr):
        super().add_edges_from(ebunch_to_add, **attr)
        self.edge_array = np.array(self.edges)
        
        if weights is None:
            # Nếu không có trọng số, tạo mảng trọng số với mỗi cạnh có trọng số là mảng tất cả 1s
            self.weights = np.vstack([self.weights, np.ones(self.weights.shape[1]) for _ in ebunch_to_add])
        else:
            # Nếu có trọng số, thêm chúng vào mảng trọng số
            self.weights = np.vstack([self.weights, np.array(weights)])
        
        nx.set_edge_attributes(self, {edge: weight.tolist() for edge, weight in zip(self.edges, self.weights)}, "weight")

    def add_weighted_edges_from(self, ebunch_to_add, weight='weight', weights: Union[List, np.array] = None, **attr):
        super().add_weighted_edges_from(ebunch_to_add, weight=weight, **attr)
        self.edge_array = np.array(self.edges)
        
        if weights is None:
            self.weights = np.vstack([self.weights, np.ones(self.weights.shape[1]) for _ in ebunch_to_add])
        else:
            self.weights = np.vstack([self.weights, np.array(weights)])
        
        nx.set_edge_attributes(self, {edge: weight.tolist() for edge, weight in zip(self.edges, self.weights)}, "weight")

    def set_weights(self, weights: Union[List, np.array]) -> "Graph":
        """
        Cập nhật trọng số cho các cạnh trong đồ thị.

        Trọng số cho mỗi cạnh bây giờ là một mảng với n_topic trọng số.
        """
        assert len(weights) == self.count_edges(), "Số lượng trọng số không khớp với số lượng cạnh"
        self.weights = np.array(weights)
        nx.set_edge_attributes(self, {edge: weight.tolist() for edge, weight in zip(self.edges, self.weights)}, "weight")
        return self

    def get_edge_weight_by_topic(self, u_of_edge, v_of_edge, topic_index: int) -> float:
        """
        Lấy trọng số của một cạnh theo chỉ số topic.
        """
        edge_weight = self[u_of_edge][v_of_edge].get('weight', None)
        if edge_weight is not None and 0 <= topic_index < len(edge_weight):
            return edge_weight[topic_index]
        return None  # Nếu không có trọng số cho topic đó

    def get_edges(self, as_array: bool = False) -> Union[np.array, List[Tuple[int, int]]]:
        if as_array:
            return self.edge_array
        return list(self.edges)

    def get_vertices(self) -> set:
        return set(self.nodes)
    
    def get_sources(self) -> set: 
        return set(self.edge_array[:, 0])
    
    def get_sinks(self) -> set: 
        return set(self.edge_array[:, 1])

    def count_edges(self) -> int:
        return len(self.edges)

    def count_vertices(self) -> int:
        return len(self.nodes)

    def get_children_mask(self, vertex: int) -> np.array:
        return self.edge_array[:, 0] == vertex

    def get_children(self, vertex: int, out_type: Union[set, list, np.array] = set) -> Union[set, list, np.array]:
        return out_type(self.edge_array[:, 1][self.get_children_mask(vertex)])

    def get_parents_mask(self, vertex: int) -> np.array:
        return self.edge_array[:, 1] == vertex

    def get_parents(self, vertex: int, out_type: Union[set, list, np.array] = set) -> Union[set, list, np.array]:
        return out_type(self.edge_array[:, 0][self.get_parents_mask(vertex)])

    def get_outdegree(self, vertex: int, weighted: bool = False, topic: int = None) -> Union[float, int]:
        children_mask = self.edge_array[:, 0] == vertex
        children_weights = self.weights[children_mask]
        
        if weighted:
            if topic is None:
                raise ValueError("Topic must be specified when weighted=True.")
            # Chọn trọng số của topic chỉ định và tính tổng trọng số cho topic đó
            return np.sum(children_weights[:, topic])
        else:
            # Nếu không có trọng số, chỉ đếm số lượng cạnh đi ra
            return len(children_weights)

    def get_all_outdegrees(self, weighted: bool = False, topic: int = None) -> np.array:
        outdegrees = []
        for v in self.get_vertices():
            outdegrees.append(self.get_outdegree(v, weighted=weighted, topic=topic))
        return np.array(outdegrees)

    def get_avg_outdegree(self, weighted: bool = False, topic: int = None) -> Union[float, int]:
        return np.mean(self.get_all_outdegrees(weighted=weighted, topic=topic))

    def get_max_outdegree(self, weighted: bool = False, topic: int = None) -> Union[float, int]:
        return np.max(self.get_all_outdegrees(weighted=weighted, topic=topic))

    def get_indegree(self, vertex: int, weighted: bool = False, topic: int = None) -> Union[float, int]:
        parent_mask = self.edge_array[:, 1] == vertex
        parent_weights = self.weights[parent_mask]
        
        if weighted:
            if topic is None:
                raise ValueError("Topic must be specified when weighted=True.")
            # Chọn trọng số của topic chỉ định và tính tổng trọng số cho topic đó
            return np.sum(parent_weights[:, topic])
        else:
            # Nếu không có trọng số, chỉ đếm số lượng cạnh vào
            return len(parent_weights)

    def get_all_indegrees(self, weighted: bool = False, topic: int = None) -> np.array:
        indegrees = []
        for v in self.get_vertices():
            indegrees.append(self.get_indegree(v, weighted=weighted, topic=topic))
        return np.array(indegrees)

    def get_avg_indegree(self, weighted: bool = False, topic: int = None) -> float:
        return np.mean(self.get_all_indegrees(weighted=weighted, topic=topic))

    def get_max_indegree(self, weighted: bool = False, topic: int = None) -> Union[int, float]:
        return np.max(self.get_all_indegrees(weighted=weighted, topic=topic))

    def get_vertex_2_indegree_dict(self, weighted: bool = False, topic: int = None) -> Dict[int, Union[float, int]]:
        """
        Trả về một từ điển ánh xạ mỗi đỉnh với bậc vào của nó.
        Nếu weighted = True, tính bậc vào có trọng số; nếu không tính bậc vào đơn giản.
        """
        return {vertex: self.get_indegree(vertex, weighted=weighted, topic=topic) for vertex in self.get_vertices()}

    def get_edges_mask_from_set_to_vertex(self, vertex_set: set, vertex: int) -> np.array:
        parent_edges_mask = self.get_parents_mask(vertex)
        parent_vertices = self.edge_array[:, 0][parent_edges_mask]
        parent_edges_mask[parent_edges_mask] = [(parent in vertex_set) for parent in parent_vertices]
        return parent_edges_mask

    def is_edge_in_graph(self, edge: Tuple[int, int]) -> bool:
        source, sink = edge
        return sink in self.get_children(source)
