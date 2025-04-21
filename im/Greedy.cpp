#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include "myGraph.hpp"

class Greedy {
public:
    // Constructor nhận vào đối tượng đồ thị và ngưỡng b
    Greedy(myGraph& g, double b) : graph(g), b(b) {}

    // Hàm tìm tập hạt giống tối ưu với ảnh hưởng lớn nhất, tuân theo giới hạn tổng trọng số chủ đề
    std::vector<std::pair<int, int>> findOptimalSeedSet(int max_simulations = 1000) {
        std::vector<std::pair<int, int>> seedSet;  // Tập hạt giống (nút, chủ đề)
        std::vector<bool> selected(graph.num_vertices(), false); // Đánh dấu đỉnh đã chọn
        std::vector<double> topicWeights(3, 0.0); // Tổng trọng số cho các chủ đề

        while (true) {
            double maxInfluence = -1;
            int bestNode = -1;
            int bestTopic = -1;

            // Tìm đỉnh có ảnh hưởng lớn nhất mà không vượt quá ngưỡng b
            for (int node = 0; node < graph.num_vertices(); ++node) {
                if (!selected[node]) {
                    // Tính ảnh hưởng của đỉnh node cho từng chủ đề
                    for (int topic = 0; topic < 3; ++topic) {
                        if (topicWeights[topic] + graph[node].weight <= b) { // Kiểm tra ngưỡng trọng số chủ đề
                            double influence = computeInfluenceForNode(node, topic, max_simulations);
                            if (influence > maxInfluence) {
                                maxInfluence = influence;
                                bestNode = node;
                                bestTopic = topic;
                            }
                        }
                    }
                }
            }

            // Nếu không thể chọn đỉnh nào nữa, dừng lại
            if (bestNode == -1) {
                break;
            }

            // Chọn đỉnh và chủ đề tốt nhất
            selected[bestNode] = true;
            seedSet.push_back({bestNode, bestTopic});
            topicWeights[bestTopic] += graph[bestNode].weight; // Cập nhật trọng số cho chủ đề đã chọn
        }

        return seedSet;
    }

private:
    myGraph& graph;  // Tham chiếu đến đối tượng đồ thị
    double b;  // Ngưỡng tổng trọng số của từng chủ đề

    // Hàm tính ảnh hưởng của một đỉnh cho một chủ đề cụ thể
    double computeInfluenceForNode(int node, int topic, int simulations) {
        // Mô phỏng ICM để tính ảnh hưởng cho node và chủ đề này
        std::vector<std::pair<int, int>> seeds = {{node, topic}};  // Tạo hạt giống từ node và topic
        return graph.computeInfluence(seeds, simulations);
    }
};
