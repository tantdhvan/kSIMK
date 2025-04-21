#include "myGraph.hpp"
#include <iostream>

int main() {
    myGraph graph;

    // Đọc đồ thị từ file
    if (!graph.readGraphFromFile("facebook.txt")) {
        return 1;
    }

    // Tập hạt giống: mỗi nút được gán một chủ đề
    std::vector<std::pair<int, int>> seeds = {
        {0, 0}, // Nút 0, chủ đề 0
        {1, 1}  // Nút 1, chủ đề 1
    };

    // Tính mức độ ảnh hưởng
    double influence = graph.computeInfluence(seeds, 1000);
    std::cout << "Mức độ ảnh hưởng: " << influence << std::endl;

    return 0;
}