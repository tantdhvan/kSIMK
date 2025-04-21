#include "myGraph.hpp"
#include "Greedy.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Cần 3 đối số: file đầu vào, file đầu ra, ngưỡng trọng số b." << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    double b = std::stod(argv[3]);

    myGraph graph;

    if(!graph.readGraphFromFileWithWeights(inputFile)) {
        std::cerr << "Lỗi khi đọc đồ thị từ file " << inputFile << "." << std::endl;
        return 1;
    }
    //graph.printGraph();

    Greedy greedy(graph, b);
    auto seedSet = greedy.findOptimalSeedSet();

    std::cout << "Tập hạt giống tối ưu (nút, chủ đề):" << std::endl;
    for (const auto& seed : seedSet) {
        std::cout << "Node: " << seed.first << ", Topic: " << seed.second << std::endl;
    }

    return 0;
}
