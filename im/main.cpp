#include "myGraph.hpp"
#include "Greedy.hpp"  // Bao gồm header của class Greedy
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Kiểm tra đối số dòng lệnh
    if (argc != 4) {
        std::cerr << "Cần 3 đối số: file đầu vào, file đầu ra, ngưỡng trọng số b." << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    double b = std::stod(argv[3]); // Ngưỡng trọng số

    myGraph graph;

    // Đọc đồ thị từ file đầu vào
    if (!graph.readGraphFromFile(inputFile)) {
        std::cerr << "Lỗi khi đọc đồ thị từ file " << inputFile << "." << std::endl;
        return 1;
    }

    // Ghi đồ thị vào file đầu ra
    if (!graph.writeGraphToFile(outputFile)) {
        std::cerr << "Lỗi khi ghi đồ thị vào file " << outputFile << "." << std::endl;
        return 1;
    }

    // Tạo đối tượng Greedy và tìm tập hạt giống tối ưu
    Greedy greedy(graph, b);
    auto seedSet = greedy.findOptimalSeedSet();

    // In ra tập hạt giống
    std::cout << "Tập hạt giống tối ưu (nút, chủ đề):" << std::endl;
    for (const auto& seed : seedSet) {
        std::cout << "Node: " << seed.first << ", Topic: " << seed.second << std::endl;
    }

    return 0;
}
