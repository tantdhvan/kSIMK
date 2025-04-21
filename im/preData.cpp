#include "myGraph.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Kiểm tra xem người dùng đã truyền đủ 2 đối số vào chương trình chưa
    if (argc != 3) {
        std::cerr << "Cần truyền vào 2 tên file: file đầu vào và file đầu ra." << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];  // File đầu vào
    std::string outputFile = argv[2]; // File đầu ra

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

    // Đọc lại đồ thị từ file đã ghi
    myGraph newGraph;
    if (!newGraph.readGraphFromFileWithWeights(outputFile)) {
        std::cerr << "Lỗi khi đọc lại đồ thị từ file " << outputFile << "." << std::endl;
        return 1;
    }

    std::cout << "Đọc và ghi đồ thị thành công từ " << inputFile << " đến " << outputFile << "." << std::endl;

    return 0;
}
