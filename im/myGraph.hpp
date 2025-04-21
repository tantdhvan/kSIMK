#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <ctime>

class myGraph {
public:
    // Định nghĩa thuộc tính cho cạnh (3 trọng số cho 3 chủ đề)
    struct EdgeProperty {
        std::vector<double> topic_weights; // 3 trọng số cho 3 chủ đề
        EdgeProperty() : topic_weights(3, 0.0) {}
    };

    // Định nghĩa thuộc tính cho đỉnh (1 trọng số)
    struct VertexProperty {
        double weight;
        VertexProperty() : weight(0.0) {}
    };

    // Định nghĩa đồ thị sử dụng adjacency_list
    typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::directedS,
        VertexProperty, EdgeProperty> Graph;

    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;

private:
    Graph g;
    boost::random::mt19937 rng; // Bộ sinh số ngẫu nhiên

public:
    myGraph() : rng(static_cast<unsigned>(std::time(0))) {}

    // Đọc đồ thị từ file và gán trọng số ngẫu nhiên
    bool readGraphFromFile(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Không thể mở file: " << filename << std::endl;
            return false;
        }

        std::vector<std::pair<int, int>> edges;
        int u, w; // Đổi tên biến v thành w để tránh xung đột
        int max_vertex = -1;

        // Đọc danh sách cạnh
        while (infile >> u >> w) {
            edges.emplace_back(u, w);
            max_vertex = std::max({max_vertex, u, w});
        }
        infile.close();

        // Tạo đồ thị với số đỉnh đủ lớn
        g = Graph(max_vertex + 1);

        // Thêm các cạnh
        for (const auto& edge : edges) {
            boost::add_edge(edge.first, edge.second, g);
        }

        // Gán trọng số ngẫu nhiên cho đỉnh
        boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it) {
            g[*it].weight = dist(rng);
        }

        // Gán trọng số ngẫu nhiên cho cạnh (3 trọng số cho mỗi cạnh)
        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it) {
            for (int i = 0; i < 3; ++i) {
                g[*it].topic_weights[i] = dist(rng);
            }
        }

        return true;
    }

    // Tính mức độ ảnh hưởng của tập hạt giống
    double computeInfluence(const std::vector<std::pair<int, int>>& seeds, int simulations = 1000) {
        // Kiểm tra tính hợp lệ của tập hạt giống
        for (const auto& seed : seeds) {
            if (seed.first < 0 || seed.first >= boost::num_vertices(g)) {
                std::cerr << "Nút không hợp lệ trong tập hạt giống: " << seed.first << std::endl;
                return 0.0;
            }
            if (seed.second < 0 || seed.second >= 3) {
                std::cerr << "Chủ đề không hợp lệ trong tập hạt giống: " << seed.second << std::endl;
                return 0.0;
            }
        }

        double total_influence = 0.0;
        boost::random::uniform_real_distribution<double> dist(0.0, 1.0);

        // Chạy mô phỏng Monte Carlo
        #pragma omp parallel for
        for (int sim = 0; sim < simulations; ++sim) {
            double influence = runICMSimulation(seeds, dist);
            #pragma omp critical
            {
                total_influence += influence;
            }
        }

        return total_influence / simulations;
    }

private:
    // Chạy một lần mô phỏng ICM
    double runICMSimulation(const std::vector<std::pair<int, int>>& seeds,
                           boost::random::uniform_real_distribution<double>& dist) {
        std::vector<bool> activated(boost::num_vertices(g), false);
        std::vector<int> node_topic(boost::num_vertices(g), -1); // Lưu chủ đề kích hoạt của mỗi nút
        std::queue<std::pair<Vertex, int>> q; // Hàng đợi chứa cặp (nút, chủ đề)

        // Kích hoạt các hạt giống
        for (const auto& seed : seeds) {
            Vertex u = seed.first;
            int topic = seed.second;
            if (!activated[u]) {
                activated[u] = true;
                node_topic[u] = topic;
                q.push({u, topic});
            }
        }

        // Lan truyền theo ICM
        while (!q.empty()) {
            auto [u, topic] = q.front();
            q.pop();

            // Duyệt các láng giềng
            auto out_edges = boost::out_edges(u, g);
            for (auto e_it = out_edges.first; e_it != out_edges.second; ++e_it) {
                Vertex v = boost::target(*e_it, g);
                if (!activated[v]) {
                    // Thử kích hoạt với xác suất bằng trọng số của cạnh cho chủ đề
                    if (dist(rng) < g[*e_it].topic_weights[topic]) {
                        activated[v] = true;
                        node_topic[v] = topic; // Gán chủ đề của nút v
                        q.push({v, topic});
                    }
                }
            }
        }

        // Tính tổng trọng số của các đỉnh được kích hoạt
        double influence = 0.0;
        for (size_t i = 0; i < activated.size(); ++i) {
            if (activated[i]) {
                influence += g[i].weight;
            }
        }

        return influence;
    }
};