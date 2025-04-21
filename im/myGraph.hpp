#ifndef MYGRAPH_HPP
#define MYGRAPH_HPP

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <ctime>
#include <fstream>
#include <queue>

class myGraph {
public:
    struct EdgeProperty {
        std::vector<double> topic_weights; // 3 trọng số cho 3 chủ đề
        EdgeProperty() : topic_weights(3, 0.0) {}
    };

    struct VertexProperty {
        double weight;
        VertexProperty() : weight(0.0) {}
    };

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty, EdgeProperty> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;

private:
    Graph g;
    boost::random::mt19937 rng; // Bộ sinh số ngẫu nhiên

public:
    myGraph() : rng(static_cast<unsigned>(std::time(0))) {}

    Graph& getGraph() {
        return g;
    }

    const Graph& getGraph() const {
        return g;
    }

    // Đọc đồ thị từ file
    bool readGraphFromFile(const std::string &filename) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Không thể mở file: " << filename << std::endl;
            return false;
        }

        std::vector<std::pair<int, int>> edges;
        std::unordered_set<int> vertices; // Để lưu các đỉnh thực tế
        int u, w;

        // Đọc danh sách các cạnh và xác định các đỉnh thực tế
        while (infile >> u >> w) {
            edges.emplace_back(u, w);
            vertices.insert(u);
            vertices.insert(w);
        }
        infile.close();

        // Tạo ánh xạ từ số thứ tự cũ sang số thứ tự mới
        std::unordered_map<int, int> old_to_new;
        int new_index = 0;

        // Đánh lại số thứ tự cho các đỉnh, chỉ xét các đỉnh trong vertices
        for (int vertex : vertices) {
            old_to_new[vertex] = new_index++;
        }

        // Tạo đồ thị mới với số đỉnh thực tế
        g = Graph(vertices.size());

        // Thêm các cạnh vào đồ thị mới, sử dụng số thứ tự đỉnh mới
        for (const auto &edge : edges) {
            int new_u = old_to_new[edge.first];
            int new_v = old_to_new[edge.second];
            boost::add_edge(new_u, new_v, g);
        }

        // Gán trọng số ngẫu nhiên cho các đỉnh
        boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it) {
            g[*it].weight = dist(rng);
        }

        // Gán trọng số ngẫu nhiên cho các cạnh
        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it) {
            for (int i = 0; i < 3; ++i) {
                g[*it].topic_weights[i] = dist(rng);
            }
        }

        return true;
    }

    bool writeGraphToFile(const std::string &filename) {
        std::ofstream outfile(filename);
        if (!outfile) {
            std::cerr << "Không thể mở file để ghi: " << filename << std::endl;
            return false;
        }

        // Ghi các đỉnh và trọng số của chúng (ghi theo định dạng mà readGraphFromFileWithWeights có thể đọc)
        outfile<< boost::num_vertices(g) << " " << boost::num_edges(g) << std::endl;
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it) {
            outfile << *it << " " << g[*it].weight << std::endl; // Ghi theo định dạng: id weight
        }

        // Ghi các cạnh và trọng số của chúng (3 trọng số cho mỗi cạnh)
        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it) {
            Vertex u = boost::source(*it, g);
            Vertex v = boost::target(*it, g);
            outfile << u << " " << v; // Ghi theo định dạng: u v
            for (int i = 0; i < 3; ++i) {
                outfile << " " << g[*it].topic_weights[i]; // Ghi trọng số của các chủ đề
            }
            outfile << std::endl;
        }

        outfile.close();
        return true;
    }

    bool readGraphFromFileWithWeights(const std::string &filename) {
        std::ifstream infile(filename);
        if (!infile) {
            std::cerr << "Không thể mở file: " << filename << std::endl;
            return false;
        }

        int u, v; // Các đỉnh u và v
        double vertex_weight;
        double edge_weights[3]; // Trọng số cho 3 chủ đề mỗi cạnh
        int max_vertex = -1;
        int n,m;
        infile >> n >> m;
        // Đọc các đỉnh và trọng số của chúng
        for(int i = 0; i < n; i++) 
        {
            infile >> u >> vertex_weight;
            // Thêm đỉnh vào đồ thị nếu nó chưa tồn tại
            while (boost::num_vertices(g) <= u) {
                boost::add_vertex(g); // Thêm đỉnh vào đồ thị
            }
            // Gán trọng số cho đỉnh u
            g[u].weight = vertex_weight;
            max_vertex = std::max(max_vertex, u);
        }

        for(int i = 0; i < m; i++){
            infile >> u >> v >> edge_weights[0] >> edge_weights[1] >> edge_weights[2];
            auto e = boost::add_edge(u, v, g);
            for (int i = 0; i < 3; ++i) {
                g[e.first].topic_weights[i] = edge_weights[i];
            }
        }

        infile.close();
        return true;
    }

    // Tính mức độ ảnh hưởng của tập hạt giống
    double computeInfluence(const std::vector<std::pair<int, int>> &seeds, int simulations = 1000) {
        // Kiểm tra tính hợp lệ của tập hạt giống
        for (const auto &seed : seeds) {
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
        for (int sim = 0; sim < simulations; ++sim) {
            double influence = runICMSimulation(seeds, dist);
            total_influence += influence;
        }

        return total_influence / simulations;
    }

    int num_vertices() const {
        return boost::num_vertices(g);
    }
    void printGraph() const {
        // In các đỉnh và trọng số của chúng
        std::cout << "Vertices and their weights:" << std::endl;
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it) {
            std::cout << *it << " " << g[*it].weight << std::endl;
        }
    
        // In các cạnh và trọng số của chúng
        std::cout << "Edges and their topic weights:" << std::endl;
        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it) {
            Vertex u = boost::source(*it, g);
            Vertex v = boost::target(*it, g);
            std::cout << u << " " << v;
            for (int i = 0; i < 3; ++i) {
                std::cout << " " << g[*it].topic_weights[i]; // Ghi trọng số của các chủ đề
            }
            std::cout << std::endl;
        }
    }
private:
    // Chạy một lần mô phỏng ICM
    double runICMSimulation(const std::vector<std::pair<int, int>> &seeds, boost::random::uniform_real_distribution<double> &dist) {
        std::vector<bool> activated(boost::num_vertices(g), false);
        std::vector<int> node_topic(boost::num_vertices(g), -1); // Lưu chủ đề kích hoạt của mỗi nút
        std::queue<std::pair<Vertex, int>> q;                    // Hàng đợi chứa cặp (nút, chủ đề)

        for (const auto &seed : seeds) {
            Vertex u = seed.first;
            int topic = seed.second;
            if (!activated[u]) {
                activated[u] = true;
                node_topic[u] = topic;
                q.push({u, topic});
            }
        }

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

        // Tính số nút được kích hoạt
        double influence = 0.0;
        for (size_t i = 0; i < activated.size(); ++i) {
            if (activated[i]) {
                influence++;
            }
        }

        return influence;
    }
};

#endif // MYGRAPH_HPP
