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
#include <omp.h>

class myGraph
{
public:
    struct EdgeProperty
    {
        std::vector<double> topic_weights;
        EdgeProperty() : topic_weights(3, 0.0) {}
    };

    struct VertexProperty
    {
        double weight;
        VertexProperty() : weight(0.0) {}
    };

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperty, EdgeProperty> Graph;
    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;

private:
    Graph g;
    boost::random::mt19937 rng;
    int simulations = 1000;

public:
    myGraph() : rng(static_cast<unsigned>(std::time(0))) {}

    Graph &getGraph()
    {
        return g;
    }

    const Graph &getGraph() const
    {
        return g;
    }

    bool readGraphFromFile(const std::string &filename)
    {
        std::ifstream infile(filename);
        if (!infile)
        {
            std::cerr << "Không thể mở file: " << filename << std::endl;
            return false;
        }

        std::vector<std::pair<int, int>> edges;
        std::unordered_set<int> vertices;
        int u, w;

        while (infile >> u >> w)
        {
            edges.emplace_back(u, w);
            vertices.insert(u);
            vertices.insert(w);
        }
        infile.close();

        std::unordered_map<int, int> old_to_new;
        int new_index = 0;

        for (int vertex : vertices)
        {
            old_to_new[vertex] = new_index++;
        }

        g = Graph(vertices.size());

        for (const auto &edge : edges)
        {
            int new_u = old_to_new[edge.first];
            int new_v = old_to_new[edge.second];
            boost::add_edge(new_u, new_v, g);
        }

        boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it)
        {
            g[*it].weight = dist(rng);
        }

        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it)
        {
            for (int i = 0; i < 3; ++i)
            {
                g[*it].topic_weights[i] = dist(rng);
            }
        }

        return true;
    }

    bool writeGraphToFile(const std::string &filename)
    {
        std::ofstream outfile(filename);
        if (!outfile)
        {
            std::cerr << "Không thể mở file để ghi: " << filename << std::endl;
            return false;
        }

        outfile << boost::num_vertices(g) << " " << boost::num_edges(g) << std::endl;
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it)
        {
            outfile << *it << " " << g[*it].weight << std::endl;
        }

        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it)
        {
            Vertex u = boost::source(*it, g);
            Vertex v = boost::target(*it, g);
            outfile << u << " " << v;
            for (int i = 0; i < 3; ++i)
            {
                outfile << " " << g[*it].topic_weights[i];
            }
            outfile << std::endl;
        }

        outfile.close();
        return true;
    }

    bool readGraphFromFileWithWeights(const std::string &filename)
    {
        std::ifstream infile(filename);
        if (!infile)
        {
            std::cerr << "Không thể mở file: " << filename << std::endl;
            return false;
        }

        int u, v;
        double vertex_weight;
        double edge_weights[3];
        int max_vertex = -1;
        int n, m;
        infile >> n >> m;
        for (int i = 0; i < n; i++)
        {
            infile >> u >> vertex_weight;
            while (boost::num_vertices(g) <= u)
            {
                boost::add_vertex(g);
            }
            g[u].weight = vertex_weight;
            max_vertex = std::max(max_vertex, u);
        }

        for (int i = 0; i < m; i++)
        {
            infile >> u >> v >> edge_weights[0] >> edge_weights[1] >> edge_weights[2];
            auto e = boost::add_edge(u, v, g);
            for (int i = 0; i < 3; ++i)
            {
                g[e.first].topic_weights[i] = edge_weights[i];
            }
        }

        infile.close();
        return true;
    }

    double computeInfluence(const std::vector<std::pair<int, int>> &seeds)
    {
        double total_influence = 0.0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            boost::random::mt19937 local_rng(static_cast<unsigned>(std::time(nullptr)) + tid); // mỗi thread có RNG riêng

            #pragma omp for reduction(+ : total_influence)
            for (int sim = 0; sim < simulations; ++sim)
            {
                std::vector<std::pair<int, int>> seeds_copy = seeds;
                double influence = runICMSimulation(seeds_copy, local_rng);
                total_influence += influence;
            }
        }

        return total_influence / simulations;
    }

    int num_vertices() const
    {
        return boost::num_vertices(g);
    }
    void printGraph() const
    {
        std::cout << "Vertices and their weights:" << std::endl;
        auto v = boost::vertices(g);
        for (auto it = v.first; it != v.second; ++it)
        {
            std::cout << *it << " " << g[*it].weight << std::endl;
        }

        std::cout << "Edges and their topic weights:" << std::endl;
        auto e = boost::edges(g);
        for (auto it = e.first; it != e.second; ++it)
        {
            Vertex u = boost::source(*it, g);
            Vertex v = boost::target(*it, g);
            std::cout << u << " " << v;
            for (int i = 0; i < 3; ++i)
            {
                std::cout << " " << g[*it].topic_weights[i];
            }
            std::cout << std::endl;
        }
    }

private:
    double runICMSimulation(const std::vector<std::pair<int, int>> &seeds,
                            boost::random::mt19937 &rng)
    {
        std::queue<std::pair<int, int>> q;
        std::set<std::pair<int, int>> active;                      
        boost::random::uniform_real_distribution<> dist(0.0, 1.0);

        for (const auto &seed : seeds)
        {
            q.push(seed);
            active.insert(seed);
        }

        while (!q.empty())
        {
            auto current = q.front();
            q.pop();

            int u = current.first;
            int topic = current.second;

            auto out_edges = boost::out_edges(u, g);
            for (auto it = out_edges.first; it != out_edges.second; ++it)
            {
                int v = boost::target(*it, g);
                std::pair<int, int> neighbor = {v, topic};

                if (active.find(neighbor) == active.end())
                {
                    double prob = g[*it].topic_weights[topic];
                    double random_val = dist(rng);

                    if (random_val <= prob)
                    {
                        active.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }
        }

        return static_cast<double>(active.size());
    }
};

#endif // MYGRAPH_HPP
