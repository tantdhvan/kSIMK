#pragma once
#include "Kcommon.h"
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>

using namespace std;

class Network {
public:
    Network();
    ~Network();
    int get_no_nodes();

    // for social network
    int get_out_degree(uint n);
    bool read_network_from_file(int no_nodes, string file, bool is_directed);
    bool read_graph_from_file_custom(string file); // Hàm đọc đồ thị theo định dạng mới
    void generate_random_network(int no_nodes, double p, bool is_directed);
    uint sample_influence(const kseeds &seeds); // estimate influence
    uint sample_influence_reverse(const kseeds &seeds); // estimate influence but use reverse sampling
    uint sample_influence_linear_threshold(const kseeds &seeds);

    // for sensor
    bool read_sensor_data(int no_nodes, string file);
    double get_entropy(const kseeds &seeds);
    vector<double> read_cost();
    unordered_map<uint, uint> get_map_node_id(){ return map_node_id; };
    vector<vector<uint>> get_out_neighbors(){ return out_neighbors; };
    vector<vector<uint>> get_in_neighbors(){ return in_neighbors; }; 
    vector<vector<uint>> get_preferences(){ return preferences; };
    vector<double> get_probabilities(){ return probabilities; };
    vector<double> get_node_cost(){ return node_cost; };
    vector<vector<double>> get_node_revenue(){ return node_revenue; };
    vector<std::unordered_map<uint, double>> get_edge_weights(){ return edge_weights; };
    bool get_is_directed(){return is_directed;};
private:
    void clear();
    void recursive_entropy(int idx, const kseeds &seeds, double &re, double &prob); // used to calculate entropy

    uint number_of_nodes;
    Kcommon *common_instance;

    // for social network
    bool is_directed=false;
    std::unordered_map<uint, uint> map_node_id; // map from true id -> ordered id (used for read graph from file)
    vector<vector<uint>> out_neighbors, in_neighbors; // map from node_id -> list of out (in) neighbor of the node
    vector<vector<uint>> preferences; // map from node_id -> preferences on partition - this impacts the weight of an out-edge with adopting different product
    vector<double> probabilities; // map from preference -> probability to influence

    // Lưu trữ chi phí và lợi nhuận của mỗi đỉnh
    vector<double> node_cost;
    vector<vector<double>> node_revenue;
    vector<std::unordered_map<uint, double>> edge_weights;
    //map<uint, uint> map_node_id;
    int num_products;

    // for sensor data
    vector<ksensors> sensor_data; // map from loc id -> kind of sensor (0 temp, 1 humid, 2 light) -> bin
    int max_no_bin; // no. bins
};
