#pragma once
#include "Framework.h"

class Streaming : public Framework {
public:
  Streaming(Network *g);
  ~Streaming();
  double get_solution(bool is_ds = true);
  virtual int select_element(int j, uint e, int step) = 0;
  uint get_no_queries();

protected:
  double max_solution;
  vector<vector<kseeds>> sub_seeds;
  vector<vector<double>> sub_seeds_cost;
  vector<double> thresholds; // store value of (1+\delta)^j
  vector<vector<double>> influences;
  vector<uint> node_sequence;
  uint no_queries;
};
