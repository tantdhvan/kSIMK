#pragma once
#include "Framework.h"

class Greedy : public Framework {
public:
  Greedy(Network *g);
  ~Greedy();
  double get_solution(bool is_ds = true);
  double get_solution1(bool is_ds = true);
  double get_solution2(bool is_ds = true);
  double get_imax(int e, int &i_max);
  void update_nguong(vector<myType> &nguong,vector<int> &nguongj, double f_e_i_max);
  int get_no_queries();

private:
  int no_queries;
};
