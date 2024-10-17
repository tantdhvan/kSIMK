#include "Deterministic.h"
#include <iostream>
#include <math.h>

Deterministic::Deterministic(Network *g) : Streaming(g) {
  Constants::ALPHA = 0.5;
  Constants::EPS_TAG = 4 * Constants::EPS;

  // Re-initialize cost_matrix
  int no_nodes = g->get_no_nodes();
}

Deterministic::~Deterministic() {}

int Deterministic::select_element(int j, uint e, int step) {
  uint i_max = 0;
  double max_inf = 0.0;

  /* Find i_max */
  for (int i = 0; i < Constants::K; ++i) {
    kseeds seeds_tmp = sub_seeds[j][step];
    seeds_tmp.push_back(kpoint(e, i));
    double current_inf = estimate_influence(seeds_tmp);
    ++no_queries;
    if (max_inf < current_inf) {
      max_inf = current_inf;
      i_max = i;
    }
  }

  /*
   * No need to consider denoise step in deterministic
   * double eps = Constants::EPS / (Constants::NO_DENOISE_STEPS > 1 ?
   * Constants::NO_DENOISE_STEPS - 1 : Constants::NO_DENOISE_STEPS) * step;
   */
  if (i_max >= 0 &&
      (max_inf / (sub_seeds_cost[j][step] + cost_matrix[e])) >=
          (Constants::ALPHA * thresholds[j] / Constants::BUDGET)) {
    max_solution = max(max_solution, max_inf);
    return i_max;
  } else {
    return -1;
  }
}
