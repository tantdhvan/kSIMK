#include "Streaming.h"
#include <algorithm>
#include <iostream>
#include <math.h>

Streaming::Streaming(Network *g) : Framework(g) {
  int no_nodes = g->get_no_nodes();
  for (int i = 0; i < no_nodes; ++i) {
    node_sequence.push_back(i);
  }
  random_shuffle(node_sequence.begin(), node_sequence.end());
  no_queries = 0;
  max_solution = 0.0;
}

Streaming::~Streaming() {}

double Streaming::get_solution(bool is_dstream) {
  int j_min, j_max = 0, p_max = 0;
  int c_passed = 0;
  double max_gain = 0.0;
  double log_delta = log(1 + Constants::EPS_TAG);

  for (uint e : node_sequence) {
    ++c_passed;

    /* get max gain of a single seed (e,i), find max_gain */
    for (int i = 0; i < Constants::K; ++i) {
      kseeds single_seed = {kpoint(e, i)};
      double current_inf = estimate_influence(single_seed);
      ++no_queries;

      if (max_gain < current_inf) {
        max_gain = current_inf;
        j_min = ceil(log(max_gain / log_delta));

        if (is_dstream) {
          /* upper bound if ds is running */
          j_max = floor(log(max_gain * Constants::BUDGET) / log_delta);
        } else {
          /* upper bound if rs is running */
          j_max = floor(log(max_gain * Constants::BUDGET) / log_delta);
        }

        j_min = max(j_min, 0); /* could < 0 */
        j_max = max(j_max, 0); /* could < 0 */

        /* Resize subseeds if having new j variable */
        if (j_max + 1 > sub_seeds.size()) {
          sub_seeds.resize(j_max + 1,
                           vector<kseeds>(Constants::NO_DENOISE_STEPS));
          sub_seeds_cost.resize(
              j_max + 1, vector<double>(Constants::NO_DENOISE_STEPS, 0.0));
          influences.resize(j_max + 1,
                            vector<double>(Constants::NO_DENOISE_STEPS, 0.0));
          for (int j = thresholds.size(); j < j_max + 1; ++j) {
            thresholds.push_back(pow(1 + Constants::EPS_TAG, j));
          }
        }
      }
    }

    /* Special case for Algorithm 5 */
    vector<int> rhos(j_max + 1);
    if (Constants::NO_DENOISE_STEPS > 1) {
      for (int j = j_min; j <= j_max; ++j) {
        int p = floor(
            (4 * log(1 / Constants::DELTA)) /
            (pow(Constants::EPS_TAG, 2) * Constants::ALPHA * thresholds[j]));
        cout << "/* Special case for Algorithm 5 */"
             << " " << p << endl;
        rhos[j] = max(0, p);
        if (p + 1 > sub_seeds_cost[j].size()) {
          sub_seeds[j].resize(p + 1, vector<kpoint>(0));
          sub_seeds_cost[j].resize(p + 1, 0.0);
          influences[j].resize(p + 1, 0.0);
        }
      }
    }

    /* select element for each sub seed */
    if (sub_seeds.size() > 0) {
      for (int j = j_min; j <= j_max; ++j) {
        int denoise_step = max(1, rhos[j]);
        for (int step = 0; step < denoise_step; ++step) {
          if (sub_seeds_cost[j][step] + cost_matrix[e] < Constants::BUDGET) {
            int i = select_element(j, e, step);
            if (i != -1) {
              sub_seeds[j][step].push_back(kpoint(e, i));
              sub_seeds_cost[j][step] += cost_matrix[e];
            }
            // cout << "Select element ... " << e << ", " << i << endl;
          }
        }
      }
    }
  }
  return max_solution;
}

uint Streaming::get_no_queries() { return no_queries; }
