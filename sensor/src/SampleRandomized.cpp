#include "SampleRandomized.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

SampleRandomized::SampleRandomized(Network *g) : Streaming(g) {
  // Calculating ALPHA and EPSILON_TAG
  Constants::ALPHA =
      2 / (3 + Constants::BETA - (Constants::BETA / Constants::K));
  Constants::EPS_TAG = (2 * Constants::EPS) / Constants::ALPHA;
  Constants::NO_DENOISE_STEPS = 2;
  cout << "Algorithm 5 is running ..." << endl;
}

SampleRandomized::~SampleRandomized() {}

int SampleRandomized::select_element(int j, uint e, int step) {
  vector<double> temp_inf(Constants::K), p(Constants::K);
  uint J_size = 0;

  for (int i = 0; i < Constants::K; ++i) {
    kseeds tmp_seeds = sub_seeds[j][step];
    tmp_seeds.push_back(kpoint(e, i));
    temp_inf[i] = estimate_influence(tmp_seeds);
    ++no_queries;

    bool critical_cond =
        sub_seeds_cost[j][step] + cost_matrix[e] <= Constants::BUDGET &&
        ((temp_inf[i] - influences[j][step]) / cost_matrix[e] >=
         (Constants::ALPHA * thresholds[j]) / Constants::BUDGET);

    if (critical_cond) {
      p[i] = (temp_inf[i] - influences[j][step]) / cost_matrix[e];
      if (p[i] > 0)
        ++J_size;
      // p[i] = p[i] >= thresholds[j] ? p[i] : 0;
    }
  }

  if (J_size == 0) {
    return -1;
  } else if (J_size == 1) {
    for (int i = 0; i < Constants::K; ++i) {
      if (p[i] > 0) {
        influences[j][step] = temp_inf[i];
        if (influences[j][step] > max_solution)
          max_solution = influences[j][step];
        return i;
      }
    }
  } else {
    double T = 0.0;
    double sum = 0.0;

    for (int i = 0; i < Constants::K; ++i) {
      p[i] = pow(p[i], J_size - 1);
      T += p[i];
    }

    double random = (double)(rand() % 10000) / 10000 * T;
    for (int i = 0; i < Constants::K; ++i) {
      if (sum <= random && random < sum + p[i]) {
        influences[j][step] = temp_inf[i];
        max_solution = max(max_solution, influences[j][step]);
        return i;
      }
      sum += p[i];
    }
  }

  return -1;
}
