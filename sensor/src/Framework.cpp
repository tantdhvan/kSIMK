#include "Framework.h"
#include <math.h>

Framework::Framework(Network *g)
{
  uint no_nodes = g->get_no_nodes();
  cost_matrix = g->read_cost();
  this->g = g;
  no_samples = ceil(6.75 * ((double)(g->get_no_nodes())) / (Constants::EPS * Constants::EPS));
  no_samples = 400;
}

Framework::~Framework() {}

double Framework::estimate_test(const kseeds &seeds, uint n)
{
  double re = 0.0;
  vector<bool> a(n, false);

#pragma omp parallel for
  for (int i = 0; i < n; ++i)
  {
    a[i] = (g->sample_influence_linear_threshold(seeds) == 1);
  }

  for (bool b : a)
    if (b)
      re += 1.0;

  return re / n * g->get_no_nodes();
}

double Framework::estimate_influence(const kseeds &seeds)
{
  if (Constants::DATA == Social)
  {
    double re = 0.0;
    uint need_sample = 4000; // Hard code need_sample
    vector<bool> a(need_sample, false);

    #pragma omp parallel for
    for (int i = 0; i < need_sample; ++i)
    {
      a[i] = (g->sample_influence_linear_threshold(seeds) == 1);
    }

    for (bool b : a)
      if (b)
        re += 1.0;

    return re / need_sample * g->get_no_nodes();
  }
  else if (Constants::DATA == Revenue)
  { 
    //cout<<"start f(.)"<<endl;
    uint no_nodes=g->get_no_nodes();
    std::vector<std::vector<char>> seed_sets(Constants::K, std::vector<char>(no_nodes,false));
    std::vector<char> seeded_nodes(no_nodes, false);
    for (const auto &kp : seeds)
    {
      uint node = kp.first;
      uint product = kp.second;
      if (seeded_nodes[node])
      {
        continue;
      }
      seed_sets[product][node] = 1;
      seeded_nodes[node] = 1;
    }
    //cout<<"1"<<endl;
    double total_revenue = 0.0;
    vector<vector<uint>> out_neighbors=g->get_out_neighbors();
    vector<unordered_map<uint, double>> edge_weights=g->get_edge_weights();
    #pragma omp parallel for reduction(+:total_revenue) schedule(dynamic)
    for (uint i = 0; i < no_nodes; ++i)
    {
      if (!seeded_nodes[i])
      {
        double user_revenue = 0.0;
        for (size_t p = 0; p < Constants::K; ++p)
        {
          double influence_sum = 0.0;
          for (int jj=0;jj<out_neighbors[i].size(); jj++)
          {
            int j=out_neighbors[i][jj];
            if (seed_sets[p][j])
            {
                influence_sum += edge_weights[i][j];
            }
          }         
          user_revenue += pow(influence_sum, g->get_node_revenue()[i][p]);
        }
        total_revenue += user_revenue;
      }
    }
    //cout<<"end"<<endl;
    return total_revenue;
  }
  else
  {
    return g->get_entropy(seeds);
  }
}
