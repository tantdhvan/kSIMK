#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <tuple>

#include "Constants.h"
#include "Degree.h"
#include "Deterministic.h"
#include "Greedy.h"
#include "Network.h"
#include "Randomized.h"
#include "SampleRandomized.h"
#include "SingleGreedy.h"
#include "StreamingGreedy.h"

// #include <time.h>
#include <chrono>
#include <iomanip> // setprecision
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

using namespace std;

#pragma warning(disable : 4996)
vector<uint> B_vals = {15, 20, 25, 30, 35, 40, 45, 50};
vector<uint> M_vals = {3};
vector<double> Delta_vals = {1};

Network *g;

void print_help()
{
  cout << "Options: " << endl;
  cout << "-f <data filename>" << endl
       << "-V <size of V> # default: 50" << endl
       << "-t <type of experiment, 0: influence maximization, 1: sensor "
          "placement> # default: 0"
       << endl
       << "-k <value of k> # default: 3" << endl
       << "-B <value of B> # default: 10" << endl
       << "-M <value of M> # default: 3" << endl
       << "-e <value of epsilon> # default: 0.5 for IM and 0 for sensor "
          "placement"
       << endl
       << "-n <value of eta - denoise step for RStream> # default: 2" << endl
       << "-g <value of gamma> # default: 1.0" << endl
       << "-a <algorithm, 1: DStream, 2: RStream, 3: SGr>, 4: SampleRstream # "
          "default: 1, please use SSA source code for testing IM algorithm"
       << endl
       << "-p <number of threads to running algorithms> # default: 4" << endl;
}

pair<string, int> parseArgs(int argc, char **argv)
{
  string filename = "error";
  int no_nodes = 50;

  int i = 1;
  while (i < argc - 1)
  {
    string arg = argv[i];
    if (arg == "-f")
    {
      filename = argv[i + 1];
    }
    else if (arg == "-g" || arg == "-e")
    {
      string s_val = argv[i + 1];
      std::string::size_type sz;
      Constants::EPS = std::stod(s_val, &sz);
    }
    else
    {
      string s_val = argv[i + 1];
      std::string::size_type sz;
      int val = std::stoi(s_val, &sz);
      if (arg == "-t")
      {
        if(val == 0)Constants::DATA=Social;
        if(val == 1)Constants::DATA=Sensor;
        if(val == 2)Constants::DATA=Revenue;
        //Constants::DATA = val == 0 ? Social : Sensor;

        if (Constants::DATA == Sensor)
        {
          Constants::NO_DENOISE_STEPS = 1;
        }
      }
      else if (arg == "-V")
      {
        no_nodes = val;
      }
      else if (arg == "-k")
      {
        Constants::K = val;
      }
      else if (arg == "-r")
      {
        Constants::RHO = val;
      }
      else if (arg == "-b")
      {
        Constants::BETA = val;
      }
      else if (arg == "-B")
      {
        Constants::BUDGET = val;
      }
      else if (arg == "-n")
      {
        Constants::NO_DENOISE_STEPS = val;
        if (Constants::DATA == Sensor)
        {
          Constants::NO_DENOISE_STEPS = 1;
        }
      }
      else if (arg == "-a")
      {
        switch (val)
        {
        case 0:
          Constants::ALGORITHM = aGreedy;
          break;
        case 5:
          Constants::ALGORITHM = Alg1;
          break;
        }
      }
      else if (arg == "-p")
      {
        Constants::NUM_THREAD = val;
      }
    }
    i += 2;
  }

  return pair<string, int>(filename, no_nodes);
}

void run_command(string filename, int no_nodes)
{
  Network *g = new Network();
  bool r = false;
  if (Constants::DATA == Social)
  {
    r = g->read_network_from_file(no_nodes, filename, false);
  }
  else if (Constants::DATA == Revenue)
  {
    r=g->read_graph_from_file_custom(filename);
  }
  else
    r = g->read_sensor_data(no_nodes, filename);

  if (!r)
  {
    cout << "Wrong file, format or arguments" << endl;
    print_help();
  }
  else
  {
    double sol = 0.0;
    int no_queries = 0;
    omp_set_num_threads(Constants::NUM_THREAD);
    switch (Constants::ALGORITHM)
    {
    case aGreedy:
    {
      Greedy *gr = new Greedy(g);
      sol = gr->get_solution();
      no_queries = gr->get_no_queries();
      delete gr;
      break;
    }
    case Alg1:
    {
      Greedy *gr = new Greedy(g);
      sol = gr->get_solution1();
      no_queries = gr->get_no_queries();
      delete gr;
      break;
    }
    }
    if (Constants::DATA == Sensor)
      sol = sol / 100;

    std::cout << Constants::ALGORITHM << "," << Constants::EPS << ","
              << Constants::BUDGET << "," << sol << "," << no_queries << ",";
  }
  delete g;
}

void print_performances(struct rusage *performance_start,
                        struct rusage *performance_end)
{
  cout << (performance_end->ru_utime.tv_sec -
           performance_start->ru_utime.tv_sec) +
              1e-6 * (performance_end->ru_utime.tv_usec -
                      performance_start->ru_utime.tv_usec)
       << endl;
}

int main(int argc, char *argv[])
{
  srand(time(NULL));
  pair<string, int> r = parseArgs(argc, argv);
  struct rusage performance_start, performance_end;
  getrusage(RUSAGE_SELF, &performance_start);

  if (r.first == "error")
  {
    std::cout << "Wrong file, format or arguments" << endl;
    print_help();
  }
  else
  {
    run_command(r.first, r.second);
  }

  getrusage(RUSAGE_SELF, &performance_end);
  print_performances(&performance_start, &performance_end);
  return 0;
}