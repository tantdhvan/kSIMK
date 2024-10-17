#pragma once
#include "Constants.h"
#include <omp.h>
#include <string>
#include <vector>

using namespace std;

typedef unsigned int uint;
typedef pair<uint, uint> kpoint;
typedef vector<kpoint> kseeds;
typedef vector<double> kbins;
typedef vector<kbins> ksensors;

typedef struct myType{
    kseeds s;
    int nguong;
    double nguongd;
    double current_f=0;
    vector<bool> check;
    vector<double> cost;
    string code;
};

class Kcommon {
public:
  Kcommon();
  ~Kcommon();

  static Kcommon *getInstance();
  unsigned randomInThread(int thread_id);

private:
  static Kcommon *instance;
  int *seed;
};
