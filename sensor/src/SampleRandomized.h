#pragma once
#include "Streaming.h"

class SampleRandomized : public Streaming {
public:
  SampleRandomized(Network *g);
  ~SampleRandomized();
  int select_element(int j, uint e, int step);
};
