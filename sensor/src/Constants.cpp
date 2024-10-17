#include "Constants.h"
#include <vector>

Constants::Constants() {}

Constants::~Constants() {}

ALG Constants::ALGORITHM = Dstream;
DATATYPE Constants::DATA = Social;
int Constants::NUM_THREAD = 70;
int Constants::K = 3;
double Constants::BUDGET = 10;

// for social
double Constants::ALPHA = 0.5;
double Constants::BETA = 0.5;
double Constants::DELTA = 0.5;
double Constants::EPS_TAG = 0.5;
double Constants::RHO = 0.5;
double Constants::EPS = 0.1;
int Constants::NO_DENOISE_STEPS = 2;