#include "mygraph.cpp"
#include "algs.cpp"

#include <iostream>
#include <string>
#include <unistd.h>
#include <chrono>

using namespace mygraph;
using namespace std;

void print_help()
{
   cout << "Options: " << endl;
   cout << "-g <graph filename in binary format>" << endl
        << "-k <cardinality constraint>" << endl
        << "-G [run Greedy]" << endl       
        << "-Q [run Str]" << endl       
        << "-o <outputFileName>" << endl
        << "-N <repetitions>" << endl
        << "-e <accuracy parameter epsilon (default 0.1)>" << endl
        << "-q [quiet mode]" << endl;
}

void parseArgs(int argc, char **argv, Args &arg)
{
   int c;
   extern char *optarg;

   if (argc == 1)
   {
      print_help();
      exit(2);
   }

   string sarg;

   while ((c = getopt(argc, argv, ":g:k:b:GQKPTBLSRN:Co:e:c:pt:qd:")) != -1)
   {
      switch (c)
      {
         case 'c':
            sarg.assign(optarg);
            arg.c = stoi(sarg);
            break;
         case 't':
            sarg.assign(optarg);
            arg.tradeoff = stod(sarg);
            break;
         case 'e':
            sarg.assign(optarg);
            arg.epsi = stod(sarg);
            break;
         case 'd':
            sarg.assign(optarg);
            arg.delta = stod(sarg);
            break;
         case 'o':
            sarg.assign(optarg);
            arg.outputFileName = sarg;
            break;
         case 'g':
            arg.graphFileName.assign(optarg);
            break;
         case 'k':
            sarg.assign(optarg);
            arg.k = stoi(sarg);
            break;
         case 'b':
            sarg.assign(optarg);
            arg.B = stod(sarg);
            break;
         case 'N':
            sarg.assign(optarg);
            arg.N = stoi(sarg);
            break;
         case 'G':
            arg.alg = SG;
            break;
         case 'Q':
            arg.alg = STR;
            break;
         case 'p':
            arg.plusplus = true;
            break;
         case 'q':
            arg.quiet = true;
            break;
         case '?':
            print_help();
            exit(2);
            break;
      }
   }

   if (arg.quiet)
   {
      arg.logg.enabled = false;
      arg.g.logg.enabled = false;
   }
}

void readGraph(Args &args)
{
   args.logg(INFO, "Reading graph from file: " + args.graphFileName + "...");
   args.g.read_bin(args.graphFileName);
   args.logg(INFO, "Input finished.");
   args.logg << "Nodes: " << args.g.n << ", edges: " << args.g.m << endL;
}

void runAlg(Args &args)
{
   size_t N = args.N;
   allResults.init("obj");
   allResults.init("nEvals");
   allResults.init("k");
   allResults.init("n");
   allResults.init("mem");
   allResults.add("k", args.k);
   allResults.add("n", args.g.n);

   for (size_t i = 0; i < N; ++i)
   {
      args.g.logg << "runAlg: Repetition = " << i << endL;
      clock_t t_start = clock();
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      switch (args.alg)
      {
         case SG:
         {
            args.logg(INFO, "Starting Greedy...");
            Sg sg(args);
            sg.run();
         }
         break;
         case STR:
         {
            args.logg(INFO, "Starting this work...");
            Streaming streaming(args);
            streaming.run();
         }
         break;
      }

      args.tElapsed = elapsedTime(t_start);
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      size_t WallTimeMillis = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
      args.wallTime = WallTimeMillis / 1000.0;

      cout<<","<<args.epsi<<","<<args.wallTime<<endl;
   }
}

void outputResults(Args &args)
{

   if (args.outputFileName != "")
   {
      args.g.logg << "Writing output to file: " << args.outputFileName << endL;
      ofstream of(args.outputFileName.c_str(), ofstream::out | ofstream::app);
      allResults.print(of);
   }
   else
   {
      allResults.print("obj");
      cout << ' ';
      allResults.print("nEvals");
   }
}

int main(int argc, char **argv)
{
   Args args;
   parseArgs(argc, argv, args);
   readGraph(args);
   runAlg(args);
}
