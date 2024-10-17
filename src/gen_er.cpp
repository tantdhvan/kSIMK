
#include "mygraph.cpp"

#include <iostream>
#include <string>

using namespace std;
using namespace mygraph;

int main(int argc, char** argv) {
   if (argc == 1) {
      cout << "Usage: " << argv[0] << " <er-n> <er-p> <out-fname>" << endl;
      cout << "ER graph is undirected and written in binary format." << endl;
      return 1;
   }

   string arg1( argv[1] );
   string arg2( argv[2] );
   unsigned n = stoi( arg1 );
   double p = stod( arg2 );
   
   tinyGraph g;
   g.erdos_renyi_undirected( n, p );

   //g.write_bin( argv[3] );
   g.write_edge_list( argv[3] );

   return 0;
}
