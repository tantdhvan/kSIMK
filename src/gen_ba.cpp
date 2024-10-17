
#include "mygraph.cpp"

#include <iostream>
#include <string>

using namespace std;
using namespace mygraph;

int main(int argc, char** argv) {
   if (argc == 1) {
      cout << "Usage: " << argv[0] << " <n> <m = m_0> <out-fname>" << endl;
      cout << "BA graph is undirected and written in binary format." << endl;
      return 1;
   }

   string arg1( argv[1] );
   string arg2( argv[2] );
   unsigned n = stoi( arg1 );
   unsigned m = stoi( arg2 );
   
   tinyGraph g;
   g.barabasi_albert( n, m, m );

   //g.write_bin( argv[3] );
   g.write_edge_list( argv[3] );

   return 0;
}
