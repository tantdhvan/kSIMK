#include "mygraph.cpp"
#include <iostream>
#include <string>

using namespace std;
using namespace mygraph;

int main( int argc, char** argv ) {
   if (argc < 2) {
      cout << "Usage: " << argv[0] << " <input file> <output file>" << endl;
      cout << "Input is unweighted edge list\n";
      cout << "Graph is simplified, then written to output file (as undirected binary)" << endl;
      exit(1);
   }
   
  string inputGraphName ( argv[1] );
  simplifyGraph g;

  g.read_edge_list( inputGraphName );
  Logger logg;
  logg << "Removing isolates..." << endL;
  g.remove_isolates();
  logg << "Renumbering vertices..." << endL;
  g.renumber_vertices();

  cout << "Writing binary file..." << endl;
  string outputFName( argv[2] );
  g.write_bin( outputFName );

  //verify
  cout << "Checking binary file..." << endl;
  tinyGraph h;
  h.read_bin( outputFName );

  cout << g.n << ' ' << h.n << endl;

  return 0;
}
