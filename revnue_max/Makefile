CPPFLAGS=-std=c++11 -Wall -O2

all: im revmax preproc

im: src/main.cpp src/mygraph.cpp src/algs.cpp
	g++ src/main.cpp -o im  ${CPPFLAGS} -DREVMAX
revmax: src/main.cpp src/mygraph.cpp src/algs.cpp
	g++ src/main.cpp -o revmax  ${CPPFLAGS} -DIM
debug: src/main.cpp src/mygraph.cpp src/algs.cpp
	g++ src/main.cpp -o maxcov_debug -std=c++11 -Wall -Og -g
revmax-debug: src/main.cpp src/mygraph.cpp src/algs.cpp
	g++-4.9 src/main.cpp -o revmax-debug -std=c++11 -Wall -Og -g  -DREVMAX
preproc: src/preprocess.cpp src/mygraph.cpp
	g++ src/preprocess.cpp -o preproc  ${CPPFLAGS}
nodes: src/nodes.cpp src/mygraph.cpp
	g++ src/nodes.cpp -o nodes  ${CPPFLAGS}
er: src/gen_er.cpp src/mygraph.cpp
	g++ -std=c++11 src/gen_er.cpp -o er
ba: src/gen_ba.cpp src/mygraph.cpp
	g++ -std=c++11 src/gen_ba.cpp -o ba
clean:
	rm -f preproc
	rm -f maxcov
	rm -f revmax
	rm -f exp/single-pass/*.txt*
	rm -f exp/single-pass/*.bin*
	rm -f exp/single-pass/*.pdf*
	rm -f exp/general/*.txt*
	rm -f exp/general/*.bin*
	rm -f exp/general/*.pdf*
reproduce: maxcov preproc
	cd exp/single-pass; bash reproduce.bash
	cd exp/general; bash reproduce.bash
