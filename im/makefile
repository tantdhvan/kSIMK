CXX = g++
CXXFLAGS = -std=c++17 -I$HOME/local/boost/include
LDFLAGS = -L$HOME/local/boost/lib -lboost_graph
SRC = *.cpp
OUT = im

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(OUT) -fopenmp

clean:
	rm -f $(OUT)
