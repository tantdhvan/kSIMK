#ifndef ALGS_CPP
#define ALGS_CPP

#include "mygraph.cpp"
#include <set>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

using namespace std;
using namespace mygraph;

enum Algs
{
	SG,
	STR
};

uniform_real_distribution<double> unidist(1e-10, 1);

resultsHandler allResults;
vector<double> alpha;
vector<vector<double>> alpha2;
int numSimulations=4000;

void init_alpha(tinyGraph &g)
{
	alpha.assign(g.n, 0.0);
	alpha2.assign(g.k, vector<double>(g.n, 0.0));
	mt19937 gen(0); // same sequence each time

	for (node_id u = 0; u < g.n; ++u)
	{
		alpha[u] = unidist(gen);
	}
	for (int i = 0; i < g.k; i++)
	{
		for (node_id u = 0; u < g.n; ++u)
		{
			alpha2[i][u] = unidist(gen);
		}
	}	
}

struct Args
{
	Algs alg;
	string graphFileName;
	string outputFileName = "";
	size_t k = 2;
	tinyGraph g;
	double tElapsed;
	double wallTime;
	double B;
	Logger logg;
	bool steal = true;
	double epsi = 0.1;
	double delta = 0.1;
	double c = 1;
	size_t N = 1;
	size_t P = 10;
	bool plusplus = false;
	double tradeoff = 0.5;
	bool quiet = false;
};

class MyPair
{
public:
	node_id u;
	double gain; // may be negative

	MyPair() {}
	MyPair(node_id a,
		   double g)
	{
		u = a;
		gain = g;
	}

	MyPair(const MyPair &rhs)
	{
		u = rhs.u;
		gain = rhs.gain;
	}

	void operator=(const MyPair &rhs)
	{
		u = rhs.u;
		gain = rhs.gain;
	}
};

struct gainLT
{
	bool operator()(const MyPair &p1, const MyPair &p2)
	{
		return p1.gain < p2.gain;
	}
} gainLTobj;

struct revgainLT
{
	bool operator()(const MyPair &p1, const MyPair &p2)
	{
		return (p1.gain > p2.gain);
	}
} revgainLTobj;

vector<bool> emptySetVector;

#ifndef IM //hàm f(.) cho bài toán tối đa ảnh hưởng
size_t simulate(tinyGraph &g, const vector<kpoint>& S) {
	std::random_device rd;  // Sinh số ngẫu nhiên từ phần cứng
    std::mt19937 gen(rd()); // Mersenne Twister 19937
    std::uniform_real_distribution<double> dist(0.0, 1.0); // Phân phối đều từ 0 đến 1

    std::vector<bool> active(g.n, false);
    for (kpoint node : S) {
        active[node.first] = true;
    }

    std::queue<pair<kpoint,double>> queue;
    for (kpoint node : S) {
        queue.push(make_pair(node,1.0));
    }

    while (!queue.empty()) {
        kpoint u = queue.front().first;
		double prob = queue.front().second;
        queue.pop();
		int count_u=0;
		vector<tinyEdge> &neis = g.adjList[u.first].neis;
		for (size_t j = 0; j < neis.size(); ++j)
		{
			node_id v = neis[j].target;
			if (!active[v]) {
                double randNum = dist(gen);
                if (randNum <= neis[j].prob_influence[u.second]*prob) {
                    active[v] = true;
                    queue.push(make_pair(kpoint(v, u.second),neis[j].prob_influence[u.second]));
					count_u++;
                }
            }			
		}   
    }
    int count = 0;
    for (bool a : active) {
        if (a) count++;
    }
    return count;
}
size_t simulate(tinyGraph &g, const vector<kpoint>& S,kpoint e) {
    std::vector<bool> active(g.n, false);
    for (kpoint node : S) {
        active[node.first] = true;
    }
	if(active[e.first]) return 0;

	active[e.first] = true;

	std::queue<pair<kpoint,double>> queue;
    queue.push(make_pair(e,1.0));
	int count = 0;
    while (!queue.empty()) {
		kpoint u = queue.front().first;
		double prob = queue.front().second;
        queue.pop();
		vector<tinyEdge> &neis = g.adjList[u.first].neis;
		for (size_t j = 0; j < neis.size(); ++j)
		{
			node_id v = neis[j].target;
			if (!active[v]) {
                double randNum = unidist(gen);
                if (randNum <= neis[j].prob_influence[u.second]*prob) {
                    active[v] = true;
					count++;
                    queue.push(make_pair(kpoint(v, u.second),neis[j].prob_influence[u.second]));
                }
            }
		}
    }
    return count;
}
double compute_valSet(size_t &nEvals, tinyGraph &g, const vector<kpoint>& S) {
	nEvals++;
	if(S.size()==0) return 0;
    double total = 0;
	#pragma omp parallel for reduction(+ : total)
    for (int i = 0; i < numSimulations; i++) {
        total += simulate(g,S);
    }
    return total / numSimulations;
}
double marge(size_t &nEvals, tinyGraph &g, const vector<kpoint>& S,kpoint e) {
	nEvals++;
    double total = 0;
	#pragma omp parallel for reduction(+ : total)
    for (int i = 0; i < numSimulations; i++) {
        total += simulate(g,S,e);
    }
    return total / numSimulations;
}
#else // REVMAX
double compute_valSet(size_t &nEvals, tinyGraph &g, vector<bool> &set, vector<bool> &cov = emptySetVector)
{
	if (alpha.size() == 0)
	{
		init_alpha(g);
	}

	++nEvals;
	cov.assign(g.n, false);
	double val = 0;

	for (node_id u = 0; u < g.n; ++u)
	{
		vector<tinyEdge> &neis = g.adjList[u].neis;
		double valU = 0.0;
		for (size_t j = 0; j < neis.size(); ++j)
		{
			node_id v = neis[j].target;
			if (set[v])
			{
				valU += neis[j].weight;
			}
		}
		valU = pow(valU, alpha[u]);
		val += valU;
	}

	return val;
}

double compute_valSet(size_t &nEvals, tinyGraph &g, vector<kpoint> &sset)
{
	if (alpha.size() == 0)
	{
		init_alpha(g);
	}
	vector<vector<bool>> set(g.k, vector<bool>(g.n, false));
#pragma omp parallel for
	for (size_t i = 0; i < sset.size(); ++i)
	{
		set[sset[i].second][sset[i].first] = true;
	}

	++nEvals;

	double val = 0;
#pragma omp parallel for
	for (int p = 0; p < g.k; p++)
	{
#pragma omp parallel for
		for (node_id u = 0; u < g.n; ++u)
		{
			vector<tinyEdge> &neis = g.adjList[u].neis;
			double valU = 0.0;
			for (size_t j = 0; j < neis.size(); ++j)
			{
				node_id v = neis[j].target;
				if (set[p][v])
				{
					valU += neis[j].weight;
				}
			}
			valU = pow(valU, alpha2[p][u]);
#pragma omp critical
			{
				val += valU;
			}
		}
	}
	return val;
}
double marge(size_t &nEvals, tinyGraph &g, vector<kpoint> &sset, kpoint x)
{
	if (alpha.size() == 0)
	{
		init_alpha(g);
	}
	vector<vector<bool>> set(g.k, vector<bool>(g.n, false));
#pragma omp parallel for
	for (size_t i = 0; i < sset.size(); ++i)
	{
		set[sset[i].second][sset[i].first] = true;
	}
	if (set[x.second][x.first])
		return 0;

	vector<tinyEdge> &neis = g.adjList[x.first].neis;
	double gain = 0.0;
#pragma omp parallel for reduction(+ : gain)
	for (size_t j = 0; j < neis.size(); ++j)
	{
		node_id v = neis[j].target;
		vector<tinyEdge> &neisV = g.adjList[v].neis;
		double valV = 0.0;
		double valVwithX = 0.0;
		for (size_t k = 0; k < neisV.size(); ++k)
		{
			node_id w = neisV[k].target;
			if (w != x.first)
			{
				if (set[x.second][w])
				{
					valV += neisV[k].weight;
					valVwithX += neisV[k].weight;
				}
			}
			else
			{
				valVwithX += neisV[k].weight;
			}
		}

		if (valV == 0)
			gain += pow(valVwithX, alpha2[x.second][v]);
		else
			gain += pow(valVwithX, alpha2[x.second][v]) - pow(valV, alpha2[x.second][v]);
	}
	++nEvals;
	return gain;
}
double compute_valSet(size_t &nEvals, tinyGraph &g, vector<node_id> &sset)
{
	if (alpha.size() == 0)
	{
		init_alpha(g);
	}
	vector<bool> set(g.n, false);
	for (size_t i = 0; i < sset.size(); ++i)
	{
		set[sset[i]] = true;
	}

	++nEvals;

	double val = 0;

	for (node_id u = 0; u < g.n; ++u)
	{
		vector<tinyEdge> &neis = g.adjList[u].neis;
		double valU = 0.0;
		for (size_t j = 0; j < neis.size(); ++j)
		{
			node_id v = neis[j].target;
			if (set[v])
			{
				valU += neis[j].weight;
			}
		}
		valU = pow(valU, alpha[u]);
		val += valU;
	}

	return val;
}

double marge(size_t &nEvals, tinyGraph &g, node_id x, vector<bool> &set, vector<bool> &cov = emptySetVector)
{
	if (alpha.size() == 0)
	{
		init_alpha(g);
	}

	if (set[x])
		return 0;

	vector<tinyEdge> &neis = g.adjList[x].neis;
	double gain = 0.0;
	for (size_t j = 0; j < neis.size(); ++j)
	{
		node_id v = neis[j].target;
		vector<tinyEdge> &neisV = g.adjList[v].neis;
		double valV = 0.0;
		double valVwithX = 0.0;
		for (size_t k = 0; k < neisV.size(); ++k)
		{
			node_id w = neisV[k].target;
			if (w != x)
			{
				if (set[w])
				{
					valV += neisV[k].weight;
					valVwithX += neisV[k].weight;
				}
			}
			else
			{
				valVwithX += neisV[k].weight;
			}
		}

		if (valV == 0)
			gain += pow(valVwithX, alpha[v]);
		else
			gain += pow(valVwithX, alpha[v]) - pow(valV, alpha[v]);
	}
	++nEvals;
	return gain;
}

#endif

void reportResults(size_t nEvals, size_t obj, size_t maxMem = 0)
{
	allResults.add("obj", obj);
	allResults.add("nEvals", nEvals);
	allResults.add("mem", maxMem);
}

// Standard Greedy
class Sg
{
	size_t k;
	double B;
	double b;
	tinyGraph &g;
	size_t nEvals = 0;
	double eps;

public:
	Sg(Args &args) : g(args.g)
	{
		k = args.k;
		b = args.B;
		B = args.B * g.total_cost;
		eps = args.epsi;
	}

	void run()
	{
		init_alpha(g);
		cout<<"Start greedy"<<endl;
		nEvals = 0;
		vector<kpoint> seedsf;
		int no_nodes = g.n;
		vector<bool> v(no_nodes, false);
		double C_S[] = {0.0, 0.0, 0.0};
		double f=0;
		while (true)
		{
			int i_max = -1, e_max = -1;
			double max_f=0;
			double delta = 0;
			for (int e = 0; e < no_nodes; e++)
			{
				if (v[e] == true) continue;
				for (int i = 0; i < g.k; i++)
				{
					if (C_S[i] + g.adjList[e].wht > B) continue;
					seedsf.push_back(kpoint(e, i));
					double tmp_f = compute_valSet(nEvals, g, seedsf);
					seedsf.pop_back();
					double tmp_delta = (tmp_f-f)/ g.adjList[e].wht;
					if (tmp_delta > delta)
					{
						e_max = e;
						i_max = i;
						delta = tmp_delta;
						max_f = tmp_f;
					}
				}
			}
			cout<<"e_max: "<<e_max<<" i_max: "<<i_max<<" delta: "<<delta<<endl;
			if (i_max == -1 || e_max == -1) break;
			seedsf.push_back(kpoint(e_max, i_max));
			f=max_f;
			C_S[i_max] += g.adjList[e_max].wht;
			v[e_max] = true;
		}
		cout << "Greedy," << b << "," << B << "," << f << "," << nEvals;
	}
};

class Streaming
{
	size_t k;
	double B;
	double b;
	tinyGraph &g;
	size_t nEvals = 0;
	double eps;

public:
	Streaming(Args &args) : g(args.g)
	{
		k = args.k;
		b = args.B;
		B = args.B * g.total_cost;
		eps = args.epsi;
	}

	void run()
	{
		init_alpha(g);
		nEvals = 0;
		vector<kpoint> seedsf;
		int no_nodes = g.n;

		double alpha = 0.25;
		vector<myType> nguong;
		vector<int> nguongj;
		int e_max = -1, i_max = -1;
		double f_e_i_max = -1;
		size_t count = 0;
		for (int e = 0; e < no_nodes; e++)
		{
			int i_m = -1;
			double f_i = -1;
			for (int i = 0; i < g.k; i++)
			{
				vector<kpoint> tmp_seeds1;
				double tmp_f = marge(nEvals, g, tmp_seeds1,kpoint(e, i));

				if (tmp_f > f_i)
				{
					{
						i_m = i;
						f_i = tmp_f;
					}
				}
			}
			if (f_i > f_e_i_max)
			{
				e_max = e;
				i_max = i_m;
				f_e_i_max = f_i;
				int j = 0;
				while (true)
				{
					double tmp = pow((1 + eps), j);
					if (tmp > f_e_i_max * B * g.k) break;
					if (tmp >= f_e_i_max)
					{
						auto it = std::find(nguongj.begin(), nguongj.end(), j);
						if (it == nguongj.end())
						{
							myType tmp_ng;
							tmp_ng.nguong = j;
							tmp_ng.nguongd = tmp;
							tmp_ng.cost = {0.0, 0.0, 0.0};
							tmp_ng.check = vector<bool>(no_nodes, false);
							nguong.push_back(tmp_ng);
							nguongj.push_back(j);
						}
					}
					j++;
				}
			}

			for (int t = 0; t < nguong.size(); t++)
			{
				myType &nguongt = nguong[t];
				if (nguongt.nguongd < f_e_i_max) continue;
				double delta=0;
				int iv_max = -1;
				for (int i = 0; i < g.k; i++)
				{
					if (nguongt.cost[i] + g.adjList[e].wht > B) continue;
					double tmp_delta = marge(nEvals, g, nguongt.s, kpoint(e, i));
					if (tmp_delta > delta)
					{
						{
							iv_max = i;
							delta = tmp_delta;
						}
					}
				}
				if (iv_max != -1)
				{
					if (delta >= g.adjList[e].wht * alpha * nguongt.nguongd / B)
					{
						{
							nguongt.s.push_back(kpoint(e, iv_max));
							nguongt.cost[iv_max] += g.adjList[e].wht;
							nguongt.check[e] = true;
						}
					}
				}
			}
		}
		for (int e = 0; e < no_nodes; e++)
		{
			for (int t = 0; t < nguong.size(); t++)
			{
				myType &nguongt = nguong[t];
				if (nguongt.nguongd < f_e_i_max) continue;
				if (nguongt.check[e] == true) continue;
				int imax = -1;
				double delta = 0;
				for (int i = 0; i < g.k; i++)
				{
					if (nguongt.cost[i] + g.adjList[e].wht > B) continue;
					double tmp_delta = marge(nEvals, g, nguongt.s, kpoint(e,i));
					if (tmp_delta> delta)
					{
						#pragma omp critical
						{
							imax = i;
							delta = tmp_delta;
						}
					}
				}
				if (imax != -1)
				{
					{
						nguongt.s.push_back(kpoint(e, imax));
						nguongt.cost[imax] += g.adjList[e].wht;
						nguongt.check[e] = true;
					}
				}
			}
		}
		double result = 0;
		int k_max = -1;
		for (int i = 0; i < nguong.size(); i++)
		{
			if(nguong[i].nguongd < f_e_i_max) continue;
			double f_tmp=compute_valSet(nEvals, g, nguong[i].s);
			if (f_tmp > result)
			{
				{
					result = f_tmp;
				}
			}
		}
		cout << "Streaming," << b << "," << B << "," << result << "," << nEvals;
	}
};
#endif
