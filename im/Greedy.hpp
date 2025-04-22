#ifndef GREEDY_HPP
#define GREEDY_HPP

#include "myGraph.hpp"
#include <vector>
#include <utility>
using namespace std;
class Greedy
{
public:
    Greedy(myGraph &g, double b) : graph(g), b(b) {}

    std::vector<std::pair<int, int>> findOptimalSeedSet(int max_simulations = 1000)
    {

        std::vector<std::pair<int, int>> seedSet;
        std::vector<bool> selected(graph.num_vertices(), false);
        std::vector<double> topicWeights(3, 0.0);
        double currentInfluence = 0.0;
        while (true)
        {
            double maxInfluence = -1;
            double maxDeltaInfluence = -1;
            int bestNode = -1;
            int bestTopic = -1;

            for (int node = 0; node < graph.num_vertices(); ++node)
            {
                if (!selected[node])
                {
                    for (int topic = 0; topic < 3; ++topic)
                    {
                        if (topicWeights[topic] + graph.getGraph()[node].weight <= b)
                        {
                            std::vector<std::pair<int, int>> seeds = seedSet;
                            seeds.push_back({node, topic});
                            double influence = graph.computeInfluence(seeds);
                            double deltaInfluence = (influence - currentInfluence)/graph.getGraph()[node].weight;
                            if (deltaInfluence > maxDeltaInfluence)
                            {
                                maxDeltaInfluence = deltaInfluence;
                                bestNode = node;
                                bestTopic = topic;
                                maxInfluence=influence;
                            }
                        }
                    }
                }
            }

            if (bestNode == -1)
            {
                break;
            }

            selected[bestNode] = true;
            seedSet.push_back({bestNode, bestTopic});
            topicWeights[bestTopic] += graph.getGraph()[bestNode].weight;
            currentInfluence = maxInfluence;
            cout<<"maxInfluence:"<<maxInfluence<<" best weight:"<<graph.getGraph()[bestNode].weight<<endl;
            cout<<topicWeights[0]<<" "<<topicWeights[1]<<" "<<topicWeights[2]<<endl;
        }
        std::cout<<topicWeights[0]<<" "<<topicWeights[1]<<" "<<topicWeights[2]<<std::endl;
        return seedSet;
    }

private:
    myGraph &graph;
    double b;
};

#endif // GREEDY_HPP
