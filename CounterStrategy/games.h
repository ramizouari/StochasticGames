//
// Created by ramizouari on 24/01/23.
//

#ifndef COUNTERSTRATEGY_GAMES_H
#define COUNTERSTRATEGY_GAMES_H


#include <vector>

struct DirectedEdge
{
    int dest;
    int weight;
    DirectedEdge(){} // default constructor
    DirectedEdge(int dest, int weight): dest(dest), weight(weight)
    {}
};

struct LabeledGraph
{
    int V,E;
    std::vector<std::vector<DirectedEdge>> adjacencyList,reverseList;
    LabeledGraph(int V, int E): V(V), E(E), adjacencyList(V),reverseList(V)
    {}
    virtual  ~LabeledGraph()= default;

    void addEdge(int u, int v, int w)
    {
        adjacencyList[u].emplace_back(v, w);
        reverseList[v].emplace_back(u,w);
    }

};

struct MeanPayOffGame : public LabeledGraph
{
    int v;
    MeanPayOffGame(int V, int E,int v): LabeledGraph(V,E),v(v)
    {}
    virtual ~MeanPayOffGame()= default;
};


#endif //COUNTERSTRATEGY_GAMES_H
