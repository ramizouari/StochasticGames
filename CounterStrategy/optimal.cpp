//
// Created by ramizouari on 26/01/23.
//
#include <iostream>
#include <vector>
#include <array>
#include <bitset>
#include "extended_integer.h"
#include "games.h"

using integer = std::int64_t;

/**
 * @brief The fast exponentiation algorithm
 * @details Computes u^n in O(log(n)) arithmetic operations
 */
template<typename M,MonoidOperation O=Multiplication<M>>
M pow(const M& u,std::uint64_t n, O op =Multiplication<M>{})
{
    if(n==0)
        return O::identity;
    if(n==1)
        return u;
    return op(pow(op(u,u),n/2,op),pow(u,n%2,op));
}

struct Strategy
{
    std::vector<int> strategy;
    std::vector<ExtendedInteger> cost;
};

std::array<Strategy,2> optimalStrategy1(const MeanPayOffGame &game,int iters)
{

    std::vector<ExtendedInteger> g(game.V,0),h(game.V,0),tmp_g(game.V,0),tmp_h(game.V,0);
    while(iters--)
    {
        for(int u=0;u<game.V;u++)
        {
            tmp_g[u]=inf_p;
            tmp_h[u]=inf_m;
            for(auto &[v, a]: game.adjacencyList[u])
            {
                ExtendedInteger s=inf_m;
                for (auto &[w, b]: game.adjacencyList[v])
                    s = std::max(s, g[w] + a + b);
                tmp_g[u] = std::min(tmp_g[u],s);
                s=inf_p;
                for (auto &[w, b]: game.adjacencyList[v])
                    s = std::min(s, h[w] + a + b);
                tmp_h[u]=std::max(tmp_h[u],s);
            }
        }
        if(g==tmp_g && h==tmp_h)
            break;
        g=tmp_g;
        h=tmp_h;
    }
    std::array<Strategy,2> result;
    result[0].strategy.resize(game.V);
    result[1].strategy.resize(game.V);
    for(int u=0;u<game.V;u++)
    {
        ExtendedInteger t=inf_p;
        for(auto &[v, a]: game.adjacencyList[u])
        {
            ExtendedInteger s=inf_m;
            for (auto &[w, b]: game.adjacencyList[v])
                s = std::max(s, g[w] + a + b);
            if(s < t)
            {
                result[0].strategy[u]=v;
                t=s;
            }
        }

        t=inf_m;
        for(auto &[v, a]: game.adjacencyList[u])
        {
            ExtendedInteger s=inf_p;
            for (auto &[w, b]: game.adjacencyList[v])
                s = std::min(s, h[w] + a + b);
            if(s > t)
            {
                result[1].strategy[u]=v;
                t=s;
            }
        }
    }
    result[1].cost=std::move(h);
    result[0].cost=std::move(g);
    return result;
}

std::array<Strategy,2> optimalStrategy2(const MeanPayOffGame &game,int iters)
{

    std::vector<ExtendedInteger> g(game.V,0),h(game.V,0),tmp(game.V,0);
    std::bitset<2> convergence;
    for(int iter=1;iter <= 2*iters && !convergence.all();iter++)
    {
        for(int u=0;u<game.V;u++)
        {
            if(iter%2==0)
            {
                tmp[u]=inf_p;
                for(auto &[v, a]: game.adjacencyList[u])
                    tmp[u] = std::min(tmp[u],h[v] + a);
            }
            else
            {
                tmp[u]=inf_m;
                for(auto &[v, a]: game.adjacencyList[u])
                    tmp[u] = std::max(tmp[u],g[v] + a);
            }
        }
        if(iter%2==0)
        {
            if(g==tmp)
                convergence[0]=true;
            g = tmp;
        }
        else
        {
            if(h==tmp)
                convergence[1]=true;
            h = tmp;
        }
    }
    std::array<Strategy,2> result;
    result[0].strategy.resize(game.V);
    result[1].strategy.resize(game.V);
    for(int u=0;u<game.V;u++)
    {
        ExtendedInteger t=inf_p;
        for(auto &[v, a]: game.adjacencyList[u])
        {
            ExtendedInteger s=inf_m;
            for (auto &[w, b]: game.adjacencyList[v])
                s = std::max(s, g[w] + a + b);
            if(s < t)
            {
                result[0].strategy[u]=v;
                t=s;
            }
        }

        t=inf_m;
        for(auto &[v, a]: game.adjacencyList[u])
        {
            ExtendedInteger s=inf_p;
            for (auto &[w, b]: game.adjacencyList[v])
                s = std::min(s, h[w] + a + b);
            if(s > t)
            {
                result[1].strategy[u]=v;
                t=s;
            }
        }
    }
    result[1].cost=std::move(h);
    result[0].cost=std::move(g);
    return result;
}



int main() {
    // Graph parameters
    int V,E;
    std::cin >> V >> E;
    MeanPayOffGame game(V,E,0);
    // Edges of the graph
    for(int i=0;i<E;i++)
    {
        int u,v,w;
        std::cin >> u >> v >> w;
        game.addEdge(u,v,w);
    }
    // Number of iterations
    int iters;
    std::cin >> iters;
    // Compute the optimal strategy
    auto strategyPair=optimalStrategy2(game,iters);
    for(int i=0;i<strategyPair.size();i++)
    {
        std::cout << "Player " << i << " strategy: ";
        for(int j=0;j<strategyPair[i].strategy.size();j++)
            std::cout << strategyPair[i].strategy[j] << " ";
        std::cout << std::endl;
        std::cout << "Player " << i << " cost: ";
        for(int j=0;j<strategyPair[i].cost.size();j++)
            std::cout << strategyPair[i].cost[j] << " ";
        std::cout << std::endl;
    }
    return 0;
}
