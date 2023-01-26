#include <iostream>
#include <vector>
#include <algorithm>
#include "extended_integer.h"
#include "tropical_matrix.h"
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

std::pair<std::vector<int>,std::vector<ExtendedInteger>> counterStrategy1(const MeanPayOffGame &game,const std::vector<int> &strategy,int iters)
{
    std::vector<ExtendedInteger> strategyCost(game.V);
    for(int i=0;i<game.V;i++)
        for(auto &[v, a]: game.adjacencyList[i]) if(v==strategy[i])
                strategyCost[i] = a;

    std::vector<ExtendedInteger> f(game.V,0),g(game.V,0);
    while(iters--)
    {
        for(int u=0;u<game.V;u++)
        {
            g[u]=inf_plus_t{};
            auto a=strategyCost[u];
            for(auto &[v, b]: game.adjacencyList[strategy[u]])
                g[u] = std::min(g[u],f[v] + a+b);
        }
        if(f==g)
            break;
        f=g;
    }
    std::vector<int> counter(game.V);
    for(int i=0;i<game.V;i++)
        counter[i]=std::min_element(game.adjacencyList[i].begin(),game.adjacencyList[i].end(),[&](auto &a,auto &b){
            return f[a.dest]+a.weight < f[b.dest]+b.weight;
        })->dest;
    return std::make_pair(counter,f);
}



std::pair<std::vector<int>,std::vector<ExtendedInteger>> counterStrategy2(const MeanPayOffGame &game,
                                                                                const std::vector<int> &strategy,
                                                                                std::uint64_t iters)
{
    // M is a tropical matrix that represents the valuation of going from i to j during a full turn.
    MaskedTropicalMatrix M(game.V,game.V,std::vector<std::vector<bool>>(game.V,std::vector<bool>(game.V,false)));
    // Strategy cost is the cost of the first player's strategy
    std::vector<ExtendedInteger> strategyCost(game.V);
    for(int i=0;i<game.V;i++)
        for(auto &[v, a]: game.adjacencyList[i]) if(v==strategy[i])
                strategyCost[i] = a;
    /*
     * We build the matrix M, using the adjacency list of the strategy
     * M is a square matrix, and it is a tropical matrix
     * M is a masked tropical matrix, as we should not consider the edges that are not in the strategy
     * */
    for(int i=0,t=strategy[i];i<game.V;i++,t=strategy[i]) for(auto [j, a]: game.adjacencyList[t])
    {
        M.matrix[i][j] = strategyCost[i] + a;
        M.mask[i][j] = true;
    }

    std::vector<ExtendedInteger> f(game.V,0);
    // We compute the power of M, using the fast exponentiation algorithm
    f=pow(M,iters)*f;
    std::vector<int> counter(game.V);
    for(int i=0;i<game.V;i++)
        counter[i]=std::min_element(game.adjacencyList[i].begin(),game.adjacencyList[i].end(),[&](auto &a,auto &b){
            return f[a.dest]+a.weight < f[b.dest]+b.weight;
        })->dest;
    return std::make_pair(counter,f);
}

std::pair<std::vector<int>,std::vector<ExtendedInteger>> counterStrategy3(const MeanPayOffGame &game,
                                                                                const std::vector<int> &strategy)
{
    // M is a tropical matrix that represents the valuation of going from i to j during a full turn.
    MaskedTropicalMatrix M(game.V,game.V,std::vector<std::vector<bool>>(game.V,std::vector<bool>(game.V,false)));
    // Strategy cost is the cost of the first player's strategy
    std::vector<ExtendedInteger> strategyCost(game.V);
    for(int i=0;i<game.V;i++)
        for(auto &[v, a]: game.adjacencyList[i]) if(v==strategy[i])
                strategyCost[i] = a;
    /*
     * We build the matrix M, using the adjacency list of the strategy
     * M is a square matrix, and it is a tropical matrix
     * M is a masked tropical matrix, as we should not consider the edges that are not in the strategy
     * */
    for(int i=0,t=strategy[i];i<game.V;i++,t=strategy[i]) for(auto [j, a]: game.adjacencyList[t])
    {
        M.matrix[i][j]=strategyCost[i]+a;
        M.mask[i][j]=true;
    }
    std::vector<ExtendedInteger> f(game.V,0);
    /*
     * An exponential algorithm for finding the fixed point of a matrix
     * */
    while(M*f != (M*M)*f)
        M = M * M;
    //Now, M*f is the fixed point, and it represents the valuation of the counter strategy
    f=M*f;
    //Now, we find the counter strategy
    std::vector<int> counter(game.V);
    for(int i=0;i<game.V;i++)
        counter[i]=std::min_element(game.adjacencyList[i].begin(),game.adjacencyList[i].end(),[&](auto &a,auto &b){
            return f[a.dest]+a.weight < f[b.dest]+b.weight;
        })->dest;

    return std::make_pair(counter,f);
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
    // Strategy of the first player
    std::vector<int> S(V);
    for(auto &s:S)
        std::cin >> s;
    /*
     * We compute the counter strategy of the first player
     * result.first is the counter strategy
     * result.second is the valuation of the counter strategy
     * */
    auto result = counterStrategy3(game,S);
    for(auto &r:result.second)
        std::cout << r <<' ';
    std::cout << std::endl;
    for(auto &r:result.first)
        std::cout << r <<' ';
    return 0;
}
