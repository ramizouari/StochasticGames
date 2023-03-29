//
// Created by ramizouari on 27/03/23.
//

#include "game/MeanPayoffGame.h"
#include "csp/MaxAtomSystem.h"
#include "csp/MaxAtomSolver.h"
#include "csp/solver/ParallelMaxAtomSolver.h"
#include <iostream>

using Solver=Implementation::Parallel::HashMap::MaxAtomSystemSolver<integer>;
int main()
{

    int n,m;
    std::cin >> n >> m;
    Implementation::HashMap::MeanPayoffGame<integer> game(n);
    while(m--)
    {
        int u,v,w;
        std::cin >> u >> v >> w;
        game.add_edge(u,v,w);
    }
    Solver solver;
    auto [p0,p1]= optimal_strategy_pair(game,solver);
    auto [mp0,mp1]= mean_payoffs(game,{p0,p1});
    auto [w0,w1]= winners(game,{p0,p1});
    std::cout << "Mean payoffs for player 0: ";
    for(auto x:mp0)
        std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "Mean payoffs for player 1: ";
    for(auto x:mp1)
        std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "Winners for player 0: ";
    for(auto x:w0)
        std::cout << x << " ";
    std::cout << std::endl;
    std::cout << "Winners for player 1: ";
    for(auto x:w1)
        std::cout << x << " ";
}