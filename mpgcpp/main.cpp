#include <iostream>
#include "MeanPayoffGame.h"
#include "MinMaxSystem.h"
#include <fstream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

int main() {
    int n,m;
    std::cin >>n>>m;
    Implementation::HashMap::MeanPayoffGame<int> game(n);
    for(int i=0;i<m;i++)
    {
        int u,v,w;
        std::cin >>u>>v >> w;
        game.add_edge(u,v,w);
    }
    auto S=mean_payoff_game_to_min_max_system(game);
    Implementation::TreeMap::MaxAtomSystemSolver<int> solver;
    auto [S1,S2]= optimal_strategy_pair(game,solver);
    for(int i=0;i<game.count_nodes();i++)
    {
        std::cout <<i << ' ' <<S1[i]<<" "<<S2[i]<<std::endl;
    }
    return 0;
}
