#include <iostream>
#include "MeanPayoffGame.h"
#include "MinMaxSystem.h"
#include "MPGReader.h"


int main(int argc, char *argv[]) {
    int n=0,m=0;
    std::vector<std::tuple<int,int,int>> edges;

    if(argc>1)
    {

        MPGFileReader<Implementation::Matrix::MeanPayoffGame<integer>> reader(argv[1]);
        MPG auto game=reader.read();
        auto S=mean_payoff_game_to_min_max_system(game);
        Implementation::Vector::MaxAtomSystemSolver<integer> solver;
        auto [S0,S1]= optimal_strategy_pair(game,solver);
        for(int i=0;i<game.count_nodes();i++)
            std::cout << i << ':' << S0[i] << ',';
        std::cout << std::endl;
        for(int i=0;i<game.count_nodes();i++)
            std::cout << i << ':' << S1[i] << ',';
    }
    else
    {
        std::cin >>n>>m;
        for(int i=0;i<m;i++)
        {
            int u,v,w;
            std::cin >>u>>v >> w;
            edges.emplace_back(u,v,w);
        }
        Implementation::Matrix::MeanPayoffGame<integer> game(n);
        for(auto [u,v,w]:edges)
            game.add_edge(u,v,w);
        auto S=mean_payoff_game_to_min_max_system(game);
        Implementation::Vector::MaxAtomSystemSolver<integer> solver;
        auto [S0,S1]= optimal_strategy_pair(game,solver);
        std::cout << '{';
        for(int i=0;i<game.count_nodes();i++)
            std::cout << i << ':' << S0[i] << (i==game.count_nodes()-1?'}':',');
        std::cout << std::endl;
        std::cout << '{';
        for(int i=0;i<game.count_nodes();i++)
            std::cout << i << ':' << S1[i] << (i==game.count_nodes()-1?'}':',');
    }



    return 0;
}
