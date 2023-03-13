#include <iostream>
#include "MeanPayoffGame.h"
#include "MinMaxSystem.h"
#include "MPGReader.h"
#include <fstream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>

int main(int argc, char *argv[]) {
    int n=0,m=0;
    std::vector<std::tuple<int,int,int>> edges;
    if(argc>1)
    {
        /*std::ifstream file(argv[1], std::ios_base::in | std::ios_base::binary);
        boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
        in.push(boost::iostreams::gzip_decompressor());
        in.push(file);
        std::istream input(&in);
        int u,v,w;
        while(input >>u>>v >> w)
        {
            edges.emplace_back(u, v, w);
            m++;
            n=std::max({n,u+1,v+1});
        }*/
        MPGFileReader<Implementation::Matrix::MeanPayoffGame<int>> reader(argv[1]);
        MPG auto game=reader.read();
        n=game.count_nodes();
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
    }

    Implementation::Matrix::MeanPayoffGame<integer> game(n);
    for(auto [u,v,w]:edges)
        game.add_edge(u,v,w);

    auto S=mean_payoff_game_to_min_max_system(game);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto [S1,S2]= optimal_strategy_pair(game,solver);
    for(int i=0;i<game.count_nodes();i++)
    {
        std::cout <<i << ' ' <<S1[i]<<" "<<S2[i]<<std::endl;
    }
    return 0;
}
