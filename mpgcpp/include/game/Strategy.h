//
// Created by ramizouari on 13/04/23.
//

#ifndef MPGCPP_STRATEGY_H
#define MPGCPP_STRATEGY_H

#include <vector>
#include <random>
#include "MeanPayoffGame.h"

template<typename R>
class Strategy
{
public:
    virtual int get_action(const MeanPayoffGameBase<R> &game,const std::vector<int> &plays)=0;
    virtual ~Strategy()=default;
    int operator()(const MeanPayoffGameBase<R> &game,const std::vector<int>& plays)
    {
        return get_action(game,plays);
    }
};

template<typename R>
class MemorylessStrategy : public Strategy<R>
{
public:
    virtual int get_action(const MeanPayoffGameBase<R> &game,int state)=0;
    int get_action(const MeanPayoffGameBase<R> &game,const std::vector<int> &plays) override
    {
        return get_action(game,plays.back());
    }
    int operator()(const MeanPayoffGameBase<R> &game,int state)
    {
        return get_action(game,state);
    }

};

template<typename R>
class DeterministicMemorylessStrategy : public MemorylessStrategy<R>
{
    const std::vector<int> actions;
public:
    DeterministicMemorylessStrategy(const std::vector<int> &actions): actions(actions)
    {
    }
    DeterministicMemorylessStrategy(std::vector<int> &&actions): actions(std::move(actions))
    {
    }
    int get_action(int state) override
    {
        return actions[state];
    }
};

template<typename R>
class FractionalMemorylessStrategy : public MemorylessStrategy<R>
{
    using probabilistic_action=std::vector<std::pair<int,R>>;
    std::vector<probabilistic_action> actions;
    std::mt19937_64 generator;
    std::vector<std::discrete_distribution<int>> distributions;
public:
    FractionalMemorylessStrategy(const std::vector<probabilistic_action> &actions,size_t seed=0): actions(actions),generator(seed)
    {
        for(auto &action:actions)
        {
            std::vector<R> probabilities;
            for(auto &p:action)
                probabilities.push_back(p);
            distributions.emplace_back(probabilities.begin(),probabilities.end());
        }
    }
    FractionalMemorylessStrategy(std::vector<probabilistic_action> &&actions,size_t seed=0): actions(std::move(actions)),generator(seed)
    {
        for(auto &action:this->actions)
        {
            std::vector<R> probabilities;
            for(auto &p:action)
                probabilities.push_back(p);
            distributions.emplace_back(probabilities.begin(),probabilities.end());
        }
    }
    int get_action(int state) override
    {
        return actions[state][distributions[state](generator)];
    }

    const std::vector<probabilistic_action> &get_actions() const
    {
        return actions;
    }
};

template<typename R>
class RandomMemorylessStrategy : public MemorylessStrategy<R>
{
    const std::vector<std::vector<R>> actions;
    std::mt19937_64 generator;
    std::vector<std::discrete_distribution<int>> distributions;
public:
    RandomMemorylessStrategy(const std::vector<std::vector<R>> &actions,size_t seed=0): actions(actions),generator(seed)
    {
        for(auto &action:actions)
        {
            std::vector<R> probabilities;
            for(auto &p:action)
                probabilities.push_back(p);
            distributions.emplace_back(probabilities.begin(),probabilities.end());
        }
    }
    RandomMemorylessStrategy(std::vector<std::vector<R>> &&actions,size_t seed=0): actions(std::move(actions)),generator(seed)
    {
        for(auto &action:this->actions)
        {
            std::vector<R> probabilities;
            for(auto &p:action)
                probabilities.push_back(p);
            distributions.emplace_back(probabilities.begin(),probabilities.end());
        }
    }
    int get_action(int state) override
    {
        return distributions[state](generator);
    }
};

template<typename R,typename Real=double>
Real mean_payoff(const MeanPayoffGameBase<R> &game,int starting_state, const Strategy<R> &S0, const Strategy<R> &S1, int max_plays)
{
    std::vector<int> plays;
    plays.reserve(max_plays);
    int state=starting_state;
    std::map<std::pair<int,int>,int> payoff_counts;
    plays.push_back(state);
    for(int i=0;i<max_plays;i++)
    {
        int u=S0(game,plays);
        payoff_counts[{state,u}]++;
        plays.push_back(u);
        int v=S1(game,plays);
        payoff_counts[{u,v}]++;
        plays.push_back(v);
    }
    Real sum=0;
    for(auto [u,v,w]:game.get_edges())
        sum+=w*payoff_counts[{u,v}];
    return sum/max_plays;
}

template<typename R,typename Real=double>
Real mean_payoff(const MeanPayoffGameBase<R> &game,int starting_state, const DeterministicMemorylessStrategy<R> &S0,
              const DeterministicMemorylessStrategy<R> &S1)
{
    return mean_payoffs<R,Real>(game,S0,S1)[Bipartite::Player::PLAYER_0][starting_state];
}


#endif //MPGCPP_STRATEGY_H
