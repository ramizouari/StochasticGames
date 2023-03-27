//
// Created by ramizouari on 12/03/23.
//

#ifndef MPGCPP_MEANPAYOFFGAME_H
#define MPGCPP_MEANPAYOFFGAME_H
#include "graph/graph.h"
#include "csp/MinMaxSystem.h"
#include <optional>
#include <memory>
#include <numeric>

template<typename R>
struct DirectedEdge
{
    R weight;
    int source;
    int target;
public:
    DirectedEdge(int source,int target,R weight):source(source),target(target),weight(weight){}
};

template<typename R>
class MeanPayoffGameBase
{
    std::unique_ptr<MeanPayoffGameBase<R>> dual_game;
    std::vector<DirectedEdge<R>> edges;
protected:
    virtual void add_edge_impl(int source, int target, R weight)=0;
    inline static constexpr class NullDual_t{} NullDual{};
    explicit MeanPayoffGameBase(NullDual_t){}
    virtual void set_dual(MeanPayoffGameBase<R> *dual)
    {
        dual_game.reset(dual);
    }
public:
    using weights_type=R;
    MeanPayoffGameBase();
    MeanPayoffGameBase(MeanPayoffGameBase &&other) noexcept;
    MeanPayoffGameBase(const MeanPayoffGameBase &other);

    void add_edge(int source,int target,R weight)
    {
        add_edge_impl(source,target,weight);
        edges.emplace_back(source,target,weight);
        dual()->edges.emplace_back(source,target,-weight);
    }

    void add_edge(DirectedEdge<R> e)
    {
        add_edge_impl(e.source, e.target, e.weight);
        edges.push_back(e);
        e.weight=-e.weight;
        dual()->edges.push_back(e);

    }
    void add_edges(std::vector<DirectedEdge<R>> edges)
    {
        for(auto e:edges)
            add_edge(e);
    }

    virtual std::optional<R> get_weight(int source,int target) const=0;
    std::optional<R> get_weight(DirectedEdge<R> e) const
    {
        return get_weight(e.source,e.target);
    }

    virtual void set_weight(int source,int target,R weight)=0;

    void set_weight(DirectedEdge<R> e)
    {
        set_weight(e.source,e.target,e.weight);
    }

    virtual MeanPayoffGameBase* dual() const
    {
        return dual_game.get();
    }

    [[nodiscard]] virtual size_t count_nodes() const=0;

    virtual const std::vector<DirectedEdge<R>>& get_edges() const
    {
        return edges;
    }

    virtual ~MeanPayoffGameBase()=default;

};

template<typename R>
class DualMeanPayoffGame: public MeanPayoffGameBase<R>
{
    MeanPayoffGameBase<R> *original;
    //friend class MeanPayoffGameBase<R>;
protected:
    void add_edge_impl(int source, int target, R weight) override
    {
        original->add_edge(target, source, -weight);
    }

    void set_dual(MeanPayoffGameBase<R> *dual) override
    {
        original=dual;
    }

public:
    using weights_type=R;
    DualMeanPayoffGame(MeanPayoffGameBase<R> *original): MeanPayoffGameBase<R>(MeanPayoffGameBase<R>::NullDual),original(original){}

    std::optional<R> get_weight(int source,int target) const override
    {
        if(auto w=original->get_weight(source,target))
            return -*w;
        else
            return std::nullopt;
    }
    void set_weight(int source,int target,R weight) override
    {
        original->set_weight(target,source,-weight);
    }
    MeanPayoffGameBase<R>* dual() const override
    {
        return original;
    }

    [[nodiscard]] size_t count_nodes() const override
    {
        return original->count_nodes();
    }
    friend class MeanPayoffGameBase<R>;
};

template<typename R>
MeanPayoffGameBase<R>::MeanPayoffGameBase()
{
    dual_game=std::make_unique<DualMeanPayoffGame<R>>(this);
}

template<typename R>
MeanPayoffGameBase<R>::MeanPayoffGameBase(MeanPayoffGameBase && other) noexcept: edges(std::move(other.edges)),dual_game(std::move(other.dual_game))
{
    dual_game->set_dual(this);
}
template<typename R>
MeanPayoffGameBase<R>::MeanPayoffGameBase(const MeanPayoffGameBase &other):edges(other.edges)
{
    dual_game=std::make_unique<DualMeanPayoffGame<R>>(this);
    for(auto e:edges)
        dual_game->edges.emplace_back(e.source,e.target,-e.weight);
}

namespace Implementation
{
    namespace Vector
    {
        template<typename R>
        class MeanPayoffGame: public MeanPayoffGameBase<R>
        {
            int n;
            std::vector<std::vector<DirectedEdge<R>>> adjacency_list;
            bool is_original=true;
        protected:
            void add_edge_impl(int source, int target, R weight) override
            {
                adjacency_list[source].emplace_back(source,target,weight);
            }
        public:
            using weights_type=R;
            explicit MeanPayoffGame(int n):n(n),adjacency_list(n){}
            virtual ~MeanPayoffGame()
            {

            }

            std::optional<R> get_weight(int source,int target) const override
            {
                for(auto e:adjacency_list[source])
                    if(e.target==target)
                        return e.weight;
                return std::nullopt;
            }

            void set_weight(int source,int target,R weight) override
            {
                for(auto &e:adjacency_list[source]) if(e.target==target)
                {
                    e.weight=weight;
                    return;
                }
                throw std::runtime_error("Edge not found");
            }

            [[nodiscard]] size_t count_nodes() const override
            {
                return n;
            }
        };
    }

    namespace Matrix {
        template<typename R>
        class MeanPayoffGame : public MeanPayoffGameBase<R> {
            int n;
            std::vector<std::vector<std::optional<R>>> adjacency_matrix;
            bool is_original = true;
        protected:
            void add_edge_impl(int source, int target, R weight) override {
                adjacency_matrix[source][target] = weight;
            }

        public:
            using weights_type=R;
            explicit MeanPayoffGame(int n) : n(n), adjacency_matrix(n, std::vector<std::optional<R>>(n,std::nullopt)) {}

            std::optional<R> get_weight(int source, int target) const override
            {
                return adjacency_matrix[source][target];
            }

            void set_weight(int source, int target, R weight) override {
                adjacency_matrix[source][target] = weight;
            }

            [[nodiscard]] size_t count_nodes() const override
            {
                return n;
            }
        };
    }
    namespace HashMap
    {
        template<typename R>
        class MeanPayoffGame: public MeanPayoffGameBase<R> {
            int n;
            std::vector<std::unordered_map<int, R>> adjacency_list;
            bool is_original = true;
        protected:
            void add_edge_impl(int source, int target, R weight) override
            {
                adjacency_list[source][target] = weight;
            }
        public:
            using weights_type=R;
            explicit MeanPayoffGame(int n) : n(n), adjacency_list(n) {}

            virtual ~MeanPayoffGame() {}

            std::optional<R> get_weight(int source, int target) const override {
                auto it = adjacency_list[source].find(target);
                if (it == adjacency_list[source].end())
                    return std::nullopt;
                return it->second;
            }

            void set_weight(int source, int target, R weight) override {
                adjacency_list[source][target] = weight;
            }

            [[nodiscard]] size_t count_nodes() const override
            {
                return n;
            }
        };
    }

    namespace TreeMap
    {
        template<typename R>
        class MeanPayoffGame: public MeanPayoffGameBase<R> {
            int n;
            std::vector<std::map<int, R>> adjacency_list;
            bool is_original = true;
        protected:
            void add_edge_impl(int source, int target, R weight) override
            {
                adjacency_list[source][target] = weight;
            }
        public:
            using weights_type=R;
            explicit MeanPayoffGame(int n) : n(n), adjacency_list(n) {}

            virtual ~MeanPayoffGame() {}

            std::optional<R> get_weight(int source, int target) const override {
                auto it = adjacency_list[source].find(target);
                if (it == adjacency_list[source].end())
                    return std::nullopt;
                return it->second;
            }

            void set_weight(int source, int target, R weight) override {
                adjacency_list[source][target] = weight;
            }

            [[nodiscard]] size_t count_nodes() const override
            {
                return n;
            }
        };
        namespace Hash=HashMap;
        namespace Map=TreeMap;
    }
}

template<typename R>
concept MPG=requires(R *r,int i,int j)
{
    typename R::weights_type;
    {r->count_nodes()}->std::convertible_to<size_t>;
    {r->get_edges()}->std::convertible_to<std::vector<DirectedEdge<typename R::weights_type>>>;
    {r->get_weight(i,j)}->std::convertible_to<std::optional<typename R::weights_type>>;
    {r->set_weight(i,j,typename R::weights_type())};
    {r->dual()}->std::convertible_to<MeanPayoffGameBase<typename R::weights_type>*>;
};

namespace Bipartite
{

    enum Player : bool
    {
        PLAYER_0,
        PLAYER_1,
        PLAYER_MAX=PLAYER_0,
        PLAYER_MIN=PLAYER_1
    };


    int encode(int vertex,int player)
    {
        return (vertex<<1)+player;
    }

    struct Vertex
    {
        int vertex;
        Player player;
    };

    Vertex decode(int vertex)
    {
        return {vertex>>1,static_cast<Player>(vertex&1)};
    }

    int encode(const Vertex &v)
    {
        return encode(v.vertex,v.player);
    }

    Player get_player(int vertex)
    {
        return static_cast<Player>(vertex&1);
    }

    Player operator!(Player player)
    {
        return static_cast<Player>(!static_cast<bool>(player));
    }

    Player operator~(Player player)
    {
        return !player;
    }

    template<MPG Game>
    Game bipartite_meanpayoff_game(const Game &game)
    {
        auto n=game.count_nodes();
        Game bipartite_mpg(2*n);
        for(auto e:game.get_edges())
        {
            bipartite_mpg.add_edge(encode(e.source,PLAYER_0), encode(e.target,PLAYER_1), e.weight);
            bipartite_mpg.add_edge(encode(e.source,PLAYER_1), encode(e.target,PLAYER_0), e.weight);
        }
        return bipartite_mpg;
    }
}




template<MPG Game>
MinMaxSystem<typename Game::weights_type> mean_payoff_game_to_min_max_system(const Game &game)
{
    using R=typename Game::weights_type;
    MinMaxSystem<R> system;
    VariableFactory factory;
    auto n=game.count_nodes();
    std::vector<Variable> variables(2*n);
    std::vector<MinMaxConstraint<R>> constraints;
    for (int i = 0; i < (2*n); i++)
    {
        variables[i]=*factory.create();
        system.add_variable(variables[i]);
        constraints.emplace_back(Bipartite::get_player(i)==Bipartite::PLAYER_0?MinMaxType::MAX:MinMaxType::MIN,variables[i],std::vector<VariableOffset<R>>{});
    }
    using namespace Bipartite;
    for (auto e:game.get_edges())
    {
        constraints[encode(e.source,PLAYER_0)].add_argument(variables[encode(e.target,PLAYER_1)]+ e.weight);
        constraints[encode(e.source,PLAYER_1)].add_argument(variables[encode(e.target,PLAYER_0)]+ e.weight);
    }
    for(auto c:constraints)
        system.add_constraint(c);
    return system;
}

struct StrategyPair
{
    std::vector<int> strategy_0;
    std::vector<int> strategy_1;
    auto& operator[](Bipartite::Player player)
    {
        return player==Bipartite::PLAYER_0?strategy_0:strategy_1;
    }
    const auto& operator[](Bipartite::Player player) const
    {
        return player==Bipartite::PLAYER_0?strategy_0:strategy_1;
    }
};

template<MPG Game,typename Solver>
std::vector<int> optimal_strategy(const Game &game,Bipartite::Player player,Solver &solver)
{
    if(player==Bipartite::PLAYER_1)
        return optimal_strategy(*game.dual(),Bipartite::PLAYER_0,solver);
    using R=typename Game::weights_type;
    MinMaxSystem<R> system=mean_payoff_game_to_min_max_system(game);
    auto solution=solver.solve(system.to_nary_max_system().to_max_atom_system());
    std::vector<int> strategy(game.count_nodes());
    std::vector<order_closure<R>> values(game.count_nodes(),inf_min);
    for(auto e:game.get_edges()) if(solution[Bipartite::encode(e.target,Bipartite::PLAYER_1)]+e.weight >= values[e.source])
    {
        values[e.source]=solution[Bipartite::encode(e.target,Bipartite::PLAYER_1)]+e.weight;
        strategy[e.source]=e.target;
    }
    return strategy;
}

template<MPG Game,typename Solver>
StrategyPair optimal_strategy_pair(const Game &game, Solver &solver)
{
    return {optimal_strategy(game,Bipartite::PLAYER_0,solver),optimal_strategy(game,Bipartite::PLAYER_1,solver)};
}

template<MPG Game,typename Solver>
std::map<std::pair<int,Bipartite::Player>,Bipartite::Player> winner(const Game &game, Solver &solver)
{
    using R=typename Game::weights_type;
    MinMaxSystem<R> system=mean_payoff_game_to_min_max_system(game);
    auto solution=solver.solve(system.to_nary_max_system().to_max_atom_system());
    std::map<std::pair<int,Bipartite::Player>,Bipartite::Player> winner;
    for(auto variable:system.get_variables())
    {
        auto s=variable.get_id();
        auto [v,player]=Bipartite::decode(s);
        if(solution[s]>inf_min)
            winner[{v,player}]=Bipartite::PLAYER_0;
        else
            winner[{v,player}]=Bipartite::PLAYER_1;
    }
    return winner;

}

template<typename Real>
struct MeanPayoffsPair
{
    std::vector<Real> payoffs_0;
    std::vector<Real> payoffs_1;
    auto & operator[](Bipartite::Player player)
    {
        if(player==Bipartite::PLAYER_0)
            return payoffs_0;
        else
            return payoffs_1;
    }
    const auto & operator[](Bipartite::Player player) const
    {
        if(player==Bipartite::PLAYER_0)
            return payoffs_0;
        else
            return payoffs_1;
    }
};

struct WinnersPair
{
    std::vector<Bipartite::Player> winner_0;
    std::vector<Bipartite::Player> winner_1;
    auto & operator[](Bipartite::Player player)
    {
        if(player==Bipartite::PLAYER_0)
            return winner_0;
        else
            return winner_1;
    }
    const auto & operator[](Bipartite::Player player) const
    {
        if(player==Bipartite::PLAYER_0)
            return winner_0;
        else
            return winner_1;
    }
};

template<typename weights_type,typename Real=double>
MeanPayoffsPair<Real> mean_payoffs(const MeanPayoffGameBase<weights_type> &game, const StrategyPair &strategy_pair)
{
    using R=weights_type;
    auto n=game.count_nodes();
    std::vector<bool> visited(2*n),has_value(2*n);
    std::vector<integer> length(2*n);
    integer current_length=0;
    MeanPayoffsPair<Real> payoffs{std::vector<Real>(n), std::vector<Real>(n)};
    for(int i=0;i<(2*n);i++) if(!visited[i])
    {
        std::vector<int> path;
        int current=i;
        Real mean_payoff=0;
        std::vector<Real> payoffs_sum(2*n);
        length[i]=1;
        while(!visited[current])
        {
            visited[current]=true;
            path.push_back(current);
            int next;
            auto [u,player]=Bipartite::decode(current);
            auto v=strategy_pair[player][u];
            next=Bipartite::encode(strategy_pair[player][u], !player);

            if(has_value[next])
                mean_payoff=payoffs[!player][v];
            else
            {
                if(visited[next])
                {
                    Real long_payoff= game.get_weight(u, v).value() + payoffs_sum[current];
                    Real short_payoff=payoffs_sum[next];
                    mean_payoff=(long_payoff-short_payoff)/(length[current]-length[next]+1);
                }
                else
                {
                    payoffs_sum[next] = game.get_weight(u, v).value() + payoffs_sum[current];
                    length[next]=length[current]+1;
                }
            }
            current=next;
        }
        for(auto k:path) if(!has_value[k])
        {
            has_value[k]=true;
            auto [u,player]=Bipartite::decode(k);
            payoffs[player][u]=mean_payoff;
        }

    }
    return payoffs;
}


template<typename weights_type,typename Real=double>
WinnersPair winners(const MeanPayoffGameBase<weights_type>& game, const MeanPayoffsPair<Real> &MPOs)
{
    using R=weights_type;
    auto n=game.count_nodes();
    std::vector<bool> visited(2*n),has_value(2*n);
    WinnersPair winners{std::vector<Bipartite::Player>(n), std::vector<Bipartite::Player>(n)};
    std::transform(MPOs.payoffs_0.begin(),MPOs.payoffs_0.end(),winners.winner_0.begin(),[](auto x){return x>=0?Bipartite::PLAYER_0:Bipartite::PLAYER_1;});
    std::transform(MPOs.payoffs_1.begin(),MPOs.payoffs_1.end(),winners.winner_1.begin(),[](auto x){return x>=0?Bipartite::PLAYER_0:Bipartite::PLAYER_1;});
    return winners;
}

template<typename weights_type,typename Real=double>
WinnersPair winners(const MeanPayoffGameBase<weights_type>& game, const StrategyPair &strategy_pair)
{
    return winners(game,mean_payoffs<weights_type,Real>(game,strategy_pair));
}

#endif //MPGCPP_MEANPAYOFFGAME_H
