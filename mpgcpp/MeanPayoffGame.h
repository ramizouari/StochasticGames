//
// Created by ramizouari on 12/03/23.
//

#ifndef MPGCPP_MEANPAYOFFGAME_H
#define MPGCPP_MEANPAYOFFGAME_H
#include "graph/graph.h"
#include "MinMaxSystem.h"
#include <optional>
#include <memory>

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
        PLAYER_1
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

    bool get_player(int vertex)
    {
        return vertex&1;
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
#endif //MPGCPP_MEANPAYOFFGAME_H
