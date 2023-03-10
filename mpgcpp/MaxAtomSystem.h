//
// Created by ramizouari on 12/03/23.
//

#ifndef MPGCPP_MAXATOMSYSTEM_H
#define MPGCPP_MAXATOMSYSTEM_H

#include <utility>
#include <queue>
#include <unordered_map>
#include <ranges>

#include "Variable.h"
#include "algebra/order.h"

template<typename R>
class MaxAtomConstraint
{

public:
    Variable x,y,z;
    R c;
    MaxAtomConstraint(Variable &x,Variable &y,Variable &z,R c):x(x),y(y),z(z),c(c){}

};

template<typename R>
class NaryMaxAtomConstraint
{

public:
    Variable x;
    std::vector<Variable> Y;
    R c;
    NaryMaxAtomConstraint(Variable x,std::vector<Variable> Y,R c):x(std::move(x)),Y(std::move(Y)),c(c){}

};

namespace std
{
    template<typename R>
    struct tuple_size<MaxAtomConstraint<R>>:std::integral_constant<size_t,4>{};
    template<typename R>
    struct tuple_element<0,MaxAtomConstraint<R>>{using type=Variable;};
    template<typename R>
    struct tuple_element<1,MaxAtomConstraint<R>>{using type=Variable;};
    template<typename R>
    struct tuple_element<2,MaxAtomConstraint<R>>{using type=Variable;};
    template<typename R>
    struct tuple_element<3,MaxAtomConstraint<R>>{using type=R;};
    template <size_t I,typename R>
    auto get(const MaxAtomConstraint<R>& c)
    {
        if constexpr(I==0)
            return c.x;
        else if constexpr(I==1)
            return c.y;
        else if constexpr(I==2)
            return c.z;
        else if constexpr(I==3)
            return c.c;
    }

    template<typename R>
    struct tuple_size<NaryMaxAtomConstraint<R>>:std::integral_constant<size_t,3>{};
    template<typename R>
    struct tuple_element<0,NaryMaxAtomConstraint<R>>{using type=Variable;};
    template<typename R>
    struct tuple_element<1,NaryMaxAtomConstraint<R>>{using type=std::vector<Variable>;};
    template<typename R>
    struct tuple_element<2,NaryMaxAtomConstraint<R>>{using type=R;};
    template <size_t I,typename R>
    auto get(const NaryMaxAtomConstraint<R>& c)
    {
        if constexpr(I==0)
            return c.x;
        else if constexpr(I==1)
            return c.Y;
        else if constexpr(I==2)
            return c.c;
    }

}

template<typename R>
class MaxAtomSystem {
protected:
    size_t n=0;
    std::vector<MaxAtomConstraint<R>> constraints;
public:
    R  radius=0;

    MaxAtomSystem()= default;

    void add_constraint(Variable x,Variable y,Variable z,R c)
    {
        constraints.emplace_back(x,y,z,c);
        n=std::max({n,x.get_id()+1,y.get_id()+1,z.get_id()+1});
        radius+=std::abs(c);
    }

    void add_constraint(MaxAtomConstraint<R> constraint)
    {
        constraints.push_back(constraint);
    }
    void add_constraints(const std::vector<MaxAtomConstraint<R>>& C)
    {
        for(auto c:C)
            add_constraint(c);
    }
    void add_variable(const Variable &x)
    {
        n=std::max(n,x.get_id()+1);
    }
    void add_variables(const std::vector<Variable *> &V)
    {
        for(auto v:V)
            add_variable(v);
    }

    [[nodiscard]] size_t count_variables() const
    {
        return n;
    }

    auto get_variables() const
    {
        return std::views::iota(0UL,count_variables()) | std::views::transform([](auto i){return Variable(i);});
    }

    [[nodiscard]] const std::vector<MaxAtomConstraint<R>> & get_constraints() const
    {
        return constraints;
    }

};


template<typename R>
class NaryMaxAtomSystem {
protected:
    std::vector<NaryMaxAtomConstraint<R>> constraints;
    size_t n=0;
public:

    NaryMaxAtomSystem()= default;

    void add_constraint(Variable x,const std::vector<Variable> &Y,R c)
    {
        constraints.emplace_back(x,Y,c);
        n=std::max(n,x.get_id()+1);
        for(const auto& y:Y)
            n=std::max(n,y.get_id()+1);
    }

    void add_constraint(const NaryMaxAtomSystem<R>& constraint)
    {
        constraints.push_back(constraint);
    }
    void add_constraints(const std::vector<MaxAtomConstraint<R>>& C)
    {
        for(auto c:C)
            add_constraint(c);
    }
    void add_variable(const Variable &v)
    {
        n=std::max(n,v.get_id()+1);
    }
    void add_variables(const std::vector<Variable> &V)
    {
        for(auto v:V)
            add_variable(v);
    }

    [[nodiscard]] size_t count_variables() const
    {
        return n;
    }

    auto get_variables() const
    {
        return std::views::iota(0UL,count_variables()) | std::views::transform([](auto i){return Variable(i);});
    }
    const std::vector<MaxAtomConstraint<R>>& get_constraints() const
    {
        return constraints;
    }

    MaxAtomSystem<R> to_max_atom_system() const
    {
        VariableFactoryRange factory(count_variables());
        std::map<std::pair<Variable,Variable>,Variable> mapper;
        MaxAtomSystem<R> system;
        for(auto v:get_variables())
            mapper[{v,v}]=v;
        for(auto C:constraints)
        {
            auto x=C.x;
            auto Y=C.Y;
            auto c=C.c;
            while(Y.size()>2)
            {
                auto y1=Y.back();
                Y.pop_back();
                auto y2=Y.back();
                Y.pop_back();
                if(!mapper.contains({y1,y2}))
                {
                    auto z= mapper[{y1, y2}] = *factory.create();
                    mapper[{y2,y1}]=z;
                    system.add_constraint(z,y1,y2,0);
                }
                auto z=mapper[{y1,y2}];
                Y.push_back(z);
            }
            if(Y.size()==0)
                throw std::runtime_error("Empty constraint");
            system.add_constraint(x,Y[0],Y[Y.size()-1],c);

        }
        return system;
    }

};

template<typename Map,typename R>
class MaxAtomSolver
{
public:
    virtual Map solve(const MaxAtomSystem<R> &system)=0;
};

template<typename Map,typename R>
class MaxAtomArcConsistencySolver: public MaxAtomSolver<Map,R>
{

public:
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        Map assignment;
        for(auto v:variables)
            assignment[v.get_id()]=system.radius;
        for(auto C:system.get_constraints())
        {
            auto x=std::get<0>(C);
            auto y=std::get<1>(C);
            auto z=std::get<2>(C);
            auto c=std::get<3>(C);
            rhs_constraints[y].push_back({x,z,c});
            rhs_constraints[z].push_back({x,y,c});
        }

        std::queue<Variable> Q;
        for(auto v:variables)
            Q.push(v);
        while(!Q.empty())
        {
            auto y=Q.front();
            Q.pop();
            for(auto [x,z,c]:rhs_constraints[y])
            {
                if(assignment[x.get_id()]>std::max(assignment[y.get_id()],assignment[z.get_id()])+c)
                {
                    assignment[x.get_id()]=std::max(assignment[y.get_id()],assignment[z.get_id()])+c;
                    Q.push(x);
                }
            }
            if (assignment[y.get_id()] < - system.radius)
                assignment[y.get_id()] = inf_min;
        }
        return assignment;
    }
};

namespace Implementation
{
    namespace Vector
    {
        template<typename R>
        using VariableAssignment=std::pair<Variable,order_closure<R>>;
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::vector<VariableAssignment<R>>,R>;
    }
    namespace HashMap
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::unordered_map<Variable,order_closure<R>>,R>;
    }

    namespace TreeMap
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::map<Variable,order_closure<R>>,R>;
    }
}


#endif //MPGCPP_MAXATOMSYSTEM_H
