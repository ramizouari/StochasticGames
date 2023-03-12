//
// Created by ramizouari on 12/03/23.
//

#ifndef MPGCPP_MINMAXSYSTEM_H
#define MPGCPP_MINMAXSYSTEM_H

#include <utility>
#include <vector>
#include <map>
#include "Variable.h"
#include "MaxAtomSystem.h"

template<typename R>
struct VariableOffset
{
    Variable variable;
    R offset;
    public:
    VariableOffset(Variable  variable, R offset=R{}):variable(std::move(variable)),offset(offset){}
    template<typename T>
    VariableOffset(VariableOffset<T> other):variable(other.variable),offset(other.offset){}
};

template<typename R>
VariableOffset<R> operator+(Variable v,R offset)
{
    return VariableOffset<R>(v,offset);
}

template<typename R>
VariableOffset<R> operator+(R offset,Variable v)
{
    return VariableOffset<R>(v,offset);
}

template<typename R>
VariableOffset<R> operator-(Variable v,R offset)
{
    return VariableOffset<R>(v,-offset);
}

namespace std
{
    template<typename R>
    struct tuple_size<VariableOffset<R>>:std::integral_constant<std::size_t,2>{};
    template<typename R>
    struct tuple_element<0,VariableOffset<R>>{using type=Variable;};
    template<typename R>
    struct tuple_element<1,VariableOffset<R>>{using type=std::int64_t;};
    template <std::size_t I,typename R>
    auto get(const VariableOffset<R>& c)
    {
        if constexpr(I==0)
            return c.variable;
        else if constexpr(I==1)
            return c.offset;
    }
}

enum class MinMaxType :bool
{
    MIN,MAX
};

template<typename R>
struct ReducedMinMaxConstraint {
    MinMaxType op;
    Variable x;
    std::vector<Variable> Z;
public:
    ReducedMinMaxConstraint(MinMaxType op,Variable&x, std::vector<Variable> Z): op(op), x(x), Z(std::move(Z)){}
    void add_argument(Variable z)
    {
        Z.push_back(z);
    }
};
template<typename R>
struct MinMaxConstraint
{
    MinMaxType op;
    Variable x;
    std::vector<VariableOffset<R>> Z;
public:
    MinMaxConstraint(MinMaxType op,Variable x, std::vector<VariableOffset<R>> Z):op(op),x(std::move(x)),Z(std::move(Z)){}
    void add_argument(Variable z, std::int64_t offset)
    {
        Z.emplace_back(z,offset);
    }
    void add_argument(VariableOffset<R> z)
    {
        Z.push_back(z);
    }
};

namespace std
{
    template<typename R>
    struct tuple_size<MinMaxConstraint<R>>:std::integral_constant<size_t,3>{};
    template<typename R>
    struct tuple_element<0,MinMaxConstraint<R>>{using type=MinMaxType;};
    template<typename R>
    struct tuple_element<1,MinMaxConstraint<R>>{using type=Variable;};
    template<typename R>
    struct tuple_element<2,MinMaxConstraint<R>>{using type=std::vector<VariableOffset<R>>;};
    template <size_t I,typename R>
    auto get(const MinMaxConstraint<R>& c)
    {
        if constexpr(I==0)
            return c.op;
        else if constexpr(I==1)
            return c.x;
        else if constexpr(I==2)
            return c.Z;
    }
}

template<typename R>
class MinMaxSystem
{
    std::vector<MinMaxConstraint<R>> constraints;
    size_t n=0;
public:
    MinMaxSystem()= default;
    void add_constraint(MinMaxType op,Variable x, std::vector<VariableOffset<R>> Z)
    {
        constraints.emplace_back(op,x,std::move(Z));
    }
    void add_constraint(const MinMaxConstraint<R>& constraint)
    {
        constraints.push_back(constraint);
    }

    void add_variable(const Variable& x)
    {
        n=std::max(n,x.get_id()+1);
    }

    [[nodiscard]] auto get_variables() const
    {
        return std::views::iota(0UL,n) | std::views::transform([](auto x){return Variable(x);});
    }
    const std::vector<MinMaxConstraint<R>>& get_constraints() const
    {
        return constraints;
    }

    NaryMaxAtomSystem<R> to_nary_max_system()
    {
        NaryMaxAtomSystem<R> system;
        std::map<std::pair<Variable,R>,Variable> mapper;
        VariableFactoryRange factory(n);
        for(auto x:get_variables())
        {
            system.add_variable(x);
            mapper[{x,0}]=x;
        }
        for(auto &C:constraints)
        {
            auto x=C.x;
            auto op=C.op;
            auto Z=C.Z;
            if(op==MinMaxType::MIN) for(auto &z:Z)
                system.add_constraint(x,{z.variable},z.offset);
            else
            {
                std::vector<Variable> S;
                S.reserve(Z.size());
                for (auto &z: Z)
                {
                    if(!mapper.count({z.variable,z.offset}))

                    {
                        auto y=*factory.create();
                        mapper[{z.variable,z.offset}]=y;
                        S.push_back(y);
                        system.add_constraint(y,{z.variable},z.offset);
                    }
                    S.push_back(mapper[{z.variable,z.offset}]);
                }
                system.add_constraint(x,S,0);
            }
        }
        return system;
    }
};

#endif //MPGCPP_MINMAXSYSTEM_H
