//
// Created by ramizouari on 04/04/23.
//

#ifndef MPGCPP_DENSESOLVER_H
#define MPGCPP_DENSESOLVER_H
#include "../MaxAtomSolver.h"

template<typename Map,typename R>
class MaxAtomArcConsistencyDenseSolver: public MaxAtomSolver<Map,R>
{
    using MaxAtomSolver<Map,R>::bound_estimator;
public:
    using MaxAtomSolver<Map,R>::set_bound_estimator;
    using MaxAtomSolver<Map,R>::solve;
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        auto K=bound_estimator->estimate(system);
        std::set<Variable> maximal_variables(variables.begin(),variables.end());
        Map assignment;
        for(auto v:variables)
            assignment[v.get_id()]=K;
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
        std::vector<bool> in_queue(system.count_variables(),false);
        for(auto v:variables)
        {
            Q.push(v);
            in_queue[v.get_id()]=true;
        }
        while(!Q.empty() && !maximal_variables.empty())
        {
            auto y=Q.front();
            Q.pop();
            in_queue[y.get_id()]=false;
            for(auto [x,z,c]:rhs_constraints[y])
            {
                if(assignment[x.get_id()]>std::max(assignment[y.get_id()],assignment[z.get_id()])+c)
                {
                    assignment[x.get_id()]=std::max(assignment[y.get_id()],assignment[z.get_id()])+c;
                    if(!in_queue[x.get_id()])
                        Q.push(x);
                    in_queue[x.get_id()]=true;
                    maximal_variables.erase(x);
                }
            }
            if (assignment[y.get_id()] < - system.radius)
                assignment[y.get_id()] = inf_min;
        }
        if(maximal_variables.empty())
            std::fill(assignment.begin(),assignment.end(),inf_min);
        return assignment;
    }
};

template<typename R>
class MaxAtomArcConsistencyDenseSolver<std::vector<order_closure<R>>,R>: public MaxAtomSolver<std::vector<order_closure<R>>,R>
{
    using MaxAtomSolver<std::vector<order_closure<R>>,R>::bound_estimator;
public:
    using Map=std::vector<order_closure<R>>;
    using MaxAtomSolver<Map,R>::set_bound_estimator;
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        Map assignment(system.count_variables());
        auto K=bound_estimator->estimate(system);
        std::set<Variable> maximal_variables(variables.begin(),variables.end());
        for(auto v:variables)
            assignment[v.get_id()]=K;
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
        std::vector<bool> in_queue(system.count_variables(),false);
        for(auto v:variables)
        {
            Q.push(v);
            in_queue[v.get_id()]=true;
        }
        while(!Q.empty() && !maximal_variables.empty())
        {
            auto y=Q.front();
            Q.pop();
            in_queue[y.get_id()]=false;
            for(auto [x,z,c]:rhs_constraints[y]) if(assignment[x.get_id()]>std::max(assignment[y.get_id()],assignment[z.get_id()])+c)
                {
                    assignment[x.get_id()]=std::max(assignment[y.get_id()],assignment[z.get_id()])+c;
                    maximal_variables.erase(x);
                    if(!in_queue[x.get_id()])
                        Q.push(x);
                    in_queue[x.get_id()]=true;
                    if (assignment[x.get_id()] < - system.radius)
                        assignment[x.get_id()] = inf_min;
                }
        }
        if(maximal_variables.empty())
            std::fill(assignment.begin(),assignment.end(),inf_min);
        return assignment;
    }
};

namespace Implementation
{
    namespace Vector
    {
        template<typename R>
        using DenseSolver=MaxAtomArcConsistencyDenseSolver<std::vector<order_closure<R>>,R>;
    }
    namespace HashMap
    {
        template<typename R>
        using DenseSolver=MaxAtomArcConsistencyDenseSolver<std::unordered_map<Variable,order_closure<R>>,R>;
    }
    namespace TreeMap
    {
        template<typename R>
        using DenseSolver=MaxAtomArcConsistencyDenseSolver<std::map<Variable,order_closure<R>>,R>;
    }
}

#endif //MPGCPP_DENSESOLVER_H
