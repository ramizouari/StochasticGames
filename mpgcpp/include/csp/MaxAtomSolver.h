//
// Created by ramizouari on 27/03/23.
//

#ifndef MPGCPP_MAXATOMSOLVER_H
#define MPGCPP_MAXATOMSOLVER_H

#include <map>
#include "MaxAtomSystem.h"
#include "MinMaxSystem.h"

template<typename Map,typename R>
class MaxAtomSolver
{
public:
    virtual Map solve(const MaxAtomSystem<R> &system)=0;
    virtual Map solve(const NaryMaxAtomSystem<R> &system)
    {
        auto assignment=solve(system.to_max_atom_system());
        Map result;
        for(auto v:system.get_variables())
            result[v]=assignment[v];
        return result;
    }
    virtual Map solve(const MinMaxSystem<R> &system)
    {
        auto assignment=solve(system.to_nary_max_system());
        Map result;
        for(auto v:system.get_variables())
            result[v]=assignment[v];
        return result;
    }
    virtual ~MaxAtomSolver()= default;
};

template<typename R>
class MaxAtomSolver<std::vector<order_closure<R>>,R>
{
public:
    virtual std::vector<order_closure<R>> solve(const MaxAtomSystem<R> &system)=0;
    virtual std::vector<order_closure<R>> solve(const NaryMaxAtomSystem<R> &system)
    {
        auto assignment=solve(system.to_max_atom_system());
        assignment.resize(system.count_variables());
        return assignment;
    }
    virtual std::vector<order_closure<R>> solve(const MinMaxSystem<R> &system)
    {
        auto assignment=solve(system.to_nary_max_system());
        assignment.resize(system.count_variables());
        return assignment;
    }
    virtual ~MaxAtomSolver()= default;
};

template<typename Map,typename R>
class MaxAtomArcConsistencySolver: public MaxAtomSolver<Map,R>
{

public:
    using MaxAtomSolver<Map,R>::solve;
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
        std::vector<bool> in_queue(system.count_variables(),false);
        for(auto v:variables)
        {
            Q.push(v);
            in_queue[v.get_id()]=true;
        }
        while(!Q.empty())
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
                }
            }
            if (assignment[y.get_id()] < - system.radius)
                assignment[y.get_id()] = inf_min;
        }
        return assignment;
    }
};

template<typename R>
class MaxAtomArcConsistencySolver<std::vector<order_closure<R>>,R>: public MaxAtomSolver<std::vector<order_closure<R>>,R>
{

public:
    using Map=std::vector<order_closure<R>>;
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        Map assignment(system.count_variables());
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
        std::vector<bool> in_queue(system.count_variables(),false);
        for(auto v:variables)
        {
            Q.push(v);
            in_queue[v.get_id()]=true;
        }
        while(!Q.empty())
        {
            auto y=Q.front();
            Q.pop();
            in_queue[y.get_id()]=false;
            for(auto [x,z,c]:rhs_constraints[y]) if(assignment[x.get_id()]>std::max(assignment[y.get_id()],assignment[z.get_id()])+c)
            {
                assignment[x.get_id()]=std::max(assignment[y.get_id()],assignment[z.get_id()])+c;
                if(!in_queue[x.get_id()])
                Q.push(x);
                in_queue[x.get_id()]=true;
                if (assignment[x.get_id()] < - system.radius)
                assignment[x.get_id()] = inf_min;
            }
        }
    return assignment;
    }
};

template<typename Map,typename R>
class MaxAtomFixedPointSolver: public MaxAtomSolver<Map,R>
{

public:
    using MaxAtomSolver<Map,R>::solve;
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        Map assignment;
        for(auto v:variables)
            assignment[v.get_id()]=system.radius;
        bool convergence=false;
        while(!convergence)
        {
            convergence=true;
            for(auto C:system.get_constraints())
            {
                auto x=std::get<0>(C);
                auto y=std::get<1>(C);
                auto z=std::get<2>(C);
                auto c=std::get<3>(C);
                if(assignment[x.get_id()]>std::max(assignment[y.get_id()],assignment[z.get_id()])+c)
                {
                    assignment[x.get_id()]=std::max(assignment[y.get_id()],assignment[z.get_id()])+c;
                    convergence=false;
                    if(assignment[x.get_id()] < - system.radius)
                        assignment[x.get_id()] = inf_min;
                }
            }
        }
        return assignment;
    }
};

template<typename R>
class MaxAtomFixedPointSolver<std::vector<order_closure<R>>,R>: public MaxAtomSolver<std::vector<order_closure<R>>,R>
{
public:
    using Map=std::vector<order_closure<R>>;
    using MaxAtomSolver<Map,R>::solve;
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        Map assignment(system.count_variables());
        for(auto v:variables)
        assignment[v.get_id()]=system.radius;
        bool convergence=false;
        while(!convergence)
        {
            convergence=true;
            for(auto C:system.get_constraints())
            {
                auto x=std::get<0>(C);
                auto y=std::get<1>(C);
                auto z=std::get<2>(C);
                auto c=std::get<3>(C);
                if(assignment[x.get_id()]>std::max(assignment[y.get_id()],assignment[z.get_id()])+c)
                {
                    assignment[x.get_id()]=std::max(assignment[y.get_id()],assignment[z.get_id()])+c;
                    convergence=false;
                    if(assignment[x.get_id()] < - system.radius)
                        assignment[x.get_id()] = inf_min;
                }
            }
        }
        return assignment;
    }
};


namespace Implementation
{
    namespace Vector
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::vector<order_closure<R>>,R>;
        template<typename R>
        using MaxAtomSystemSolverFixedPoint=MaxAtomFixedPointSolver<std::vector<order_closure<R>>,R>;
    }
    namespace HashMap
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::unordered_map<Variable,order_closure<R>>,R>;
        template<typename R>
        using MaxAtomSystemSolverFixedPoint=MaxAtomFixedPointSolver<std::unordered_map<Variable,order_closure<R>>,R>;
    }

    namespace TreeMap
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::map<Variable,order_closure<R>>,R>;
        template<typename R>
        using MaxAtomSystemSolverFixedPoint=MaxAtomFixedPointSolver<std::map<Variable,order_closure<R>>,R>;
    }
}

#endif //MPGCPP_MAXATOMSOLVER_H
