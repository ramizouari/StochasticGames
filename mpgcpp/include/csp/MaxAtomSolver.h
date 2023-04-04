//
// Created by ramizouari on 27/03/23.
//

#ifndef MPGCPP_MAXATOMSOLVER_H
#define MPGCPP_MAXATOMSOLVER_H

#include <map>
#include "MaxAtomSystem.h"
#include "MinMaxSystem.h"

template<typename R>
class MaxAtomBoundEstimator
{
public:
    virtual R estimate(const MaxAtomSystem<R> &system)=0;
    virtual R estimate(const NaryMaxAtomSystem<R> &system)
    {
        return estimate(system.to_max_atom_system());
    }
    virtual R estimate(const MinMaxSystem<R> &system)
    {
        return estimate(system.to_nary_max_system());
    }
    virtual ~MaxAtomBoundEstimator()= default;
};

template<typename R>
class DefaultMaxAtomBoundEstimator: public MaxAtomBoundEstimator<R>
{
public:
    R estimate(const MaxAtomSystem<R> &system) override
    {
        return system.radius;
    }
};

template<typename R>
class LinearMaxAtomBoundEstimator: public MaxAtomBoundEstimator<R>
{
    R alpha,beta;
public:
    LinearMaxAtomBoundEstimator():LinearMaxAtomBoundEstimator(1,0)
    {
    }
    explicit LinearMaxAtomBoundEstimator(R alpha,R beta=0):alpha(alpha),beta(beta)
    {

    }
    R estimate(const MaxAtomSystem<R> &system) override
    {
        R max_c{};
        for(auto C:system.get_constraints())
        {
            auto c=std::get<3>(C);
            max_c=std::max(std::abs(max_c),c);
        }
        auto radius=system.radius;
        return alpha*max_c+beta;
    }
};

template<typename Map,typename R>
class MaxAtomSolver
{
protected:
    std::unique_ptr<MaxAtomBoundEstimator<R>> bound_estimator= std::make_unique<DefaultMaxAtomBoundEstimator<R>>();
public:
    MaxAtomSolver() = default;

    MaxAtomSolver( MaxAtomSolver && O):bound_estimator(std::move(O.bound_estimator))
    {

    }
    MaxAtomSolver & operator=(MaxAtomSolver && O) noexcept
    {
        bound_estimator=std::move(O.bound_estimator);
        return *this;
    }

    void set_bound_estimator(MaxAtomBoundEstimator<R> *_bound_estimator)
    {
        bound_estimator.reset(_bound_estimator);
    }

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
    protected:
    std::unique_ptr<MaxAtomBoundEstimator<R>> bound_estimator= std::make_unique<DefaultMaxAtomBoundEstimator<R>>();

public:

    MaxAtomSolver() = default;

    MaxAtomSolver( MaxAtomSolver && O):bound_estimator(std::move(O.bound_estimator))
    {

    }
    MaxAtomSolver & operator=(MaxAtomSolver && O)
    {
        bound_estimator=std::move(O.bound_estimator);
        return *this;
    }

    void set_bound_estimator(MaxAtomBoundEstimator<R> *_bound_estimator)
    {
        bound_estimator.reset(_bound_estimator);
    }
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

        Map assignment;
        for(auto v:variables)
            assignment[v.get_id()]=K;
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
    using MaxAtomSolver<std::vector<order_closure<R>>,R>::bound_estimator;
public:
    using Map=std::vector<order_closure<R>>;
    using MaxAtomSolver<Map,R>::set_bound_estimator;
    using MaxAtomSolver<Map,R>::solve;
    Map solve(const MaxAtomSystem<R> &system) override
    {
        using ReducedConstraint=std::tuple<Variable,Variable,R>;
        std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
        auto variables=system.get_variables();
        Map assignment(system.count_variables());
        auto K=bound_estimator->estimate(system);
        for(auto v:variables)
        assignment[v.get_id()]=K;
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
