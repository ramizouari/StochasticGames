//
// Created by ramizouari on 29/03/23.
//

#ifndef MPGCPP_PARALLELMAXATOMSOLVER_H
#define MPGCPP_PARALLELMAXATOMSOLVER_H

#include <atomic>
#include "../MaxAtomSolver.h"
#include <execution>
#include <thread>
#include <shared_mutex>
#include <iostream>
#include "concurrentqueue/blockingconcurrentqueue.h"

namespace Parallel
{
    template<typename T>
    struct Shared
    {
        T value;
        mutable std::shared_mutex mutex;
        Shared(T value):value(value){}
        Shared():value(){}
        Shared(const Shared & other):value(other.value){}
        T get() const
        {
            std::shared_lock lock(mutex);
            return value;
        }
        void set(T x)
        {
            std::unique_lock lock(mutex);
            this->value=x;
        }

        T load() const
        {
            return get();
        }

        void store(T x)
        {
            set(x);
        }

        operator T() const
        {
            return get();
        }

        T operator=(T x)
        {
            set(x);
            return x;
        }

    };

    template<typename T>
    using SharedType=Shared<T>;

    template<typename Map,typename R>
    class WorkerThread
    {
        moodycamel::BlockingConcurrentQueue<Variable> &Q;
        std::vector<SharedType<bool>> &in_queue;
        using ReducedConstraint = std::tuple<Variable, Variable, R>;
        using RHSConstraint = std::unordered_map<Variable, std::vector<ReducedConstraint>>;
        const RHSConstraint &rhs_constraints;
        Map &assignment;
        R radius;
        bool done=false;
        std::thread thread;

        void run()
        {
            Variable y;
            while (Q.try_dequeue(y))
            {
                in_queue[y.get_id()] = false;
                auto it= rhs_constraints.find(y);
                if(it!=rhs_constraints.end()) for (auto [x, z, c]: it->second)
                {
                    if (assignment[x.get_id()].load() > std::max(assignment[y.get_id()].load(), assignment[z.get_id()].load()) + c)
                    {
                        assignment[x.get_id()] = std::max(assignment[y.get_id()].load(), assignment[z.get_id()].load()) + c;
                        if (!in_queue[x.get_id()])
                            Q.enqueue(x);
                        in_queue[x.get_id()] = true;
                    }
                }
                if (assignment[y.get_id()].load() < -radius)
                    assignment[y.get_id()] = inf_min;
            }
            done=true;
        }
    public:

        WorkerThread(Map & assignment, R radius,SharedType<bool>& done, moodycamel::BlockingConcurrentQueue<Variable> &Q,
                     std::vector<SharedType<bool>> &in_queue,const RHSConstraint& rhs_constraints):
                Q(Q),assignment(assignment),in_queue(in_queue),
                rhs_constraints(rhs_constraints),radius(radius),thread(&WorkerThread::run,this)
        {
        }

        WorkerThread()=delete;
        WorkerThread(const WorkerThread&)=delete;

        void join()
        {
            thread.join();
        }

        std::thread& get_thread()
        {
            return thread;
        }
    };

    template<typename Map,typename SharedMap,typename R>
    class ParallelMaxAtomArcConsistencySolver: public MaxAtomSolver<Map,R>
    {
        size_t n_threads;
        std::vector<std::unique_ptr<WorkerThread<SharedMap,R>>> workers;
        using MaxAtomSolver<Map,R>::bound_estimator;
    public:
        using MaxAtomSolver<Map,R>::set_bound_estimator;

        explicit ParallelMaxAtomArcConsistencySolver(size_t n_threads=std::thread::hardware_concurrency()):n_threads(n_threads)
        {
        }
        using MaxAtomSolver<Map,R>::solve;
        Map solve(const MaxAtomSystem<R> &system) override
        {
            using ReducedConstraint=std::tuple<Variable,Variable,R>;
            std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
            auto variables=system.get_variables();
            SharedMap sharedAssignment;
            auto K=bound_estimator->estimate(system);
            for(auto v:variables)
                sharedAssignment[v.get_id()]=K;

            for(auto C:system.get_constraints())
            {
                auto x=std::get<0>(C);
                auto y=std::get<1>(C);
                auto z=std::get<2>(C);
                auto c=std::get<3>(C);
                rhs_constraints[y].push_back({x,z,c});
                rhs_constraints[z].push_back({x,y,c});
            }

            moodycamel::BlockingConcurrentQueue<Variable> Q;
            std::vector<SharedType<bool>> in_queue(system.count_variables());

            for(auto v:variables)
            {
                Q.enqueue(v);
                in_queue[v.get_id()]=true;
            }

            std::vector<std::unique_ptr<WorkerThread<SharedMap,R>>> workers;
            SharedType<bool> done(false);
            for(size_t i=0;i<n_threads;i++)
                workers.emplace_back(std::make_unique<WorkerThread<SharedMap,R>>(sharedAssignment, system.radius, done, Q, in_queue, rhs_constraints));
            for(auto& w:workers)
                w->get_thread().join();
            Map assignment;
            for(auto v:variables)
                assignment[v.get_id()]=sharedAssignment[v.get_id()];
            return assignment;
        }
    };

    template<typename R>
    class ParallelMaxAtomArcConsistencySolver<std::vector<order_closure<R>>,std::vector<SharedType<order_closure<R>>>,R>:
            public MaxAtomSolver<std::vector<order_closure<R>>,R>
    {
        size_t n_threads;
        using Map=std::vector<order_closure<R>>;
        using SharedMap=std::vector<SharedType<order_closure<R>>>;
        using MaxAtomSolver<Map,R>::bound_estimator;
    public:
        using MaxAtomSolver<Map,R>::set_bound_estimator;

        explicit ParallelMaxAtomArcConsistencySolver(size_t n_threads=std::thread::hardware_concurrency()):n_threads(n_threads)
        {
        }

        using MaxAtomSolver<std::vector<order_closure<R>>,R>::solve;
        Map solve(const MaxAtomSystem<R> &system) override
        {
            using ReducedConstraint=std::tuple<Variable,Variable,R>;
            std::unordered_map<Variable,std::vector<ReducedConstraint>> rhs_constraints;
            auto variables=system.get_variables();
            SharedMap sharedAssignment(system.count_variables());
            auto K=bound_estimator->estimate(system);
            for(auto v:variables)
                sharedAssignment[v.get_id()]=K;

            for(auto C:system.get_constraints())
            {
                auto x=std::get<0>(C);
                auto y=std::get<1>(C);
                auto z=std::get<2>(C);
                auto c=std::get<3>(C);
                rhs_constraints[y].push_back({x,z,c});
                rhs_constraints[z].push_back({x,y,c});
            }

            moodycamel::BlockingConcurrentQueue<Variable> Q;
            std::vector<SharedType<bool>> in_queue(system.count_variables());

            for(auto v:variables)
            {
                Q.enqueue(v);
                in_queue[v.get_id()]=true;
            }

            std::vector<std::unique_ptr<WorkerThread<SharedMap,R>>> workers;
            SharedType<bool> done(false);
            for(size_t i=0;i<n_threads;i++)
                workers.emplace_back(std::make_unique<WorkerThread<SharedMap,R>>(sharedAssignment, system.radius, done, Q, in_queue, rhs_constraints));
            for(auto& w:workers)
                w->get_thread().join();
            Map assignment(system.count_variables());
            for(auto v:variables)
                assignment[v.get_id()]=sharedAssignment[v.get_id()];
            return assignment;
        }
    };


    template<typename Map,typename R>
    class ParallelMaxAtomFixedPointSolver: public MaxAtomSolver<Map,R>
    {
        using MaxAtomSolver<Map,R>::bound_estimator;
    public:
        using MaxAtomSolver<Map,R>::solve;
        using MaxAtomSolver<Map,R>::set_bound_estimator;
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
                std::for_each(std::execution::par,system.count_constraints().begin(),system.count_constraints().end(),[&](int i)
                {
                    auto C=system.get_constraint()[i];
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
                });
            }
            return assignment;
        }
    };

    template<typename R>
    class ParallelMaxAtomFixedPointSolver<std::vector<order_closure<R>>,R>: public MaxAtomSolver<std::vector<order_closure<R>>,R>
    {
        using MaxAtomSolver<std::vector<order_closure<R>>,R>::bound_estimator;
    public:
        using Map=std::vector<order_closure<R>>;
        using MaxAtomSolver<Map,R>::solve;
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
            std::atomic<bool> convergence=false;
            while(!convergence)
            {
                convergence=true;
                std::for_each(std::execution::par,system.get_constraints().begin(),system.get_constraints().end(),[&](auto& C)
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
                });
            }
            return assignment;
        }
    };
}


namespace Implementation::Parallel
{
    namespace Vector
    {
        template<typename R>
        using MaxAtomSystemSolver=::Parallel::ParallelMaxAtomArcConsistencySolver<std::vector<order_closure<R>>,
                std::vector<::Parallel::SharedType<order_closure<R>>>,R>;
        template<typename R>
        using MaxAtomFixedPointSolver=::Parallel::ParallelMaxAtomFixedPointSolver<std::vector<order_closure<R>>,R>;
    }
    namespace HashMap
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::unordered_map<Variable,order_closure<R>>,R>;
        template<typename R>
        using MaxAtomFixedPointSolver=::Parallel::ParallelMaxAtomFixedPointSolver<std::unordered_map<Variable,order_closure<R>>,R>;
    }

    namespace TreeMap
    {
        template<typename R>
        using MaxAtomSystemSolver=MaxAtomArcConsistencySolver<std::map<Variable,order_closure<R>>,R>;
        template<typename R>
        using MaxAtomFixedPointSolver=::Parallel::ParallelMaxAtomFixedPointSolver<std::map<Variable,order_closure<R>>,R>;
    }
}


#endif //MPGCPP_PARALLELMAXATOMSOLVER_H
