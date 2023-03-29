//
// Created by ramizouari on 29/03/23.
//

#ifndef MPGCPP_SOLVERCONVERTER_H
#define MPGCPP_SOLVERCONVERTER_H
#include "../MaxAtomSolver.h"

template<typename From,typename To>
class SolverConverter: public MaxAtomSolver<To,typename From::value_type>
{
    MaxAtomSolver<From,typename From::value_type> *solver;
public:
    SolverConverter(MaxAtomSolver<From,typename From::value_type> *solver):solver(solver){}
    To solve(const MaxAtomSystem<typename From::value_type> &system) override
    {
        From fromAssignment=solver->solve(system);
        To toAssignment;
        for(auto v:system.get_variables())
            toAssignment[v]=fromAssignment[v];
        return toAssignment;
    }

};

#endif //MPGCPP_SOLVERCONVERTER_H
