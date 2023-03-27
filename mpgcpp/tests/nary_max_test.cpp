//
// Created by ramizouari on 27/03/23.
//
#include "gtest/gtest.h"
#include "csp/MaxAtomSystem.h"

TEST(UnsatisfiableSystemsTest, UNSAT_1) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,0,0,-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf}};
    ASSERT_EQ(assignment, expected);
}

TEST(UnsatisfiableSystemsTest, UNSAT_2) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,0,0,-1);
    system.add_constraint(0,0,0,1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf}};
    ASSERT_EQ(assignment, expected);
}

TEST(UnsatisfiableSystemsTest, UNSAT_3) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,0,0,-1);
    system.add_constraint(0,0,0,1);
    system.add_constraint(0,0,0,0);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf}};
    ASSERT_EQ(assignment, expected);
}
