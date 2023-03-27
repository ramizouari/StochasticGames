//
// Created by ramizouari on 27/03/23.
//
#include <gtest/gtest.h>
#include "csp/MaxAtomSystem.h"
#include "csp/MaxAtomSolver.h"

using Print::operator<<;

// Demonstrate some basic assertions.
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


TEST(UnsatisfiableSystemsTest, UNSAT_4) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,1,1,-1);
    system.add_constraint(1,0,0,-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf},{1,-inf}};
    ASSERT_EQ(assignment, expected);
}
TEST(UnsatisfiableSystemsTest, UNSAT_5) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,1,2,-1);
    system.add_constraint(1,0,2,-1);
    system.add_constraint(2,0,1,-1);

    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf},{1,-inf},{2,-inf}};
    ASSERT_EQ(assignment, expected);
}

constexpr int N_VARIABLES_LARGE=1000;

TEST(UnsatisfiableSystemsTest, UNSAT_6) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,(i+1)%N_VARIABLES_LARGE,(i+2)%N_VARIABLES_LARGE,-50);
    auto assignment=solver.solve(system);
    std::unordered_map<Variable,order_closure<integer>> expected;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        expected[i]=-inf;
    ASSERT_EQ(assignment, expected);
}

TEST(UnsatisfiableSystemsTest, UNSAT_7) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++) for(int j=0;j<N_VARIABLES_LARGE;j++)
        system.add_constraint(i,j,j,0);
    system.add_constraint(0,0,0,-1);
    auto assignment=solver.solve(system);
    std::unordered_map<Variable,order_closure<integer>> expected;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        expected[i]=-inf;
    ASSERT_EQ(assignment, expected);
}

template<typename Container>
void test_sat(Container && assignment, const MaxAtomSystem<integer> &system)
{
    for(const auto& C:system.get_constraints())
    {
        auto x=std::get<0>(C);
        auto y=std::get<1>(C);
        auto z=std::get<2>(C);
        auto w=std::get<3>(C);
        ASSERT_TRUE(assignment[x]<=assignment[y]+assignment[z]+w) << "Constraint " << x << "<=" << y << "+" << z << "+" << w << " is violated";
    }
}

TEST(SatisfactionTest, SAT_1) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,0,0,0);
    auto assignment=solver.solve(system);

    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_2) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,0,0,0);
    system.add_constraint(0,0,0,1);
    auto assignment=solver.solve(system);

    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_3) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,0,0,0);
    system.add_constraint(0,0,0,1);
    system.add_constraint(0,0,0,-1);
    auto assignment=solver.solve(system);

    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_4) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    system.add_constraint(0,1,1,1);
    system.add_constraint(1,0,0,1);
    auto assignment=solver.solve(system);

    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_5) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,(i+1)%N_VARIABLES_LARGE,(i+2)%N_VARIABLES_LARGE,50);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_6) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++) for(int j=0;j<N_VARIABLES_LARGE;j++)
        system.add_constraint(i,j,j,0);
    system.add_constraint(0,0,0,1);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_7) {
    Implementation::HashMap::MaxAtomSystemSolver<integer> solver;
    MaxAtomSystem<integer> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++) for(int j=0;j<N_VARIABLES_LARGE;j++)
            system.add_constraint(i,j,j,50);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}