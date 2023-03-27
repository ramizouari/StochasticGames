//
// Created by ramizouari on 27/03/23.
//
#include <gtest/gtest.h>
#include "csp/MaxAtomSystem.h"
#include "csp/MaxAtomSolver.h"
#include <random>

class incremental_random_engine
{
    std::uint64_t state=0;
public:
    auto operator()() -> std::uint64_t
    {
        return state++;
    }
};

struct TestFixtureBase : public ::testing::Test {
    std::mt19937_64 engine;

};

struct UnsatisfiableSystemsDoubleTest : public TestFixtureBase {
};

struct SatisfactionDoubleTest : public TestFixtureBase {
};

// Demonstrate some basic assertions.
TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_1) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,0,0,-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf}};
    ASSERT_EQ(assignment, expected);
}

TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_2) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,0,0,-1);
    system.add_constraint(0,0,0,1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf}};
    ASSERT_EQ(assignment, expected);
}

TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_3) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,0,0,-1);
    system.add_constraint(0,0,0,1);
    system.add_constraint(0,0,0,0);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf}};
    ASSERT_EQ(assignment, expected);
}


TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_4) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,1,1,-1);
    system.add_constraint(1,0,0,-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf},{1,-inf}};
    ASSERT_EQ(assignment, expected);
}
TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_5) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,1,2,-1);
    system.add_constraint(1,0,2,-1);
    system.add_constraint(2,0,1,-1);

    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf},{1,-inf},{2,-inf}};
    ASSERT_EQ(assignment, expected);
}

constexpr int N_VARIABLES_LARGE=1000,N_VARIABLES_MEDIUM=100,N_VARIABLES_SMALL=10;


TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_6) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,(i+1)%N_VARIABLES_LARGE,(i+2)%N_VARIABLES_LARGE,-50);
    auto assignment=solver.solve(system);
    std::unordered_map<Variable,order_closure<double>> expected;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        expected[i]=-inf;
    ASSERT_EQ(assignment, expected);
}

TEST_F(UnsatisfiableSystemsDoubleTest, UNSAT_7) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++) for(int j=0;j<N_VARIABLES_LARGE;j++)
        system.add_constraint(i,j,j,0);
    system.add_constraint(0,0,0,-1);
    auto assignment=solver.solve(system);
    std::unordered_map<Variable,order_closure<double>> expected;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        expected[i]=-inf;
    ASSERT_EQ(assignment, expected);
}

constexpr double EPSILON=1e-6;

using Print::operator<<;

template<typename Container>
void test_double_sat(Container && assignment, const MaxAtomSystem<double> &system)
{
    for(const auto& C:system.get_constraints())
    {
        auto x=std::get<0>(C);
        auto y=std::get<1>(C);
        auto z=std::get<2>(C);
        auto w=std::get<3>(C);
        ASSERT_TRUE(assignment[x]<=assignment[y]+assignment[z]+w+EPSILON) << "Constraint " << x << "<=" << y << "+" << z << "+" << w << " violated";
    }
}

TEST_F(SatisfactionDoubleTest, SAT_1) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,0,0,0);
    auto assignment=solver.solve(system);

    test_double_sat(assignment,system);
}

TEST_F(SatisfactionDoubleTest, SAT_2) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,0,0,0);
    system.add_constraint(0,0,0,0.5);
    auto assignment=solver.solve(system);

    test_double_sat(assignment,system);
}

TEST_F(SatisfactionDoubleTest, SAT_3) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,0,0,0);
    system.add_constraint(0,0,0,0.2);
    system.add_constraint(0,0,0,-0.01);
    auto assignment=solver.solve(system);

    test_double_sat(assignment,system);
}

TEST_F(SatisfactionDoubleTest, SAT_4) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    system.add_constraint(0,1,1,1.25);
    system.add_constraint(1,0,0,0.2);
    auto assignment=solver.solve(system);

    test_double_sat(assignment,system);
}

TEST_F(SatisfactionDoubleTest, SAT_5) {
    std::uniform_real_distribution<double> random(-10,10);
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,(i+1)%N_VARIABLES_LARGE,(i+2)%N_VARIABLES_LARGE,random(engine));
    auto assignment=solver.solve(system);
    test_double_sat(assignment,system);
}

TEST_F(SatisfactionDoubleTest, SAT_6) {
    std::uniform_real_distribution<double> random(-10,10);
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    MaxAtomSystem<double> system;
    for(int i=0;i<N_VARIABLES_MEDIUM;i++) for(int j=0;j<N_VARIABLES_MEDIUM;j++)
        system.add_constraint(i,j,j,random(engine));
    auto assignment=solver.solve(system);
    test_double_sat(assignment,system);
}
