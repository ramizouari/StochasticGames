//
// Created by ramizouari on 27/03/23.
//
#include <bitset>
#include <random>
#include "gtest/gtest.h"
#include "csp/MaxAtomSystem.h"
#include "csp/MaxAtomSolver.h"
#include "utils.h"
#include <unordered_set>

constexpr integer N_VARIABLES_LARGE=1000;
constexpr integer N_VARIABLES_MEDIUM=100;
constexpr integer N_VARIABLES_SMALL=10;

TEST(UnsatisfiableSystemsTest, UNSAT_1) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{0,0},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsTest, UNSAT_2) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{0},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsTest, UNSAT_3) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{0,1,2,3},-1);
    system.add_constraint(1,{0,1,2,3},-1);
    system.add_constraint(2,{0,1,2,3},-1);
    system.add_constraint(3,{0},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf},{1,-inf},{2,-inf},{3,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsTest, UNSAT_4) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{1,2},-1);
    system.add_constraint(1,{0,2},-1);
    system.add_constraint(2,{0,1},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf},{1,-inf},{2,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsTest, UNSAT_5) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{1,2},-1);
    system.add_constraint(1,{0,2},-1);
    system.add_constraint(2,{0,1},-1);
    system.add_constraint(3,{0,1,2},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<integer>>{{0,-inf},{1,-inf},{2,-inf},{3,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsTest, UNSAT_6) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    std::vector<Variable> vars;
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        vars.emplace_back(i);
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,vars,-1);
    auto assignment=solver.solve(system);
    auto expected=std::unordered_map<Variable,order_closure<integer>>();
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        expected[i]=-inf;
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

template<typename Container>
void test_sat(Container && assignment, const NaryMaxAtomSystem<integer> &system)
{
    for(const auto& C:system.get_constraints())
    {
        auto x=std::get<0>(C);
        auto Y=std::get<1>(C);
        auto w=std::get<2>(C);
        order_closure<integer> Z=-inf;
        for(const auto& y:Y)
            Z=std::max(Z,assignment[y]);
        ASSERT_TRUE(assignment[x]<=Z+w) << "Constraint " << x << "<= max" << Y   << "+" << w << " is violated";
    }
}


TEST(SatisfactionTest, SAT_1) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{0,1},-1);
    system.add_constraint(1,{0,1},-1);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_2) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{0,1},1);
    system.add_constraint(1,{0,1},2);
    system.add_constraint(2,{0,1},3);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_3) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    system.add_constraint(0,{0,1},1);
    system.add_constraint(1,{0,1},2);
    system.add_constraint(2,{0,1},3);
    system.add_constraint(3,{0,1},4);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}


TEST(SatisfactionTest, SAT_4) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    std::vector<Variable> vars;
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        vars.emplace_back(i);
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,vars,1);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}
TEST(SatisfactionTest, SAT_5) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    std::vector<Variable> vars;
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        vars.emplace_back(i);
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,vars,10);
    auto assignment=solver.solve(system);
    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_6) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    for(integer i=0;i<N_VARIABLES_SMALL;i++) for(int j=1;j<(1<<N_VARIABLES_SMALL);j++)
    {
        std::bitset<N_VARIABLES_SMALL> B(j);
        std::vector<Variable> vars;
        for(int k=0;k<N_VARIABLES_SMALL;k++)
            if(B[k])
                vars.emplace_back(k);
        system.add_constraint(i,vars,j);
    }
    auto assignment=solver.solve(system);
    std::cout  << "Assignment: " << assignment  << '\n';
    test_sat(assignment,system);
}

TEST(SatisfactionTest, SAT_7) {
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    for(integer i=0;i<N_VARIABLES_SMALL;i++) for(int j=1;j<(1<<N_VARIABLES_SMALL);j++)
        {
            std::bitset<N_VARIABLES_SMALL> B(j);
            std::vector<Variable> vars;
            for(int k=0;k<N_VARIABLES_SMALL;k++)
                if(B[k])
                    vars.emplace_back(k);
            if(j%2==0)
                system.add_constraint(i,vars,j);
            else
                system.add_constraint(i,vars,-j);
        }
    auto assignment=solver.solve(system);
    std::cout  << "Assignment: " << assignment  << '\n';
    test_sat(assignment,system);
}

struct SatisfiablePseudoRandomTestBase
{
    integer variables=N_VARIABLES_LARGE;
    std::mt19937_64 engine;
    std::uniform_int_distribution<integer> D;
    std::binomial_distribution<integer> B;
    SatisfiablePseudoRandomTestBase(integer R,double p): engine(0), D(-R,R), B(variables,p) {}
};

template<int R, int P>
struct SatisfiablePseudoRandomTest :public SatisfiablePseudoRandomTestBase, public ::testing::Test
{

    SatisfiablePseudoRandomTest(): SatisfiablePseudoRandomTestBase(R,0.01*P)
    {
    }

};

using SatisfiablePseudoRandomTest_10_1 = SatisfiablePseudoRandomTest<10,1>;
using SatisfiablePseudoRandomTest_10_10 = SatisfiablePseudoRandomTest<10,10>;
using SatisfiablePseudoRandomTest_10_20 = SatisfiablePseudoRandomTest<10,20>;
using SatisfiablePseudoRandomTest_10_50 = SatisfiablePseudoRandomTest<10,50>;
using SatisfiablePseudoRandomTest_10_100 = SatisfiablePseudoRandomTest<10,100>;



void pseudo_random_test(SatisfiablePseudoRandomTestBase & testBase)
{
    DefaultMaxAtomSolver<integer> solver;
    NaryMaxAtomSystem<integer> system;
    for(int i=0;i<testBase.variables;i++)
    {
        std::unordered_set<integer> vars=choice(testBase.variables,testBase.B(testBase.engine),testBase.engine);
        std::vector<Variable> V(vars.begin(),vars.end());
        if(V.empty())
            continue;
        system.add_constraint(i,V,testBase.D(testBase.engine));
    }
    auto assignment=solver.solve(system);
    std::cout  << "Assignment: " << assignment  << '\n';
    test_sat(assignment,system);
}

TEST_F(SatisfiablePseudoRandomTest_10_1, SAT_8)
{
    pseudo_random_test(*this);
}


TEST_F(SatisfiablePseudoRandomTest_10_10, SAT_9)
{
    pseudo_random_test(*this);
}


TEST_F(SatisfiablePseudoRandomTest_10_20, SAT_8)
{
    pseudo_random_test(*this);
}

TEST_F(SatisfiablePseudoRandomTest_10_50, SAT_10)
{
    pseudo_random_test(*this);
}

TEST_F(SatisfiablePseudoRandomTest_10_100, SAT_11)
{
    pseudo_random_test(*this);
}