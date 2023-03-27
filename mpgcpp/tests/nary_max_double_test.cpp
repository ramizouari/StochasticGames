//
// Created by ramizouari on 28/03/23.
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

TEST(UnsatisfiableSystemsDoubleTest, UNSAT_1) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{0,0},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsDoubleTest, UNSAT_2) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{0},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsDoubleTest, UNSAT_3) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{0,1,2,3},-1);
    system.add_constraint(1,{0,1,2,3},-1);
    system.add_constraint(2,{0,1,2,3},-1);
    system.add_constraint(3,{0},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf},{1,-inf},{2,-inf},{3,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsDoubleTest, UNSAT_4) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{1,2},-1);
    system.add_constraint(1,{0,2},-1);
    system.add_constraint(2,{0,1},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf},{1,-inf},{2,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsDoubleTest, UNSAT_5) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{1,2},-1);
    system.add_constraint(1,{0,2},-1);
    system.add_constraint(2,{0,1},-1);
    system.add_constraint(3,{0,1,2},-1);
    auto assignment=solver.solve(system);

    auto expected=std::unordered_map<Variable,order_closure<double>>{{0,-inf},{1,-inf},{2,-inf},{3,-inf}};
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

TEST(UnsatisfiableSystemsDoubleTest, UNSAT_6) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    std::vector<Variable> vars;
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        vars.emplace_back(i);
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,vars,-1);
    auto assignment=solver.solve(system);
    auto expected=std::unordered_map<Variable,order_closure<double>>();
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        expected[i]=-inf;
    ASSERT_EQ(assignment, expected) << "Expected Unsatisfiable System";
}

using Print::operator<<;

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& c)
{
    os << "{";
    for(int i=0;i<c.size();i++)
        os << c[i] << (i==c.size()-1?"":",");
    os << "}";
    return os;
}
template<typename K,typename T>
inline std::ostream& operator<<(std::ostream& os, const std::unordered_map<K,T>& c)
{
    os << "{";
    int k=0;
    for(auto it=c.begin();it!=c.end();++it)
    {
        k++;
        auto [k,v]=*it;
        os << k << "->" << v << (k==c.size()-1?"":",");
    }

    os << "}";
    return os;
}

constexpr double EPS=0.0000001;

template<typename Container>
void test_sat_approx(Container && assignment, const NaryMaxAtomSystem<double> &system)
{
    for(const auto& C:system.get_constraints())
    {
        auto x=std::get<0>(C);
        auto Y=std::get<1>(C);
        auto w=std::get<2>(C);
        order_closure<double> Z=-inf;
        for(const auto& y:Y)
            Z=std::max(Z,assignment[y]);
        ASSERT_TRUE(assignment[x]<=Z+w + EPS) << "Constraint " << x << "<= max" << Y   << "+" << w << " is violated";
    }
}


TEST(SatisfactionDoubleTest, SAT_1) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{0,1},-1);
    system.add_constraint(1,{0,1},-1);
    auto assignment=solver.solve(system);
    test_sat_approx(assignment,system);
}

TEST(SatisfactionDoubleTest, SAT_2) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{0,1},1);
    system.add_constraint(1,{0,1},2);
    system.add_constraint(2,{0,1},3);
    auto assignment=solver.solve(system);
    test_sat_approx(assignment,system);
}

TEST(SatisfactionDoubleTest, SAT_3) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    system.add_constraint(0,{0,1},1);
    system.add_constraint(1,{0,1},2);
    system.add_constraint(2,{0,1},3);
    system.add_constraint(3,{0,1},4);
    auto assignment=solver.solve(system);
    test_sat_approx(assignment,system);
}


TEST(SatisfactionDoubleTest, SAT_4) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    std::vector<Variable> vars;
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        vars.emplace_back(i);
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,vars,1);
    auto assignment=solver.solve(system);
    test_sat_approx(assignment,system);
}
TEST(SatisfactionDoubleTest, SAT_5) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
    std::vector<Variable> vars;
    for(integer i=0;i<N_VARIABLES_LARGE;i++)
        vars.emplace_back(i);
    for(int i=0;i<N_VARIABLES_LARGE;i++)
        system.add_constraint(i,vars,10);
    auto assignment=solver.solve(system);
    test_sat_approx(assignment,system);
}

TEST(SatisfactionDoubleTest, SAT_6) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
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
    test_sat_approx(assignment,system);
}

TEST(SatisfactionDoubleTest, SAT_7) {
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
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
    test_sat_approx(assignment,system);
}

struct SatisfiablePseudoRandomDoubleTestBase
{
    integer variables=N_VARIABLES_LARGE;
    std::mt19937_64 engine;
    std::uniform_real_distribution<double> D;
    std::binomial_distribution<integer> B;
    SatisfiablePseudoRandomDoubleTestBase(integer R,double p): engine(0), D(-R,R), B(variables,p) {}
};

template<int R, int P>
struct SatisfiablePseudoRandomDoubleTest :public SatisfiablePseudoRandomDoubleTestBase, public ::testing::Test
{

    SatisfiablePseudoRandomDoubleTest(): SatisfiablePseudoRandomDoubleTestBase(R,0.01*P)
    {
    }

};

using SatisfiablePseudoRandomDoubleTest_1 = SatisfiablePseudoRandomDoubleTest<1,1>;
using SatisfiablePseudoRandomDoubleTest_10 = SatisfiablePseudoRandomDoubleTest<1,10>;
using SatisfiablePseudoRandomDoubleTest_20 = SatisfiablePseudoRandomDoubleTest<1,20>;
using SatisfiablePseudoRandomDoubleTest_50 = SatisfiablePseudoRandomDoubleTest<1,50>;
using SatisfiablePseudoRandomDoubleTest_100 = SatisfiablePseudoRandomDoubleTest<1,100>;

void pseudo_random_test(SatisfiablePseudoRandomDoubleTestBase & testBase)
{
    Implementation::HashMap::MaxAtomSystemSolver<double> solver;
    NaryMaxAtomSystem<double> system;
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
    test_sat_approx(assignment,system);
}

TEST_F(SatisfiablePseudoRandomDoubleTest_1, SAT_8)
{
    pseudo_random_test(*this);
}


TEST_F(SatisfiablePseudoRandomDoubleTest_10, SAT_9)
{
    pseudo_random_test(*this);
}


TEST_F(SatisfiablePseudoRandomDoubleTest_20, SAT_8)
{
    pseudo_random_test(*this);
}

TEST_F(SatisfiablePseudoRandomDoubleTest_50, SAT_10)
{
    pseudo_random_test(*this);
}

TEST_F(SatisfiablePseudoRandomDoubleTest_100, SAT_11)
{
    pseudo_random_test(*this);
}