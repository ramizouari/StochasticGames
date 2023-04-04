#include <boost/python.hpp>
#include <iostream>
#include "game/MeanPayoffGame.h"
#include "mpgio/MPGReader.h"
#include "csp/MaxAtomSolver.h"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/list.hpp>
#include <boost/type_index.hpp>
#include <boost/python/numpy.hpp>
#include <fstream>
#include "csp/solver/ParallelMaxAtomSolver.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

template<typename R>
using MPGInstance_t=Implementation::Vector::MeanPayoffGame<R>;

template<typename R>
using MPGSolver_t = Implementation::Vector::MaxAtomSystemSolver<R>;

template<typename R>
MPGInstance_t<R> MPGInstance(int n)
{
    return MPGInstance_t<R>(n);
}

template<typename R>
MPGSolver_t<R> MPGSolver()
{
    return MPGSolver_t<R>();
};

template<typename R>
std::vector<std::tuple<int,int,R>> from_python_edges(const py::list &l)
{
    std::vector<std::tuple<int,int,R>> v;
    for (int i=0; i<py::len(l); i++)
    {
        py::tuple t=py::extract<py::tuple>(l[i]);
        v.emplace_back(py::extract<int>(t[0]),py::extract<int>(t[1]),py::extract<R>(t[2]));
    }
    return v;
}


template<typename R>
MaxAtomSystem<R> from_max_atom_constraint(const py::list &l)
{
    MaxAtomSystem<R> v;
    for (int i=0; i<py::len(l); i++)
    {
        py::tuple t=py::extract<py::tuple>(l[i]);
        v.add_constraint(Variable(py::extract<integer>(t[0])),
                         Variable(py::extract<integer>(t[1])),
                         Variable(py::extract<integer>(t[2])),
                         py::extract<R>(t[3]));
    }
    return v;
}
template<typename T>
py::list to_list(const std::vector<T> &v)
{
    py::list l;
    for (auto e : v)
        l.append(e);
    return l;
}

py::tuple to_python_pair_strategies(const std::pair<std::vector<int>, std::vector<int>> &p)
{
    return py::make_tuple(to_list(p.first),to_list(p.second));
}

py::tuple to_python_pair_strategies(const std::vector<int>&p1,const std::vector<int> &p2)
{
    return py::make_tuple(to_list(p1),to_list(p2));
}

py::tuple to_python_mean_payoffs(const std::pair<std::vector<double>, std::vector<double>> &p)
{
    return py::make_tuple(to_list(p.first),to_list(p.second));
}

py::tuple to_python_mean_payoffs(const std::vector<double>&p1,const std::vector<double> &p2)
{
    return py::make_tuple(to_list(p1),to_list(p2));
}

py::tuple to_python_winners(const std::pair<std::vector<bool>, std::vector<bool>> &p)
{
    return py::make_tuple(to_list(p.first),to_list(p.second));
}

py::tuple to_python_winners(const std::vector<bool>&p1,const std::vector<bool> &p2)
{
    return py::make_tuple(to_list(p1),to_list(p2));
}

template<typename R>
py::tuple optimal_strategy_pair_edges(const py::list &_edges)
{
    std::vector<std::tuple<int,int,R>> edges=from_python_edges<R>(_edges);
    int n=0;
    for (auto [u,v,w] : edges) {
        n = std::max(n, u+1);
        n = std::max(n, v+1);
    }
    auto mpg = MPGInstance<R>(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    auto solver = MPGSolver<R>();
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

template<typename R>
py::tuple optimal_strategy_pair_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<MPGInstance_t<R>>(filename);
    auto solver = MPGSolver<R>();
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

template<typename R>
py::tuple winners_edges(const py::list &_edges)
{
    std::vector<std::tuple<int,int,R>> edges=from_python_edges<R>(_edges);
    int n=0;
    for (auto [u,v,w] : edges) {
        n = std::max(n, u+1);
        n = std::max(n, v+1);
    }
    auto mpg = MPGInstance<R>(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    auto solver = MPGSolver<R>();
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [winners0, winners1] = winners(mpg, strategyPair);
    std::vector<bool> winners0_bool(winners0.begin(), winners0.end());
    std::vector<bool> winners1_bool(winners1.begin(), winners1.end());
    return to_python_winners(winners0_bool, winners1_bool);
}

/*
 * Length of the data is n²+2
 * First n elements define the adjacency matrix
 * Next n elements define the weights matrix
 * Before last element is the vertex to start from
 * Last element is the turn
 * */
template<typename R>
int winners_tensorflow_matrix_flattened(const py::list &_data)
{
    constexpr size_t VERTEX_POS=-2;
    constexpr size_t TURN_POS=-1;
    auto L=py::len(_data);
    if (L<3)
        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)));
    int n=(L-2)/2;
    n=std::round(std::sqrt(n));
    if(2*n*n+2 != L)
        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)) + " for n=" + std::to_string(n));
    auto turn = static_cast<Bipartite::Player>(py::extract<R>(_data[L+TURN_POS])==1);
    int vertex = py::extract<R>(_data[L+VERTEX_POS]);
    if(vertex<0 || vertex>=n)
        throw std::runtime_error("Invalid vertex " + std::to_string(vertex) + " for n=" + std::to_string(n));
    auto mpg = MPGInstance<R>(n);
    std::vector<bool> empty(n,true);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(py::extract<R>(_data[i*n+j])!=0)
    {
        mpg.add_edge(i, j, py::extract<R>(_data[i * n + j + n * n]));
        empty[i]=false;
    }
    std::string empty_str;
    for(int i=0;i<n;i++) if(empty[i]) empty_str+=std::to_string(i)+" ";
    if(!empty_str.empty()) throw std::runtime_error("Empty vertices: "+empty_str);
    auto solver = MPGSolver<R>();
    //solver.set_bound_estimator(new LinearMaxAtomBoundEstimator<R>(1,100));
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto W = winners(mpg, strategyPair);
    return static_cast<int>(W[turn][vertex]);
}

template<typename R>
int winners_tensorflow_matrix_flattened_patched(const py::list &_data)
{
    constexpr size_t VERTEX_POS=-2;
    constexpr size_t TURN_POS=-1;
    auto L=py::len(_data);
    if (L<3)
        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)));
    int n=(L-2)/2;
    n=std::round(std::sqrt(n));
    if(2*n*n+2 != L)
        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)) + " for n=" + std::to_string(n));
    auto turn = static_cast<Bipartite::Player>(py::extract<R>(_data[L+TURN_POS])==1);
    int vertex = py::extract<R>(_data[L+VERTEX_POS]);
    if(vertex<0 || vertex>=n)
        throw std::runtime_error("Invalid vertex " + std::to_string(vertex) + " for n=" + std::to_string(n));
    auto mpg = MPGInstance<R>(n);
    std::vector<bool> empty(n,true);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(py::extract<R>(_data[i*n+j])!=0)
            {
                mpg.add_edge(i, j, py::extract<R>(_data[i * n + j + n * n]));
                empty[i]=false;
            }
    std::string empty_str;
    for(int i=0;i<n;i++) if(empty[i]) empty_str+=std::to_string(i)+" ";
    if(!empty_str.empty()) throw std::runtime_error("Empty vertices: "+empty_str);
    auto solver = MPGSolver<R>();
    solver.set_bound_estimator(new LinearMaxAtomBoundEstimator<R>(1,100));
    return 1;
}

template<typename R>
py::tuple winners_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    auto solver = MPGSolver<R>();
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [winners0, winners1] = winners(mpg, strategyPair);
    std::vector<bool> winners0_bool(winners0.begin(), winners0.end());
    std::vector<bool> winners1_bool(winners1.begin(), winners1.end());
    return to_python_winners(winners0_bool, winners1_bool);
}

template<typename R>
py::tuple mean_payoffs_edges(const py::list &_edges)
{
    std::vector<std::tuple<int,int,R>> edges=from_python_edges<R>(_edges);
    int n=0;
    for (auto [u,v,w] : edges) {
        n = std::max(n, u+1);
        n = std::max(n, v+1);
    }
    auto mpg = MPGInstance<R>(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    auto solver= MPGSolver<R>();
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

template<typename R>
py::tuple mean_payoffs_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    auto solver = MPGSolver<R>();
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

template<typename R>
py::list arc_consistency(const py::list &_system)
{
    auto system=from_max_atom_constraint<R>(_system);
    auto solver = MPGSolver<R>();
    auto solution = solver.solve(system);
    py::list solution_python;
    for(auto value:solution)
    {
        if(value==inf_min)
            solution_python.append("-inf");
        else
            solution_python.append(std::get<R>(value));
    }
    return solution_python;
}

template<typename T>
struct named_type;

template<>
struct named_type<int>
{
    static constexpr const char* name = "int";
    using type=int;
};

template<>
struct named_type<float>
{
    static constexpr const char* name = "float";
    using type=float;
};

template<>
struct named_type<double>
{
    static constexpr const char* name = "double";
    using type=double;
};

template<>
struct named_type<integer>
{
    static constexpr const char* name = "integer";
    using type=integer;
};

using Types=boost::mpl::list<int,integer, float, double>;

BOOST_PYTHON_MODULE(mpgcpp)
{
    using namespace boost::python;
    boost::mpl::for_each<Types>([](auto arg){
        using R=decltype(arg);
        std::string name = named_type<R>::name;
        auto ospe="optimal_strategy_pair_"+name+"_edges_cxx";
        auto ospf="optimal_strategy_pair_"+name+"_file_cxx";
        auto we="winners_"+name+"_edges_cxx";
        auto wf="winners_"+name+"_file_cxx";
        auto mpe="mean_payoffs_"+name+"_edges_cxx";
        auto mpf="mean_payoffs_"+name+"_file_cxx";
        auto ac="arc_consistency_"+name+"_cxx";
        auto wtfl="winners_tensorflow_"+name+"_matrix_flattened_cxx";
        auto wtf="winners_tensorflow_"+name+"_matrix_flattened_patched_cxx";
        def(ospe.c_str(), optimal_strategy_pair_edges<R>);
        def(ospf.c_str(), optimal_strategy_pair_file<R>);
        def(we.c_str(), winners_edges<R>);
        def(wf.c_str(), winners_file<R>);
        def(mpe.c_str(), mean_payoffs_edges<R>);
        def(mpf.c_str(), mean_payoffs_file<R>);
        def(ac.c_str(), arc_consistency<R>);
        def(wtfl.c_str(), winners_tensorflow_matrix_flattened<R>);
        def(wtf.c_str(), winners_tensorflow_matrix_flattened_patched<R>);
    });
    def("optimal_strategy_pair_edges_cxx", optimal_strategy_pair_edges<integer>);
    def("optimal_strategy_pair_file_cxx", optimal_strategy_pair_file<integer>);
    def("winners_edges_cxx", winners_edges<integer>);
    def("winners_file_cxx", winners_file<integer>);
    def("mean_payoffs_edges_cxx", mean_payoffs_edges<integer>);
    def("mean_payoffs_file_cxx", mean_payoffs_file<integer>);
    def("arc_consistency_cxx", arc_consistency<integer>);
}
