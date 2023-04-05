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
#include "csp/solver/DenseSolver.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

template<typename R>
using MPGInstance_t=Implementation::HashMap::MeanPayoffGame<R>;

template<typename R>
using MPGSolver_t = Implementation::Vector::MaxAtomSystemSolver<R>;

template<typename R>
using MPGSolverBase=  MaxAtomSolver<std::vector<order_closure<R>>,R>;

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
std::unique_ptr<MPGSolverBase<R>> MPGSolverInstance(const std::string &heuristic)
{
    auto& h=heuristic;
    if (h=="normal")
        return std::make_unique<MPGSolver_t<R>>();
    else if(h=="dense")
    {
        auto s= std::make_unique<Implementation::Vector::DenseSolver<R>>();
        s->set_bound_estimator(new LinearMaxAtomBoundEstimator<R>(2,20));
        return s;
    }
    else if(h=="small-bound")
    {
        auto s= std::make_unique<MPGSolver_t<R>>();
        s->set_bound_estimator(new LinearMaxAtomBoundEstimator<R>(2,10));
        return s;
    }
    else if(h=="parallel")
        return std::make_unique<Implementation::Parallel::Vector::MaxAtomSystemSolver<R>>();
    else
        throw std::runtime_error("Unknown heuristic \"" + heuristic + "\"");
}

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
py::tuple optimal_strategy_pair_edges(const py::list &_edges, const std::string &heuristic)
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
    auto solver = MPGSolverInstance<R>(heuristic);
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, *solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

template<typename R>
py::tuple optimal_strategy_pair_file(const std::string &filename, const std::string & heuristic)
{
    MPG auto mpg=mpg_from_file<MPGInstance_t<R>>(filename);
    auto solver = MPGSolverInstance<R>(heuristic);
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, *solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

template<typename R>
py::tuple winners_edges(const py::list &_edges, const std::string& heuristic)
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
    auto solver = MPGSolverInstance<R>(heuristic);
    auto strategyPair = optimal_strategy_pair(mpg, *solver);
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
int winners_tensorflow_matrix_flattened(const py::list &_data, const std::string& heuristic)
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
    auto solver = MPGSolverInstance<R>(heuristic);
    auto strategyPair = optimal_strategy_pair(mpg, *solver);
    auto W = winners(mpg, strategyPair);
    return static_cast<int>(W[turn][vertex]);
}

template<typename T>
class ListView3D
{
protected:
    const py::list &l;
public:
    ListView3D(const py::list &_l) : l(_l) {}
    virtual T operator()(ssize_t i,ssize_t j, ssize_t k) const
    {
        py::list z=py::extract<py::list>(l[i]);
        py::list y=py::extract<py::list>(z[j]);
        return py::extract<T>(y[k]);
    }
    [[nodiscard]] virtual std::array<ssize_t,3> shape() const
    {
        py::list z=py::extract<py::list>(l[0]);
        return {
                py::len(l),
                py::len(z),
                py::len(py::extract<py::list>(z[0]))
            };
    }
};

template<typename T>
class ListView3DFlat : public ListView3D<T>
{
    ssize_t n,m,r;
public:
    ListView3DFlat(const py::list &_l, ssize_t _n, ssize_t _m, ssize_t _r) : ListView3D<T>(_l), n(_n),m(_m),r(_r) {}
    T operator()(ssize_t i,ssize_t j, ssize_t k) const override
    {
        return py::extract<T>(this->l[i*n*m+j*m+k]);
    }
    [[nodiscard]] std::array<ssize_t,3> shape() const override
    {
        return {n, m,r};
    }
};

template<typename R>
py::list targets_tensorflow_matrix_base(const ListView3D<R> &_data, const std::string& heuristic, const std::string &target)
{
//    auto L=py::len(_data);
//    if(2*n*n != L)
//        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)) + " for n=" + std::to_string(n));

    auto [r,n,_] = _data.shape();

    auto mpg = MPGInstance<R>(n);
    std::vector<bool> empty(n,true);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) if(_data(0,i,j)!=0)
    {
        mpg.add_edge(i, j,_data(1,i,j));
        empty[i]=false;
    }

    std::string empty_str;
    for(int i=0;i<n;i++) if(empty[i]) empty_str+=std::to_string(i)+" ";
    if(!empty_str.empty()) throw std::runtime_error("Empty vertices: "+empty_str);
    auto solver = MPGSolverInstance<R>(heuristic);
    auto strategyPair = optimal_strategy_pair(mpg, *solver);
    py::list C;
    if(target=="strategy")
    {
        auto [strategy0, strategy1] = optimal_strategy_pair(mpg, *solver);
        auto X=to_list(strategy0);
        auto Y=to_list(strategy1);
        C.append(X);
        C.append(Y);
    }
    else if(target=="winners")
    {
        auto W = winners(mpg, strategyPair);
        std::vector<bool> W0(W[Bipartite::PLAYER_0].begin(), W[Bipartite::PLAYER_0].end());
        std::vector<bool> W1(W[Bipartite::PLAYER_1].begin(), W[Bipartite::PLAYER_1].end());
        auto X=to_list(W0);
        auto Y=to_list(W1);
        C.append(X);
        C.append(Y);
    }
    else if(target=="all")
    {
        auto [strategy0, strategy1] = optimal_strategy_pair(mpg, *solver);
        auto W = winners(mpg, strategyPair);
        std::vector<bool> W0(W[Bipartite::PLAYER_0].begin(), W[Bipartite::PLAYER_0].end());
        std::vector<bool> W1(W[Bipartite::PLAYER_1].begin(), W[Bipartite::PLAYER_1].end());
        py::list A,B;
        auto S0=to_list(strategy0);
        auto S1=to_list(strategy1);
        auto Z0=to_list(W0);
        auto Z1=to_list(W1);
        A.append(S0);
        A.append(S1);
        B.append(Z0);
        B.append(Z1);
        C.append(A);
        C.append(B);
    }
    else
        throw std::runtime_error("Invalid target " + target);
    return C;
}

template<typename R>
py::list targets_tensorflow_flattened(const py::list &_data, const std::string& heuristic, const std::string &target)
{
    auto L=py::len(_data);
    int n=std::sqrt(L/2);
    if(2*n*n != L)
        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)) + " for n=" + std::to_string(n));
    ListView3DFlat<R> data(_data,2,n,n);
    return targets_tensorflow_matrix_base(data, heuristic, target);
}

template<typename R>
py::list targets_tensorflow_tensor(const py::list &_data, const std::string& heuristic, const std::string &target)
{
    auto L=py::len(_data);
    if(L!=2)
        throw std::runtime_error("Invalid data length " + std::to_string(py::len(_data)));
    ListView3D<R> data(_data);
    return targets_tensorflow_matrix_base(data, heuristic, target);
}

template<typename R>
py::list targets_tensorflow(const py::list &_data, const std::string& heuristic, const std::string &target, bool flatten)
{
    if(flatten)
        return targets_tensorflow_flattened<R>(_data, heuristic, target);
    else
        return targets_tensorflow_tensor<R>(_data, heuristic, target);
}

template<typename R>
py::tuple winners_file(const std::string &filename, const std::string & heuristic)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    auto solver = MPGSolverInstance<R>(heuristic);
    auto strategyPair = optimal_strategy_pair(mpg, *solver);
    auto [winners0, winners1] = winners(mpg, strategyPair);
    std::vector<bool> winners0_bool(winners0.begin(), winners0.end());
    std::vector<bool> winners1_bool(winners1.begin(), winners1.end());
    return to_python_winners(winners0_bool, winners1_bool);
}

template<typename R>
py::tuple mean_payoffs_edges(const py::list &_edges, const std::string &heuristic)
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
    auto solver = MPGSolverInstance<R>(heuristic);
    auto strategyPair = optimal_strategy_pair(mpg, *solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

template<typename R>
py::tuple mean_payoffs_file(const std::string &filename, const std::string &heuristic)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    auto solver = MPGSolver<R>();
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

template<typename R>
py::list arc_consistency(const py::list &_system,const std::string& heuristic)
{
    using Map=std::vector<order_closure<R>>;
    auto system=from_max_atom_constraint<R>(_system);
    auto solver = MPGSolverInstance<R>(heuristic);


    auto solution = solver->solve(system);
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
        auto wtfl="winners_tensorflow_"+name+"_flattened_cxx";
        auto ach="arc_consistency_heuristic_"+name+"_cxx";
        auto ttf = "targets_tensorflow_"+name+"_cxx";
        auto ttft = "targets_tensorflow_"+name+"_tensor_cxx";
        auto ttff = "targets_tensorflow_"+name+"_flattened_cxx";
        def(ospe.c_str(), optimal_strategy_pair_edges<R>);
        def(ospf.c_str(), optimal_strategy_pair_file<R>);
        def(we.c_str(), winners_edges<R>);
        def(wf.c_str(), winners_file<R>);
        def(mpe.c_str(), mean_payoffs_edges<R>);
        def(mpf.c_str(), mean_payoffs_file<R>);
        def(ac.c_str(), arc_consistency<R>);
        def(wtfl.c_str(), targets_tensorflow_flattened<R>);
        def(ttf.c_str(), targets_tensorflow<R>);
        def(ttft.c_str(), targets_tensorflow_tensor<R>);
        def(ttff.c_str(), targets_tensorflow_flattened<R>);
    });
    def("optimal_strategy_pair_edges_cxx", optimal_strategy_pair_edges<integer>);
    def("optimal_strategy_pair_file_cxx", optimal_strategy_pair_file<integer>);
    def("winners_edges_cxx", winners_edges<integer>);
    def("winners_file_cxx", winners_file<integer>);
    def("mean_payoffs_edges_cxx", mean_payoffs_edges<integer>);
    def("mean_payoffs_file_cxx", mean_payoffs_file<integer>);
    def("arc_consistency_cxx", arc_consistency<integer>);
    def("targets_tensorflow_cxx", targets_tensorflow<integer>);
    def("targets_tensorflow_tensor_cxx", targets_tensorflow_tensor<integer>);
    def("targets_tensorflow_flattened_cxx", targets_tensorflow_flattened<integer>);
}
