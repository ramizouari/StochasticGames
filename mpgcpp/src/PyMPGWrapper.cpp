#include <boost/python.hpp>
#include <iostream>
#include "game/MeanPayoffGame.h"
#include "mpgio/MPGReader.h"
#include "csp/MaxAtomSolver.h"
#include <boost/mpl/for_each.hpp>
#include <boost/mpl/list.hpp>
#include <boost/type_index.hpp>

namespace py = boost::python;
//TO list
//template<typename T>
//py::list to_list(const std::vector<T> &v)
//{
//    py::list l;
//    for (auto &e : v)
//        l.append(e);
//    return l;
//}
//
////FROM list
//
//template<typename T>
//concept ListType = requires(T t,int k)
//{
//    typename T::value_type;
//    typename T::iterator;
//    {t[k]} -> std::convertible_to<typename T::value_type>;
//    {t.size()} -> std::convertible_to<int>;
//    {t.begin()} -> std::convertible_to<typename T::iterator>;
//    {t.end()} -> std::convertible_to<typename T::iterator>;
//};
//
//template<typename T,size_t k>
//struct TupleK_t : std::conditional<k==0 || ( TupleK_t<T,k-1>::value) &&
//        std::is_convertible<
//            decltype(std::get<k>(std::declval<T>())),
//            typename std::tuple_element<k,T>::type>::value,
//        std::true_type ,
//        std::false_type>
//{};
//
//template<typename T>
//concept Tuple = TupleK_t<T,std::tuple_size<T>::value>::value;
//
//template<typename A...>
//std::tuple<A...> from_tuple(const py::tuple &t)
//{
//    return std::make_tuple(py::extract<A>(t[i])...);
//}
//
//template<typename T>
//std::vector<T> from_list(const py::list &l)
//{
//    std::vector<T> v;
//    for (int i=0; i<py::len(l); i++)
//        v.push_back(py::extract<T>(l[i]));
//    return v;
//}
//template <ListType L>
//std::vector<L> from_list(const py::list &l)
//{
//    std::vector<L> v;
//    for (int i=0; i<py::len(l); i++)
//        v.push_back(from_list<L>(py::extract<py::list>(l[i])));
//    return v;
//}
//
//template <Tuple T>
//std::vector<T> from_list(const py::list &l)
//{
//    std::vector<T> v;
//    for (int i=0; i<py::len(l); i++)
//        v.push_back(from_list<T>(py::extract<py::list>(l[i])));
//    return v;
//}

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
    Implementation::HashMap::MeanPayoffGame<R> mpg(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

template<typename R>
py::tuple optimal_strategy_pair_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
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
    Implementation::HashMap::MeanPayoffGame<R> mpg(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [winners0, winners1] = winners(mpg, strategyPair);
    std::vector<bool> winners0_bool(winners0.begin(), winners0.end());
    std::vector<bool> winners1_bool(winners1.begin(), winners1.end());
    return to_python_winners(winners0_bool, winners1_bool);
}

template<typename R>
py::tuple winners_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
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
    Implementation::HashMap::MeanPayoffGame<R> mpg(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

template<typename R>
py::tuple mean_payoffs_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<R>>(filename);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

template<typename R>
py::list arc_consistency(const py::list &_system)
{
    auto system=from_max_atom_constraint<R>(_system);
    Implementation::Vector::MaxAtomSystemSolver<R> solver;
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
        def(ospe.c_str(), optimal_strategy_pair_edges<R>);
        def(ospf.c_str(), optimal_strategy_pair_file<R>);
        def(we.c_str(), winners_edges<R>);
        def(wf.c_str(), winners_file<R>);
        def(mpe.c_str(), mean_payoffs_edges<R>);
        def(mpf.c_str(), mean_payoffs_file<R>);
        def(ac.c_str(), arc_consistency<R>);
    });
    def("optimal_strategy_pair_edges_cxx", optimal_strategy_pair_edges<integer>);
    def("optimal_strategy_pair_file_cxx", optimal_strategy_pair_file<integer>);
    def("winners_edges_cxx", winners_edges<integer>);
    def("winners_file_cxx", winners_file<integer>);
    def("mean_payoffs_edges_cxx", mean_payoffs_edges<integer>);
    def("mean_payoffs_file_cxx", mean_payoffs_file<integer>);
    def("arc_consistency_cxx", arc_consistency<integer>);
}
