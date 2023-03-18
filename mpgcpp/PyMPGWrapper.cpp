#include <boost/python.hpp>
#include <iostream>
#include "game/MeanPayoffGame.h"
#include "mpgio/MPGReader.h"

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

std::vector<std::tuple<int,int,int>> from_python_edges(const py::list &l)
{
    std::vector<std::tuple<int,int,int>> v;
    for (int i=0; i<py::len(l); i++)
    {
        py::tuple t=py::extract<py::tuple>(l[i]);
        v.emplace_back(py::extract<int>(t[0]),py::extract<int>(t[1]),py::extract<int>(t[2]));
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

py::tuple optimal_strategy_pair_edges(const py::list &_edges)
{
    std::vector<std::tuple<int,int,int>> edges=from_python_edges(_edges);
    int n=0;
    for (auto [u,v,w] : edges) {
        n = std::max(n, u+1);
        n = std::max(n, v+1);
    }
    Implementation::HashMap::MeanPayoffGame<integer> mpg(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

py::tuple optimal_strategy_pair_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<integer>>(filename);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto [strategy0, strategy1] = optimal_strategy_pair(mpg, solver);
    return to_python_pair_strategies(strategy0, strategy1);
}

py::tuple winners_edges(const py::list &_edges)
{
    std::vector<std::tuple<int,int,int>> edges=from_python_edges(_edges);
    int n=0;
    for (auto [u,v,w] : edges) {
        n = std::max(n, u+1);
        n = std::max(n, v+1);
    }
    Implementation::HashMap::MeanPayoffGame<integer> mpg(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [winners0, winners1] = winners(mpg, strategyPair);
    std::vector<bool> winners0_bool(winners0.begin(), winners0.end());
    std::vector<bool> winners1_bool(winners1.begin(), winners1.end());
    return to_python_winners(winners0_bool, winners1_bool);
}

py::tuple winners_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<integer>>(filename);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [winners0, winners1] = winners(mpg, strategyPair);
    std::vector<bool> winners0_bool(winners0.begin(), winners0.end());
    std::vector<bool> winners1_bool(winners1.begin(), winners1.end());
    return to_python_winners(winners0_bool, winners1_bool);
}

py::tuple mean_payoffs_edges(const py::list &_edges)
{
    std::vector<std::tuple<int,int,int>> edges=from_python_edges(_edges);
    int n=0;
    for (auto [u,v,w] : edges) {
        n = std::max(n, u+1);
        n = std::max(n, v+1);
    }
    Implementation::HashMap::MeanPayoffGame<integer> mpg(n);
    for (auto [u,v,w] : edges)
        mpg.add_edge(u,v,w);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

py::tuple mean_payoffs_file(const std::string &filename)
{
    MPG auto mpg=mpg_from_file<Implementation::HashMap::MeanPayoffGame<integer>>(filename);
    Implementation::Vector::MaxAtomSystemSolver<integer> solver;
    auto strategyPair = optimal_strategy_pair(mpg, solver);
    auto [meanPayoff0, meanPayoff1] = mean_payoffs(mpg, strategyPair);
    return to_python_mean_payoffs(meanPayoff0, meanPayoff1);
}

BOOST_PYTHON_MODULE(mpgcpp)
{
    using namespace boost::python;
    def("optimal_strategy_pair_edges_cxx", optimal_strategy_pair_edges);
    def("optimal_strategy_pair_file_cxx", optimal_strategy_pair_file);
    def("winners_edges_cxx", winners_edges);
    def("winners_file_cxx", winners_file);
    def("mean_payoffs_edges_cxx", mean_payoffs_edges);
    def("mean_payoffs_file_cxx", mean_payoffs_file);
}
