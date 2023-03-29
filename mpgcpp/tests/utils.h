//
// Created by ramizouari on 28/03/23.
//

#ifndef MPGCPP_UTILS_H
#define MPGCPP_UTILS_H

#include "algebra/abstract_algebra.h"
#include "csp/Variable.h"
#include "csp/MaxAtomSolver.h"
#include <unordered_set>
#include <random>

template<typename R>
using DefaultMaxAtomSolver= Implementation::HashMap::MaxAtomSystemSolver<R>;

std::unordered_set<integer> choice(int n, int k, std::mt19937_64& engine);
using Print::operator<<;

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& c)
{
    os << "{";
    for(int i=0;i<c.size();i++)
        os << c[i] << (i==c.size()-1?"":", ");
    os << "}";
    return os;
}
template<typename K,typename T>
inline std::ostream& operator<<(std::ostream& os, const std::unordered_map<K,T>& c)
{
    os << "{";
    int i=0;
    for(auto it=c.begin();it!=c.end();++it)
    {
        auto [k,v]=*it;
        os << k << "->" << v << (i==c.size()-1?"":", ");
        i++;
    }

    os << "}";
    return os;
}

#endif //MPGCPP_UTILS_H
