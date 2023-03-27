//
// Created by ramizouari on 28/03/23.
//

#ifndef MPGCPP_UTILS_H
#define MPGCPP_UTILS_H

#include "algebra/abstract_algebra.h"
#include <unordered_set>
#include <random>

std::unordered_set<integer> choice(int n, int k, std::mt19937_64& engine);

#endif //MPGCPP_UTILS_H
