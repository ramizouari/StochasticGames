//
// Created by ramizouari on 28/03/23.
//
#include "utils.h"

std::unordered_set<integer> choice(int n, int k, std::mt19937_64& engine)
{
    std::unordered_set<integer> result;
    std::uniform_int_distribution<integer> d(0,n-1);
    while(result.size()<k)
        result.insert(d(engine));
    return result;
}