//
// Created by ramizouari on 21/04/23.
//

#ifndef MPGCPP_VALUEITERATION_H
#define MPGCPP_VALUEITERATION_H

#include <vector>

namespace mpg::mdp
{

    template<typename Real>
    class ValueIteration
    {
        Real gamma,epsilon;
        std::vector<Real> V;
    public:
        ValueIteration(Real gamma,Real epsilon);
        //void iterate(const MarkovDecisionProcess<Real> &mdp);
        const std::vector<Real> &get_V() const;
    };
}
#endif //MPGCPP_VALUEITERATION_H
