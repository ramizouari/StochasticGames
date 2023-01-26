//
// Created by ramizouari on 24/01/23.
//

#ifndef COUNTERSTRATEGY_OPERATION_H
#define COUNTERSTRATEGY_OPERATION_H


#include <algorithm>

/*
 * This header defines some binary operations one a set M
 * Also a concept name MonoidOperation is defined that should be used for classes expecting a binary operation * on a set M such that
 * (M,*) is a monoid
 * */

template<typename M>
struct BinaryOperation
{
    using base_type=M;
    virtual M reduce(const M& u,const M& v) const = 0;

    template<typename H0,typename ...H>
    M operator()(const H0& u,const H&... h) const
    {
        if constexpr ( sizeof...(h) > 0 )
            return reduce(u,(*this)(h...));
        else return u;
    }
    virtual ~BinaryOperation()= default;
};

template<typename M>
struct Multiplication : public BinaryOperation<M>
{
    M reduce(const M& u,const M& v) const override
    {
        return u*v;
    }
    inline static M identity = M(1);
};



template<typename M>
struct Addition : public BinaryOperation<M>
{
    M reduce(const M& u,const M& v) const override
    {
        return u+v;
    }
    inline static M identity{};
};

template<typename M>
struct Max
{
    M reduce(const M& u,const M& v) const
    {
        return std::max(u,v);
    }
    inline static M identity{};
};
template<typename M>
struct Min
{
    M reduce(const M& u,const M& v) const
    {
        return std::min(u,v);
    }
    inline static M identity{};
};


/**
 * @details This is a concept for any template that expects a binary operation * on a set S such that (S,*) is a monoid
 * @remark This concept only checks whether * is a binary operation, and whether an element named identity exists,
 * Whether * is associative or not, and whether the element "identity" is really the identity element, is left as a
 * responsibility to the developer.
 * */
template<typename T>
concept MonoidOperation = requires()
{
    typename T::base_type;
    {T::identity} -> std::same_as<typename T::base_type &>;
    requires requires(T op,typename T::base_type x,typename T::base_type y)
    {
        {op(x,y)} -> std::same_as<typename T::base_type>;
    };
};

#endif //COUNTERSTRATEGY_OPERATION_H
