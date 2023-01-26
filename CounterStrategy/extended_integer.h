//
// Created by ramizouari on 24/01/23.
//

#ifndef COUNTERSTRATEGY_EXTENDED_INTEGER_H
#define COUNTERSTRATEGY_EXTENDED_INTEGER_H

#include <variant>
#include <ostream>
#include "types.h"
#include "operation.h"


namespace policy
{
/*
 * Whether to throw an exception for inf-inf scenarios, by default it is set to false
 * */
    inline bool throw_on_inf=false;
}

/**
 * @brief It is a class representing an infinite object
 * */
struct inf_t : public std::monostate
{
    virtual std::strong_ordering operator<=>(const inf_t&) const = default;
    friend std::ostream& operator<<(std::ostream& os, const inf_t& inf)
    {
        return os << "inf";
    }
};

/**
 * @brief A class representing positive infinities
 * */
struct inf_plus_t : public inf_t
{
    /*
 * All negative positive are equivalent to each other
 * */
    std::strong_ordering operator<=>(const inf_plus_t&) const = default;
    /*
     * Positive infinity is greater than any integer
     * */
    constexpr std::strong_ordering operator<=>(const integer &O) const
    {
        return std::strong_ordering::greater;
    }
    /*
     * A positive infinity is greater than negative infinity
     * */
    constexpr std::strong_ordering operator<=>(const class inf_minus_t&)
    {
        return std::strong_ordering::greater;
    }

    friend std::ostream& operator<<(std::ostream& os, const inf_plus_t& inf)
    {
        return os << '+' << static_cast<const inf_t&>(inf);
    }

};

/**
 * @brief A class representing negative infinities
 * */
struct inf_minus_t : public inf_t
{
    /*
     * All negative infinities are equivalent to each other
     * */
    std::strong_ordering operator<=>(const inf_minus_t&) const = default;
    /*
     * Negative infinity is less than any integer
     * */
    constexpr std::strong_ordering operator<=>(const integer &O) const
    {
        return std::strong_ordering::less;
    }
    /*
     * Negative infinity is less than positive infinity
     * */
    constexpr std::strong_ordering operator<=>(const class inf_plus_t&)
    {
        return std::strong_ordering::less;
    }

    friend std::ostream& operator<<(std::ostream& os, const inf_minus_t& inf)
    {
        return os << '-' << static_cast<const inf_t&>(inf);
    }

};

constexpr inf_plus_t operator+(const inf_t &)
{
    return {};
}
constexpr inf_minus_t operator-(const inf_t&)
{
    return {};
}

// Unsigned infinity
[[maybe_unused]] inline constexpr inf_t inf;
// A positive infinity
[[maybe_unused]] inline constexpr inf_plus_t inf_p;
// A negative infinity
[[maybe_unused]] inline constexpr inf_minus_t inf_m;

constexpr inf_plus_t operator+(const inf_plus_t&, const inf_plus_t&)
{
    return {};
}

integer operator-(const inf_plus_t&, const inf_plus_t&)
{
    if(policy::throw_on_inf)
        throw std::overflow_error("inf - inf");
    return 0;
}

constexpr inf_plus_t operator+=(const inf_plus_t& a, const inf_plus_t&)
{
    return {};
}

inf_plus_t operator-=(const inf_plus_t& a, const inf_plus_t&)
{
    throw std::overflow_error("inf - inf");
}


constexpr inf_minus_t operator+(const inf_minus_t&, const inf_minus_t&)
{
    return {};
}

integer operator-(const inf_minus_t&, const inf_minus_t&)
{
    if(policy::throw_on_inf)
        throw std::overflow_error("inf - inf");
    return 0;
}

constexpr inf_minus_t operator+=(const inf_minus_t& a, const inf_minus_t&)
{
    return {};
}

inf_minus_t operator-=(const inf_minus_t& a, const inf_minus_t&)
{
    throw std::overflow_error("inf - inf");
}

constexpr inf_plus_t operator-(const inf_minus_t&)
{
    return {};
}

constexpr inf_minus_t operator-(const inf_plus_t&)
{
    return {};
}

inf_plus_t operator+(const inf_plus_t&, const inf_minus_t&)
{
    if(policy::throw_on_inf)
        throw std::overflow_error("inf + -inf");
    else
        return {};
}

constexpr inf_plus_t operator-(const inf_plus_t&, const inf_minus_t&)
{
    return {};
}

inf_plus_t operator+(const inf_minus_t&, const inf_plus_t&)
{
    if(policy::throw_on_inf)
        throw std::overflow_error("inf + -inf");
    else
        return {};
}

constexpr inf_minus_t operator-(const inf_minus_t&, const inf_plus_t&)
{
    return {};
}

/*
 * Operation on integers
 * */

constexpr inf_plus_t operator+(const inf_plus_t&, const integer& a)
{
    return {};
}

constexpr inf_plus_t operator-(const integer&, const inf_plus_t& a)
{
    return {};
}

constexpr inf_plus_t operator+(const integer&, const inf_plus_t& a)
{
    return {};
}

constexpr inf_plus_t operator-(const inf_plus_t&, const integer& a)
{
    return {};
}

constexpr inf_plus_t operator+=(const inf_plus_t& a, const integer& b)
{
    return {};
}
constexpr inf_plus_t operator-=(const inf_plus_t& a, const integer& b)
{
    return {};
}

constexpr inf_minus_t operator+(const inf_minus_t&, const integer& a)
{
    return {};
}

constexpr inf_minus_t operator-(const integer&, const inf_minus_t& a)
{
    return {};
}

constexpr inf_minus_t operator+(const integer&, const inf_minus_t& a)
{
    return {};
}

constexpr inf_minus_t operator-(const inf_minus_t&, const integer& a)
{
    return {};
}

constexpr inf_minus_t operator+=(const inf_minus_t& a, const integer& b)
{
    return {};
}
constexpr inf_minus_t operator-=(const inf_minus_t& a, const integer& b)
{
    return {};
}



/**
 * @brief This is the class of integers including both infinities {±∞}
 * @details It represents the set of extended integers, on which addition, subtraction, multiplication, is defined adequately.
 * @param threshold which represents the threshold from which results are rounded to ±∞
 * */
struct ExtendedInteger: public std::variant<inf_minus_t,integer,inf_plus_t>
{
    using variant::variant;
    enum ValueType : int
    {
        INF_MINUS=-1,
        INTEGER=0,
        INF_PLUS=1
    };

    inline static integer threshold = 1e15;
    ExtendedInteger(): ExtendedInteger(0){} // default constructor

    ExtendedInteger& operator+=(const ExtendedInteger& other)
    {
        *this= std::visit([&other](auto &x,auto &y)->ExtendedInteger{
            auto result = x+y;
            if(result > threshold)
                return inf_plus_t{};
            else if(result < -threshold)
                return inf_minus_t{};
            else
                return result;
        },*this,other);
        return *this;
    }

    ExtendedInteger& operator-=(const ExtendedInteger& other)
    {
        *this= std::visit([&other](auto &x,auto &y)->ExtendedInteger{
            auto result = x-y;
            if(result > threshold)
                return inf_plus_t{};
            else if(result < -threshold)
                return inf_minus_t{};
            else
                return result;
        },*this,other);
        return *this;
    }

    ExtendedInteger operator+(const ExtendedInteger& other) const
    {
        ExtendedInteger result(*this);
        result += other;
        return result;
    }
    ExtendedInteger operator-(const ExtendedInteger& other) const
    {
        ExtendedInteger result(*this);
        result -= other;
        return result;
    }

    ExtendedInteger operator-() const
    {
        return std::visit([](auto &x)->ExtendedInteger{
            return -x;
        },*this);
    }

    friend std::ostream  &operator<<(std::ostream &os,const ExtendedInteger &a)
    {
        std::visit([&os](auto &x){
            os << x;
        },a);
        return os;
    }
};

template<>
struct Max<ExtendedInteger>
{
    [[nodiscard]] ExtendedInteger reduce(const ExtendedInteger& u,const ExtendedInteger& v) const
    {
        return std::max(u,v);
    }
    inline static ExtendedInteger identity{inf_minus_t{}};
};

template<>
struct Min<ExtendedInteger>
{
    [[nodiscard]] ExtendedInteger reduce(const ExtendedInteger& u,const ExtendedInteger& v) const
    {
        return std::min(u,v);
    }
    inline static ExtendedInteger identity{inf_plus_t{}};
};
#endif //COUNTERSTRATEGY_EXTENDED_INTEGER_H
