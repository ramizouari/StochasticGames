//
// Created by ramizouari on 12/03/23.
//

#ifndef MPGCPP_INTRANGE_H
#define MPGCPP_INTRANGE_H

template<typename Int>
class IntRange
{
    Int _begin,_end;
    class iterator {
        Int _current;
    public:
        iterator(Int current):_current(current){}
        Int operator*() const
        {
            return _current;
        }
        iterator &operator++()
        {
            ++_current;
            return *this;
        }

        iterator operator++(int) const
        {
            iterator tmp(*this);
            operator++();
            return tmp;
        }

        bool operator!=(const iterator &other) const
        {
            return _current!=other._current;
        }
    };
public:
    IntRange(Int begin,Int end):_begin(begin),_end(end){}
    iterator begin() const
    {
        return _begin;
    }
    iterator end() const
    {
        return _end;
    }
};

#endif //MPGCPP_INTRANGE_H
