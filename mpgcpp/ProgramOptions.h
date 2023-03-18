//
// Created by ramizouari on 16/03/23.
//

#ifndef MPGCPP_PROGRAMOPTIONS_H
#define MPGCPP_PROGRAMOPTIONS_H

#include <istream>
#include <ostream>
#include <regex>

namespace Options
{
    enum class OutputFormat
    {
        CSV,JSON
    };

    enum class Mode
    {
        DISCARD,RESUME
    };

    enum class Verbosity
    {
        NONE,ERRORS,INFO,DEBUG
    };

    enum class GraphImplementation
    {
        ADJACENCY_LIST_VECTOR,ADJACENCY_LIST_TREE,ADJACENCY_LIST_HASH,ADJACENCY_MATRIX
    };

    enum class GraphFormat
    {
        WEIGHTED_EDGE_LIST
    };

    enum class Compression
    {
        None=0,GZip,BZip2,LZMA
    };

    enum class SolverImplementation
    {
        ARC_CONSISTENCY,FIXED_POINT
    };

    std::strong_ordering operator<=>(Verbosity a,Verbosity b);


    std::istream& operator>>(std::istream& H,OutputFormat& format);
    std::istream& operator>>(std::istream& H,Mode& mode);
    std::istream& operator>>(std::istream& H,Verbosity& verbosity);
    std::ostream &operator<<(std::ostream &H,OutputFormat format);
    std::ostream &operator<<(std::ostream &H,Mode mode);
    std::ostream &operator<<(std::ostream &H,Verbosity verbosity);
    std::istream& operator>>(std::istream& H,GraphImplementation& impl);
    std::ostream &operator<<(std::ostream &H,GraphImplementation impl);
    std::istream& operator>>(std::istream& H,SolverImplementation& impl);
    std::ostream &operator<<(std::ostream &H,SolverImplementation impl);

    template<typename T>
    std::ostream &operator<<(std::ostream &H,const std::vector<T>& v)
    {
        H<<"[";
        for(auto it=v.begin();it!=v.end();it++)
        {
            H<<*it;
            if(it!=v.end()-1)
                H<<",";
        }
        H<<"]";
        return H;
    }
}


#endif //MPGCPP_PROGRAMOPTIONS_H
