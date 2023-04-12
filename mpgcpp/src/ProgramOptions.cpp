//
// Created by ramizouari on 16/03/23.
//

#include "ProgramOptions.h"


namespace Options
{
    std::istream &operator>>(std::istream &H, OutputFormat &format)
    {
        std::string s;
        H >> s;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        if (s == "csv")
            format = OutputFormat::CSV;
        else if (s == "json")
            format = OutputFormat::JSON;
        else
            throw std::invalid_argument("Invalid output format");
        return H;
    }

    std::istream &operator>>(std::istream &H, Mode &mode)
    {
        std::string s;
        H >> s;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        if (s == "discard")
            mode = Mode::DISCARD;
        else if (s == "resume")
            mode = Mode::RESUME;
        else
            throw std::invalid_argument("Invalid mode");
        return H;
    }

    std::istream &operator>>(std::istream &H, Verbosity &verbosity)
    {
        std::string s;
        H >> s;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        if (s == "none")
            verbosity = Verbosity::NONE;
        else if (s == "errors")
            verbosity = Verbosity::ERRORS;
        else if (s == "info")
            verbosity = Verbosity::INFO;
        else if (s == "debug")
            verbosity = Verbosity::DEBUG;
        else
            throw std::invalid_argument("Invalid verbosity");
        return H;
    }

    std::ostream &operator<<(std::ostream &H, OutputFormat format)
    {
        switch (format) {
            case OutputFormat::CSV:
                H << "CSV";
                break;
            case OutputFormat::JSON:
                H << "JSON";
                break;
        }
        return H;
    }

    std::ostream &operator<<(std::ostream &H, Mode mode)
    {
        switch (mode) {
            case Mode::DISCARD:
                H << "DISCARD";
                break;
            case Mode::RESUME:
                H << "RESUME";
                break;
        }
        return H;
    }

    std::ostream &operator<<(std::ostream &H, Verbosity verbosity)
    {
        switch (verbosity) {
            case Verbosity::NONE:
                H << "NONE";
                break;
            case Verbosity::ERRORS:
                H << "ERRORS";
                break;
            case Verbosity::INFO:
                H << "INFO";
                break;
            case Verbosity::DEBUG:
                H << "DEBUG";
                break;
        }
        return H;
    }


    std::istream &operator>>(std::istream &H, GraphImplementation &impl)
    {
        std::string s;
        H >> s;
        s = std::regex_replace(s, std::regex("-"), "_");
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        if (s == "adjacency_list_vector" || s == "alv" || s == "vector")
            impl = GraphImplementation::ADJACENCY_LIST_VECTOR;
        else if (s == "adjacency_list_tree" || s == "alt" || s == "tree" || s == "map" || s == "ordered_map")
            impl = GraphImplementation::ADJACENCY_LIST_TREE;
        else if (s == "adjacency_list_hash" || s == "alh" || s == "hash" || s == "unordered_map" || s == "hashmap" ||
                 s == "hash_map")
            impl = GraphImplementation::ADJACENCY_LIST_HASH;
        else if (s == "adjacency_matrix" || s == "am" || s == "matrix")
            impl = GraphImplementation::ADJACENCY_MATRIX;
        else
            throw std::invalid_argument("Invalid graph implementation");
        return H;
    }

    std::ostream &operator<<(std::ostream &H, GraphImplementation impl)
    {
        switch (impl) {
            case GraphImplementation::ADJACENCY_LIST_VECTOR:
                H << "adjacency_list_vector";
                break;
            case GraphImplementation::ADJACENCY_LIST_TREE:
                H << "adjacency_list_tree";
                break;
            case GraphImplementation::ADJACENCY_LIST_HASH:
                H << "adjacency_list_hash";
                break;
            case GraphImplementation::ADJACENCY_MATRIX:
                H << "adjacency_matrix";
                break;
        }
        return H;
    }

    std::istream &operator>>(std::istream &H, SolverImplementation &impl)
    {
        std::string s;
        H >> s;
        s = std::regex_replace(s, std::regex("-"), "_");
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
        if (s == "arc_consistency" || s == "ac")
            impl = SolverImplementation::ARC_CONSISTENCY;
        else if (s == "fixed_point" || s == "fp")
            impl = SolverImplementation::FIXED_POINT;
        else
            throw std::invalid_argument("Invalid solver implementation");
        return H;
    }

    std::ostream &operator<<(std::ostream &H, SolverImplementation impl)
    {
        switch (impl) {
            case SolverImplementation::ARC_CONSISTENCY:
                H << "arc_consistency";
                break;
            case SolverImplementation::FIXED_POINT:
                H << "fixed_point";
                break;
        }
        return H;
    }

    std::strong_ordering operator<=>(Verbosity a, Verbosity b) {
        return static_cast<int>(a) <=> static_cast<int>(b);
    }

    std::ostream& operator<<(std::ostream& H,SolverHeuristic impl)
    {
        switch(impl)
        {
            case SolverHeuristic::NONE:
                H<<"none";
                break;
            case SolverHeuristic::DENSE:
                H<<"dense";
                break;
            case SolverHeuristic::SMALL_DOMAIN:
                H<<"small_domain";
                break;
            case SolverHeuristic::BOTH:
                H<<"both";
                break;
        }
        return H;
    }
    std::istream& operator>>(std::istream& H,SolverHeuristic& impl)
    {
        std::string s;
        H>>s;
        s=std::regex_replace(s,std::regex("-"),"_");
        std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return std::tolower(c);});
        if(s=="none")
            impl=SolverHeuristic::NONE;
        else if(s=="dense")
            impl=SolverHeuristic::DENSE;
        else if(s=="small_domain" || s=="small" || s=="domain" || s=="smalldomain" || s=="small domain")
            impl=SolverHeuristic::SMALL_DOMAIN;
        else if(s=="both")
            impl=SolverHeuristic::BOTH;
        else
            throw std::invalid_argument("Invalid solver heuristic");
        return H;
    }
}