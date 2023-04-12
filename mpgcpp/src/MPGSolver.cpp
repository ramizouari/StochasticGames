#include <iostream>
#include "game/MeanPayoffGame.h"
#include "csp/MinMaxSystem.h"
#include "mpgio/MPGReader.h"
#include "Writer.h"
#include "ProgramOptions.h"
#include <boost/program_options.hpp>
#include <filesystem>
#include <future>
#include <syncstream>
#include "concurrentqueue/concurrentqueue.h"
#include "csp/MaxAtomSolver.h"
#include "csp/solver/DenseSolver.h"

template <class cT, class traits = std::char_traits<cT> >
class basic_nullbuf: public std::basic_streambuf<cT, traits> {
    typename traits::int_type overflow(typename traits::int_type c)
    {
        return traits::not_eof(c); // indicate success
    }
};

template <class cT, class traits = std::char_traits<cT> >
class basic_onullstream: public std::basic_ostream<cT, traits> {
public:
    basic_onullstream():
            std::basic_ios<cT, traits>(&m_sbuf),
            std::basic_ostream<cT, traits>(&m_sbuf)
    {
        this->init(&m_sbuf);
    }

private:
    basic_nullbuf<cT, traits> m_sbuf;
};

typedef basic_onullstream<char> onullstream;
typedef basic_onullstream<wchar_t> wonullstream;
onullstream cnull;

std::ostream* get_output_stream(Options::Verbosity v)
{
    using Options::operator<=>;
    if (v < Options::Verbosity::INFO)
        return &cnull;
    else
        return &std::cout;
}

std::ostream* get_error_stream(Options::Verbosity v)
{
    using Options::operator<=>;
    if (v < Options::Verbosity::ERRORS)
        return &cnull;
    else
        return &std::cerr;
}

void process_game(const std::filesystem::path& path,Result::ParallelWriter& outputWriter, const boost::program_options::variables_map& vm)
{
    std::osyncstream synced_output(*get_output_stream(vm["verbose"].as<Options::Verbosity>()));
    std::osyncstream synced_error(*get_error_stream(vm["verbose"].as<Options::Verbosity>()));
    synced_output << "Processing " << path << std::endl;
    std::map<std::string,std::string> data;
    data["dataset"]=vm["dataset"].as<std::string>();
    data["graph"]=path.filename();
    std::unique_ptr<MeanPayoffGameBase<integer>> graph;
    std::unique_ptr<MaxAtomSolver<std::vector<order_closure<integer>>,integer>> solver;
    try {
        switch (vm["graph-implementation"].as<Options::GraphImplementation>()) {
            using namespace Implementation;
            case Options::GraphImplementation::ADJACENCY_LIST_VECTOR:
                graph = std::make_unique<Matrix::MeanPayoffGame<integer>>(
                        mpg_from_file<Matrix::MeanPayoffGame<integer>>(path));
                break;
            case Options::GraphImplementation::ADJACENCY_LIST_TREE:
                graph = std::make_unique<TreeMap::MeanPayoffGame<integer>>(
                        mpg_from_file<TreeMap::MeanPayoffGame<integer>>(path));
                break;
            case Options::GraphImplementation::ADJACENCY_LIST_HASH:
                graph = std::make_unique<HashMap::MeanPayoffGame<integer>>(
                        mpg_from_file<HashMap::MeanPayoffGame<integer>>(path));
                break;
            case Options::GraphImplementation::ADJACENCY_MATRIX:
                graph = std::make_unique<Matrix::MeanPayoffGame<integer>>(
                        mpg_from_file<Matrix::MeanPayoffGame<integer>>(path));
                break;
        }
    }
    catch (std::exception& e)
    {
        synced_error << "Error while reading graph " << path << ": " << e.what() << std::endl;
        data["status"]="I/O error";
        outputWriter.writeData(data);
        return;
    }

    switch(vm["max-solver"].as<Options::SolverImplementation>())
    {
        using namespace Implementation;
        case Options::SolverImplementation::ARC_CONSISTENCY:
            if (vm["dense"].as<bool>())
                solver = std::make_unique<Vector::DenseSolver<integer>>();
            else
                solver = std::make_unique<Vector::MaxAtomSystemSolver<integer>>();
            break;
        case Options::SolverImplementation::FIXED_POINT:
            solver = std::make_unique<Vector::MaxAtomSystemSolverFixedPoint<integer>>();
            break;
    }
    if(vm["small-domain"].as<bool>())
        solver->set_bound_estimator(new LinearMaxAtomBoundEstimator<integer>(vm["domain-coefficient"].as<int>()));
    synced_output << "Calculating optimal strategy for graph " << path << std::endl;
    std::optional<StrategyPair> strategy;
    try{
        if(vm["running-time"].as<bool>())
        {
            std::chrono::high_resolution_clock::time_point start=std::chrono::high_resolution_clock::now();
            strategy=optimal_strategy_pair(*graph,*solver);
            std::chrono::high_resolution_clock::time_point end=std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            data["running_time"]=std::to_string(elapsed_seconds.count());
            synced_output << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
        }
        else
            strategy=optimal_strategy_pair(*graph,*solver);
        if(vm["mean-payoffs"].as<bool>())
        {
            synced_output << "Calculating mean payoffs for graph " << path << std::endl;
            auto MPOs=mean_payoffs(*graph,*strategy);
            auto &[MPO0,MPO1]=MPOs;
            std::ostringstream S0,S1;
            using Options::operator<<;
            S0 << MPO0;
            S1 << MPO1;
            data["mean_payoffs_max"]=S0.str();
            data["mean_payoffs_min"]=S1.str();
            if(vm["winners"].as<bool>())
            {
                synced_output << "Calculating winners for graph " << path << std::endl;
                auto [W0,W1]=winners(*graph,MPOs);
                std::ostringstream S0,S1;
                S0 << W0;
                S1 << W1;
                data["winners_max"]=S0.str();
                data["winners_min"]=S1.str();
            }
        }
        else if(vm["winners"].as<bool>())
        {
            synced_output << "Calculating winners for graph " << path << std::endl;
            auto [W0,W1]=winners(*graph,*strategy);
            std::ostringstream S0,S1;
            using Options::operator<<;
            S0 << W0;
            S1 << W1;
            data["winners_max"]=S0.str();
            data["winners_min"]=S1.str();
        }
        if(vm["adjacency-matrix"].as<bool>())
        {
            synced_output << "Calculating adjacency matrix for graph " << path << std::endl;
            std::ostringstream S;
            std::vector<std::vector<integer>> adj(graph->count_nodes(),std::vector<integer>(graph->count_nodes()));
            for(auto [u,v,w]:graph->get_edges())
                adj[u][v]=1;
            using Options::operator<<;
            S << adj;
            data["adjacency_matrix"]=S.str();
        }
        if(vm["weights-matrix"].as<bool>())
        {
            synced_output << "Calculating weights matrix for graph " << path << std::endl;
            std::ostringstream S;
            std::vector<std::vector<integer>> weights(graph->count_nodes(),std::vector<integer>(graph->count_nodes()));
            for(auto [u,v,w]:graph->get_edges())
                weights[u][v]=w;
            using Options::operator<<;
            S << weights;
            data["weights_matrix"]=S.str();
        }
    }
    catch (std::exception& e)
    {
        synced_error << "Error while calculating optimal strategy for graph " << path << ": " << e.what() << std::endl;
        data["running_time"]="-1";
        data["status"]="Algorithm error";
        outputWriter.writeData(data);
        return;
    }

    std::ostringstream S0,S1;
    using Options::operator<<;
    S0 << strategy->strategy_0;
    S1 << strategy->strategy_1;
    data["max_strategy"]=S0.str();
    data["min_strategy"]=S1.str();
    data["status"]="OK";
    outputWriter.writeData(data);
    synced_output << std::string(80,'-') << std::endl;
}

int main(int argc, char *argv[]) {
    namespace po = boost::program_options;
    po::options_description desc("Command Line options");
    unsigned int cpus=1;
    using namespace Options;
    desc.add_options()
            ("help,h", "produce help message")
            ("graphs-folder,f", po::value<std::filesystem::path>(), "Read the folder that contains graphs")
            ("from-stdin",po::bool_switch(),"Read files from stdin")
            ("mode,m",
             "Mode of the program, either discard or resume. discard will redo all the calculations, resume will resume the calculations from the last point.")
            ("output,o", po::value<std::filesystem::path>(), "Output file for the results")
            ("verbose,v", po::value<Verbosity>()->default_value(Verbosity::INFO), "Verbosity level (0-3)")
            ("winners", po::bool_switch(),
             "Calculate the winners on a mean payoff graph for each vertex and each starting player")
            ("mean-payoffs", po::bool_switch(),
             "Calculate the mean payoffs for each vertex and each starting player")
            ("running-time", po::bool_switch(),
             "Calculate the running time for each graph")
             ("adjacency-matrix", po::bool_switch(), "Save also the adjacency matrix of the graph")
             ("weights-matrix",po::bool_switch(), "Save also the weights matrix of the graph")
            ("output-format", po::value<OutputFormat>()->default_value(OutputFormat::JSON), "Format of the output file")
            ("dataset", po::value<std::string>()->default_value("dataset"), "Name of the dataset")
            ("graph-implementation",
             po::value<GraphImplementation>()->default_value(GraphImplementation::ADJACENCY_LIST_HASH),
             "Implementation of the graph (adjacency-list, adjacency-matrix, edge-list)")
            ("max-solver", po::value<SolverImplementation>()->default_value(SolverImplementation::ARC_CONSISTENCY),
             "Implementation of the max atom solver")
            ("dataset-description", po::value<std::string>()->default_value("dataset"), "Description of the dataset")
            ("threads,t", po::value<unsigned int>()->default_value(1), "Number of threads to use, Each thread will process a different graph")
            ("separator", po::value<char>()->default_value(','), "Separator for the csv file")
            ("dense", po::bool_switch(),"Use the dense graph heuristic")
            ("small-domain", po::bool_switch(),"Use the small domain heuristic")
            ("domain-coefficient",po::value<int>()->default_value(2),"Growth ratio of the domain. Ignored if the small domain heuristic is not used");
    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || argc == 1) {
        std::cout << desc << std::endl;
        return 0;
    }
    if (!vm.count("graphs-folder"))
        vm.at("from-stdin").as<bool>()=true;
    if (!vm.count("output"))
    {
        std::cout << "You must specify an output file" << std::endl;
        return 1;
    }

    std::optional<std::filesystem::path> graphs_folder;
    if (vm.count("graphs-folder"))
        graphs_folder = vm.at("graphs-folder").as<std::filesystem::path>();
    std::filesystem::path output_file = vm.at("output").as<std::filesystem::path>();
    Result::MultipleWriterUnique outputWriter;
    Result::ParallelWriter parallelWriter(outputWriter);
    std::vector<std::string> headers = {"dataset", "graph", "running_time", "min_strategy", "max_strategy", "status",
                                        "mean_payoffs_min", "mean_payoffs_max", "winners_min", "winners_max",
                                        "adjacency_matrix", "weights_matrix"};
    switch (vm.at("output-format").as<OutputFormat>()) {
        case OutputFormat::CSV:
            outputWriter.addWriter(new Result::CSVWriter(output_file, headers, vm["separator"].as<char>()));
            break;
        case OutputFormat::JSON: {
            auto writer = new Result::JSONWriter(output_file);
            outputWriter.addWriter(writer);
            break;
        }
    }
    using Options::operator<=>;
    if(vm.at("verbose").as<Verbosity>()>=Verbosity::EXECUTION)
        outputWriter.addWriter(new Result::HumanWriter(&std::cout));
    outputWriter.writeHeader();
    outputWriter.addType("dataset", Result::JSONWriter::STRING);
    outputWriter.addType("graph", Result::JSONWriter::Type::STRING);
    outputWriter.addType("running_time", Result::JSONWriter::Type::NUMBER);
    outputWriter.addType("min_strategy", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("max_strategy", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("status", Result::JSONWriter::Type::STRING);
    outputWriter.addType("mean_payoffs_min", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("mean_payoffs_max", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("winners_min", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("winners_max", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("adjacency_matrix", Result::JSONWriter::Type::ARRAY);
    outputWriter.addType("weights_matrix", Result::JSONWriter::Type::ARRAY);
    moodycamel::ConcurrentQueue<std::filesystem::path> queue;
    if(graphs_folder.has_value()) for (const auto &path: std::filesystem::recursive_directory_iterator(*graphs_folder)) if (std::filesystem::is_regular_file(path))
            queue.enqueue(path);

    if (vm.at("from-stdin").as<bool>())
    {
        std::string line;
        while (std::getline(std::cin, line)) if(std::filesystem::exists(line) && std::filesystem::is_regular_file(line))
            queue.enqueue(line);
    }

    cpus = vm["threads"].as<unsigned int>();
    if(cpus ==0)
        cpus = std::thread::hardware_concurrency();
    if(cpus == 1)
    {
        std::filesystem::path path;
        while (queue.try_dequeue(path))
            process_game(path, parallelWriter,vm);
    }
    else
    {
        std::vector<std::thread> executors;
        executors.reserve(cpus);
        for (int i = 0; i < cpus; ++i)
        {
            executors.emplace_back(
                [&queue, &vm, &parallelWriter]()
                 {
                     std::filesystem::path path;
                     while (queue.try_dequeue(path))
                         process_game(path, parallelWriter,vm);
                 }
             );
        }
        for (auto &thread : executors)
            thread.join();
    }
    outputWriter.writeFooter();
    return 0;
}
