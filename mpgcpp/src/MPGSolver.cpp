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

void process_game(const std::filesystem::path& path,Result::ParallelWriter& outputWriter, const boost::program_options::variables_map& vm)
{
    std::osyncstream synced_output(std::cout);
    std::osyncstream synced_error(std::cerr);
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
            solver = std::make_unique<Vector::MaxAtomSystemSolver<integer>>();
            break;
        case Options::SolverImplementation::FIXED_POINT:
            solver = std::make_unique<Vector::MaxAtomSystemSolverFixedPoint<integer>>();
            break;
    }
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
            ("mode,m",
             "Mode of the program, either discard or resume. discard will redo all the calculations, resume will resume the calculations from the last point.")
            ("output,o", po::value<std::filesystem::path>(), "Output file for the results")
            ("verbose,v", po::value<Verbosity>()->default_value(Verbosity::INFO), "Verbosity level (0-3)")
            ("winners", po::value<bool>()->default_value(false),
             "Calculate the winners on a mean payoff graph for each vertex and each starting player")
            ("mean-payoffs", po::value<bool>()->default_value(false),
             "Calculate the mean payoffs for each vertex and each starting player")
            ("running-time", po::value<bool>()->default_value(true),
             "Calculate the running time for each graph")
            ("output-format", po::value<OutputFormat>()->default_value(OutputFormat::CSV), "Format of the output file")
            ("dataset", po::value<std::string>()->default_value("dataset"), "Name of the dataset")
            ("graph-implementation",
             po::value<GraphImplementation>()->default_value(GraphImplementation::ADJACENCY_LIST_HASH),
             "Implementation of the graph (adjacency-list, adjacency-matrix, edge-list)")
            ("max-solver", po::value<SolverImplementation>()->default_value(SolverImplementation::ARC_CONSISTENCY),
             "Implementation of the max atom solver")
            ("dataset-description", po::value<std::string>()->default_value("dataset"), "Description of the dataset")
            ("threads,t", po::value<unsigned int>()->default_value(1), "Number of threads to use, Each thread will process a different graph")
            ("separator", po::value<char>()->default_value(','), "Separator for the csv file");
    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    if (vm.count("help") || argc == 1) {
        std::cout << desc << std::endl;
        return 0;
    }
    if (!vm.count("graphs-folder")) {
        std::cout << "You must specify a folder containing graphs" << std::endl;
        return 1;
    }
    if (!vm.count("output"))
    {
        std::cout << "You must specify an output file" << std::endl;
        return 1;
    }

    std::filesystem::path graphs_folder = vm.at("graphs-folder").as<std::filesystem::path>();
    std::filesystem::path output_file = vm.at("output").as<std::filesystem::path>();
    Result::MultipleWriterUnique outputWriter;
    Result::ParallelWriter parallelWriter(outputWriter);
    std::vector<std::string> headers = {"dataset", "graph", "running_time", "min_strategy", "max_strategy", "status",
                                        "mean_payoffs_min", "mean_payoffs_max", "winners_min", "winners_max"};
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
    moodycamel::ConcurrentQueue<std::filesystem::path> queue;
    for (const auto &path: std::filesystem::recursive_directory_iterator(graphs_folder)) if (std::filesystem::is_regular_file(path))
            queue.enqueue(path);

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
