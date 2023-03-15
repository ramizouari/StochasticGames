#include <iostream>
#include "MeanPayoffGame.h"
#include "MinMaxSystem.h"
#include "MPGReader.h"
#include "Writer.h"
#include "ProgramOptions.h"
#include <boost/program_options.hpp>
#include <filesystem>
#include <boost/log/trivial.hpp>

int main(int argc, char *argv[]) {
    namespace po=boost::program_options;
    po::options_description desc("Command Line options");
    desc.add_options()
            ("help,h", "produce help message")
            ("graphs-folder,f",po::value<std::filesystem::path>(),"Read the folder that contains graphs")
            ("mode,m",
             "Mode of the program, either discard or resume. discard will redo all the calculations, resume will resume the calculations from the last point.")
            ("output,o",po::value<std::filesystem::path>(),"Output file for the results")
            ("verbose,v",po::value<Verbosity>()->default_value(Verbosity::INFO),"Verbosity level (0-3)")
            ("calculate-winners",po::value<bool>()->default_value(true),"Calculate the winners on a mean payoff graph for each vertex and each starting player")
            ("calculate-mean-payoffs",po::value<bool>()->default_value(true),"Calculate the mean payoffs for each vertex and each starting player")
            ("calculate-running-time",po::value<bool>()->default_value(true),"Calculate the running time for each graph")
            ("output-format",po::value<OutputFormat>()->default_value(OutputFormat::CSV),"Format of the output file")
            ("dataset-name",po::value<std::string>()->default_value("dataset"),"Name of the dataset")
            ("graph-implementation",po::value<GraphImplementation>()->default_value(GraphImplementation::ADJACENCY_LIST_HASH),"Implementation of the graph (adjacency-list, adjacency-matrix, edge-list)")
            ("max-solver",po::value<SolverImplementation>()->default_value(SolverImplementation::ARC_CONSISTENCY),"Implementation of the max atom solver")
            ("dataset-description",po::value<std::string>()->default_value("dataset"),"Description of the dataset")
            ("separator",po::value<char>()->default_value(','),"Separator for the csv file");
    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    if(vm.count("help") || argc==1)
    {
        std::cout << desc << std::endl;
        return 0;
    }
    if(!vm.count("graphs-folder"))
    {
        std::cout << "You must specify a folder containing graphs" << std::endl;
        return 1;
    }
    if(!vm.count("output"))
    {
        std::cout << "You must specify an output folder" << std::endl;
        return 1;
    }

    std::filesystem::path graphs_folder=vm["graphs-folder"].as<std::filesystem::path>();
    std::filesystem::path output_file=vm["output"].as<std::filesystem::path>();
    std::unique_ptr<Result::FileWriter> outputWriter;
    std::vector<std::string > headers={"dataset","graph","running-time","winners"};
    switch (vm["output-format"].as<OutputFormat>())
    {
        case OutputFormat::CSV:
            outputWriter = std::make_unique<Result::CSVWriter>(output_file,headers,vm["separator"].as<char>());
            break;
        case OutputFormat::JSON:
            outputWriter = std::make_unique<Result::JSONWriter>(output_file);
            break;
    }

    outputWriter->writeHeader();

    for(const auto& path:std::filesystem::recursive_directory_iterator(graphs_folder)) if(std::filesystem::is_regular_file(path))
    {
        std::map<std::string,std::string> data;
        data["dataset"]=vm["dataset-name"].as<std::string>();
        data["graph"]=path.path().filename();
        std::unique_ptr<MeanPayoffGameBase<integer>> graph;
        std::unique_ptr<MaxAtomSolver<std::vector<order_closure<integer>>,integer>> solver;
        try {
            switch (vm["graph-implementation"].as<GraphImplementation>()) {
                using namespace Implementation;
                case GraphImplementation::ADJACENCY_LIST_VECTOR:
                    graph = std::make_unique<Matrix::MeanPayoffGame<integer>>(
                            mpg_from_file<Matrix::MeanPayoffGame<integer>>(path));
                    break;
                case GraphImplementation::ADJACENCY_LIST_TREE:
                    graph = std::make_unique<TreeMap::MeanPayoffGame<integer>>(
                            mpg_from_file<TreeMap::MeanPayoffGame<integer>>(path));
                    break;
                case GraphImplementation::ADJACENCY_LIST_HASH:
                    graph = std::make_unique<HashMap::MeanPayoffGame<integer>>(
                            mpg_from_file<HashMap::MeanPayoffGame<integer>>(path));
                    break;
                case GraphImplementation::ADJACENCY_MATRIX:
                    graph = std::make_unique<Matrix::MeanPayoffGame<integer>>(
                            mpg_from_file<Matrix::MeanPayoffGame<integer>>(path));
                    break;
            }
        }
        catch (std::exception& e)
        {
            std::cerr << "Error while reading graph " << path << ": " << e.what() << std::endl;
            data["status"]="I/O error";
            outputWriter->writeData(data);
            continue;
        }

        switch(vm["max-solver"].as<SolverImplementation>())
        {
            using namespace Implementation;
            case SolverImplementation::ARC_CONSISTENCY:
                solver = std::make_unique<Vector::MaxAtomSystemSolver<integer>>();
                break;
            case SolverImplementation::FIXED_POINT:
                solver = std::make_unique<Vector::MaxAtomSystemSolverFixedPoint<integer>>();
                break;
        }

        std::optional<StrategyPair> strategy;
        try{
            std::chrono::high_resolution_clock::time_point start=std::chrono::high_resolution_clock::now();

            strategy=optimal_strategy_pair(*graph,*solver);
            std::chrono::high_resolution_clock::time_point end=std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end-start;
            data["running_time"]=std::to_string(elapsed_seconds.count());
        }
        catch (std::exception& e)
        {
            std::cerr << "Error while calculating optimal strategy for graph " << path << ": " << e.what() << std::endl;
            data["running_time"]="-1";
            data["status"]="Algorithm error";
            outputWriter->writeData(data);
            continue;
        }

        std::ostringstream S0,S1;
        S0 << strategy->strategy_0;
        S1 << strategy->strategy_1;
        data["strategy_max"]=S0.str();
        data["strategy_min"]=S1.str();
        data["status"]="OK";
        outputWriter->writeData(data);
    }
    outputWriter->writeFooter();
    return 0;
}
