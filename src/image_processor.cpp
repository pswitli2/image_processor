#include "ImageProcessor.hpp"
#include "ThresholdingCPU.hpp"
#include "ThresholdingCUDA.hpp"

static bool parse_args(int argc, const char** argv, path_t& filepath, LogLevel& log_level)
{
    // validate argument exists
    if (argc != 2 && argc != 3)
    {
        std::cout << "Usage: ./image_processor <path/to/config/file> "
                  << "[LogLevel (ERROR,WARN,INFO,DEBUG,TRACE)]" << std::endl; 
        return false;
    }

    filepath = argv[1];

    if (argc == 3)
    {
        if (!string_to_enum(argv[2], log_level))
        {
            std::cout << "Invalid LogLevel argument: '" << argv[2]
                      << "' must be ERROR, WARN, INFO, DEBUG, or TRACE" << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, const char** argv)
{
    const auto start = TIME_NOW();

    path_t filepath = "";
    LogLevel log_level = LogLevel::ERROR;

    if (!parse_args(argc, argv, filepath, log_level))
        exit(1);

    Logger::initialize(log_level);

    if (!ConfigFile::initialize(filepath))
        exit(1);

    BaseImageAlgorithm_vec chain0;
    chain0.push_back(std::make_shared<ThresholdingCPU>());

    BaseImageAlgorithm_vec chain1;
    chain1.push_back(std::make_shared<ThresholdingCUDA>());

    BaseImageAlgorithm_vecs algorithm_chains;
    algorithm_chains.push_back(chain0);
    algorithm_chains.push_back(chain1);

    ImageProcessor processor;
    if (!processor.initialize(algorithm_chains))
        exit(1);
    processor.log_info();

    if (!processor.process_images())
        exit(1);

    processor.log_results();

    const auto end = TIME_NOW();
    const auto duration = DURATION_NS(end - start);
    LOG(LogLevel::INFO, "Total program execution time: ", duration * 1e-9, " seconds");
    exit(0);
}
