#include "ImageProcessor.hpp"
#include "BackgroundRemoverCPU.hpp"
#include "EnhancerCPU.hpp"
#include "LonePixelRemoverCPU.hpp"
#include "ThresholderCPU.hpp"
#include "ThresholderCUDA.hpp"

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

    BaseImageAlgorithm_vec cpu_chain;
    cpu_chain.push_back(std::make_shared<BackgroundRemoverCPU>());
    cpu_chain.push_back(std::make_shared<ThresholderCPU>());
    cpu_chain.push_back(std::make_shared<LonePixelRemoverCPU>());

    BaseImageAlgorithm_vec cuda_chain;
    cuda_chain.push_back(std::make_shared<ThresholderCUDA>());
    // cuda_chain.push_back(std::make_shared<ThresholderCUDA>());
    // cuda_chain.push_back(std::make_shared<LonePixelRemoverCUDA>());

    BaseImageAlgorithm_vec enhancer_chain;
    enhancer_chain.push_back(std::make_shared<EnhancerCPU>());

    BaseImageAlgorithm_vecs algorithm_chains;
    algorithm_chains.push_back(cpu_chain);
    algorithm_chains.push_back(cuda_chain);
    algorithm_chains.push_back(enhancer_chain);

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
