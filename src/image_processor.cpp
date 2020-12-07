#include "BackgroundRemoverCPU.hpp"
#include "BackgroundRemoverCUDA.hpp"
#include "EnhancerCPU.hpp"
#include "ImageProcessor.hpp"
#include "LonePixelRemoverCPU.hpp"
#include "LonePixelRemoverCUDA.hpp"
#include "ThresholderCPU.hpp"
#include "ThresholderCUDA.hpp"

/**
 * Parse command line arguments.
 */
static bool parse_args(int argc, const char** argv, path_t& filepath, LogLevel& log_level);

int main(int argc, const char** argv)
{
    const auto start = TIME_NOW();

    // extract command line arguments
    path_t filepath = "";
    LogLevel log_level = LogLevel::INFO;
    if (!parse_args(argc, argv, filepath, log_level))
        exit(1);

    // initialize static framework classes
    Logger::initialize(log_level);
    if (!ConfigFile::initialize(filepath))
        exit(1);

    // setup CPU image processing chain
    BaseImageAlgorithm_vec cpu_chain;
    cpu_chain.push_back(std::make_shared<BackgroundRemoverCPU>());
    cpu_chain.push_back(std::make_shared<ThresholderCPU>());
    cpu_chain.push_back(std::make_shared<LonePixelRemoverCPU>());

    // setup CUDA image processing chain
    BaseImageAlgorithm_vec cuda_chain;
    cuda_chain.push_back(std::make_shared<BackgroundRemoverCUDA>());
    cuda_chain.push_back(std::make_shared<ThresholderCUDA>());
    cuda_chain.push_back(std::make_shared<LonePixelRemoverCUDA>());

    // setup chain for viewing an enhanced raw image
    BaseImageAlgorithm_vec enhancer_chain;
    enhancer_chain.push_back(std::make_shared<EnhancerCPU>());

    // initialize the ImageProcessor
    BaseImageAlgorithm_vecs algorithm_chains;
    algorithm_chains.push_back(cpu_chain);
    algorithm_chains.push_back(cuda_chain);
    algorithm_chains.push_back(enhancer_chain);
    ImageProcessor processor;
    if (!processor.initialize(algorithm_chains))
        exit(1);
    processor.log_info();

    // process images
    if (!processor.process_images())
        exit(1);
    processor.log_results();

    // display total execution time
    const auto duration = DURATION_NS(TIME_NOW() - start);
    LOG(LogLevel::INFO, "Total program execution time: ", duration * 1e-9, " seconds");
    exit(0);
}

static bool parse_args(int argc, const char** argv, path_t& filepath, LogLevel& log_level)
{
    // validate number of arguments
    if (argc != 2 && argc != 3)
    {
        std::cout << "Usage: ./image_processor <path/to/config/file> "
                  << "[LogLevel (ERROR,WARN,INFO,DEBUG,TRACE)]" << std::endl; 
        return false;
    }

    // extract arguments
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
