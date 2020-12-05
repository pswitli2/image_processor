#include "ConfigFile.hpp"
#include "ImageProcessor.hpp"
#include "Logger.hpp"
#include "ThresholdingCPU.hpp"

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
    path_t filepath = "";
    LogLevel log_level = LogLevel::ERROR;

    if (!parse_args(argc, argv, filepath, log_level))
        exit(1);

    Logger::initialize(log_level);

    if (!ConfigFile::initialize(filepath))
        exit(1);

    BaseImageAlgorithm_vec algorithms;
    algorithms.push_back(std::make_shared<ThresholdingCPU>());

    ImageProcessor processor;
    if (!processor.initialize(algorithms))
        exit(1);

    if (!processor.process_images())
        exit(1);

    for (const auto& algorithm: algorithms)
    {
        LOG(LogLevel::INFO, algorithm->name(), " took ", algorithm->duration(),
                    " seconds to process ", processor.image_count(), " images");
    }
    exit(0);
}
