#include "ConfigFile.hpp"
#include "ImageProcessor.hpp"
#include "Logger.hpp"
#include "ThresholdingCPU.hpp"

int main(int argc, const char** argv)
{

    if (!ConfigFile::initialize(argc, argv))
    {
        std::cout << "Unable to initialize ConfigFile, error: " << ConfigFile::error() << std::endl;
        exit(1);
    }
    Logger::initialize(ConfigFile::log_level());

    Logger::log(LogLevel::INFO, ConfigFile::to_string());

    BaseImageAlgorithm_vec algorithms;
    algorithms.push_back(std::make_shared<ThresholdingCPU>());

    ImageProcessor processor;
    if (!processor.initialize(algorithms))
    {
        std::cout << "Unable to initialize ImageProcessor, error: " << processor.error() << std::endl;
        exit(1);
    }

    if (!processor.process_images())
    {
        std::cout << "Unable to process images, error: " << processor.error() << std::endl;
        exit(1);
    }

    for (const auto& algorithm: algorithms)
    {
        Logger::log(LogLevel::INFO, algorithm->name(), " took ", algorithm->duration(),
                    " seconds to process ", processor.image_count(), " images");
    }
    exit(0);
}
