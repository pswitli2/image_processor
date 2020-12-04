#include "ConfigFile.hpp"
#include "ImageProcessor.hpp"
#include "ThresholdingCPU.hpp"

int main(int argc, const char** argv)
{

    if (!ConfigFile::initialize(argc, argv))
    {
        std::cout << "Unable to initialize ConfigFile, error: " << ConfigFile::error() << std::endl;
        exit(1);
    }

    ConfigFile::print_params();

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

    exit(0);

    // const auto config_filepath = get_config_filepath(argc, argv);

    // const auto param_map = parse_configfile(config_filepath);

    // std::string input_dir = "";
    // std::string output_dir = "";
    // DisplayType display_type = DisplayType::NONE;
    // double delay = -1.0;
    // extract_params(param_map, input_dir, output_dir, display_type, delay);
    
    // exit(0);

    // std::string input_image_path = "frame_00233.png";
    // std::string output_image_path = "output.png";
    // png::image<pixel_t> image(input_image_path);

    // pixel_t max = 0;
    // size_t avg = 0;
    // const size_t h = image.get_height();
    // const size_t w = image.get_width();
    // const size_t size = h * w;

    // for (size_t c = 0; c < image.get_height(); c++)
    // {
    //     for (size_t r = 0; r < image.get_width(); r++)
    //     {
    //         const pixel_t p = image.get_pixel(r, c);
    //         if (p > max)
    //             max = p;
    //         avg += p;
    //     }
    // }
    // avg = avg / size;

    // std::cout << "height:  " << h << std::endl
    //           << "width:   " << w << std::endl
    //           << "max:     " << max << std::endl
    //           << "average: " << avg << std::endl;

    // for (size_t c = 0; c < image.get_height(); c++)
    // {
    //     for (size_t r = 0; r < image.get_width(); r++)
    //     {
    //         const pixel_t p = image.get_pixel(r, c);

    //         pixel_t set = 0;
    //         if (p > avg * 1.1)
    //             set = max;
    //         image.set_pixel(r, c, set);
            
    //     }
    // }
    // image.write(output_image_path);

    // ImageDisplay image_display("image display");
    // for (size_t i = 0; i < 5; i++)
    // {
    //     image_display.update_image(input_image_path);
    //     sleep(1);
    //     image_display.update_image(output_image_path);
    //     sleep(1);
    // }

    exit(0);
}
