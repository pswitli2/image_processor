#include "ImageDisplay.hpp"
#include "ImageProcessor.hpp"
// #include "ImageDisplay.hpp"

int main()
{
    std::string input_image_path = "frame_00233.png";
    std::string output_image_path = "output.png";
    png::image<pixel_t> image(input_image_path);

    pixel_t max = 0;
    size_t avg = 0;
    const size_t h = image.get_height();
    const size_t w = image.get_width();
    const size_t size = h * w;

    for (size_t c = 0; c < image.get_height(); c++)
    {
        for (size_t r = 0; r < image.get_width(); r++)
        {
            const pixel_t p = image.get_pixel(r, c);
            if (p > max)
                max = p;
            avg += p;
        }
    }
    avg = avg / size;

    std::cout << "height:  " << h << std::endl
              << "width:   " << w << std::endl
              << "max:     " << max << std::endl
              << "average: " << avg << std::endl;

    for (size_t c = 0; c < image.get_height(); c++)
    {
        for (size_t r = 0; r < image.get_width(); r++)
        {
            const pixel_t p = image.get_pixel(r, c);

            pixel_t set = 0;
            if (p > avg * 1.1)
                set = max;
            image.set_pixel(r, c, set);
            
        }
    }
    image.write(output_image_path);

    ImageDisplay image_display("image display");
    for (size_t i = 0; i < 5; i++)
    {
        image_display.update_image(input_image_path);
        sleep(1);
        image_display.update_image(output_image_path);
        sleep(1);
    }

    return 0;
}