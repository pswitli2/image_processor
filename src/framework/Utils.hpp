#ifndef TYPES_H_
#define TYPES_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include <magic_enum.hpp>
#include <png++/png.hpp>

/**
 * namespace shortcuts
 */
namespace fs = std::filesystem;

/**
 * forward declares
 */
class BaseImageAlgorithm;
class ImageDisplay;

/**
 * typedefs
 */
typedef fs::path path_t;
typedef png::gray_pixel_16 pixel_t;
typedef png::image<pixel_t> image_t;
typedef std::vector<pixel_t> image_data_t;
typedef std::map<std::string, std::string> params_t;

typedef std::shared_ptr<BaseImageAlgorithm> BaseImageAlgorithm_ptr;
typedef std::vector<BaseImageAlgorithm_ptr> BaseImageAlgorithm_vec;
typedef std::vector<BaseImageAlgorithm_vec> BaseImageAlgorithm_vecs;
typedef std::shared_ptr<ImageDisplay> ImageDisplay_ptr;

/**
 * param names
 */
constexpr const char INPUT_DIR_PARAM_NAME[] = "INPUT_DIR";
constexpr const char OUTPUT_DIR_PARAM_NAME[] = "OUTPUT_DIR";
constexpr const char DISPLAY_TYPE_PARAM_NAME[] = "DISPLAY_TYPE";
constexpr const char DELAY_PARAM_NAME[] = "DELAY";
constexpr const char THRESHOLD_TOLERANCE_PARAM_NAME[] = "THRESHOLD_TOLERANCE";

/**
 * enums
 */
enum class DisplayType
{
    NONE,
    FIRST_LAST,
    ALL
};
enum class LogLevel
{
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
    SIZE
};
template<class T>
static inline bool string_to_enum(const std::string& str, T& val)
{
    const auto opt = magic_enum::enum_cast<T>(str);
    if (opt.has_value())
    {
        val = opt.value();
        return true;
    }

    return false;
}

/**
 * time helper macros
 */
#define TIME_NOW() std::chrono::high_resolution_clock::now()
#define DURATION_NS(dt) (double) std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()

static inline image_t vec_to_image(const image_data_t& image_data, const size_t width, const size_t height)
{
    image_t image(width, height);
    for (size_t r = 0; r < height; r++)
        for (size_t c = 0; c < width; c++)
            image.set_pixel(c, r, image_data[width * r + c]);
    return image;
}

static inline image_data_t image_to_vec(const image_t& image)
{
    image_data_t image_data(image.get_width() * image.get_height());
    for (size_t r = 0; r < image.get_height(); r++)
        for (size_t c = 0; c < image.get_width(); c++)
            image_data[image.get_width() * r + c] = image.get_pixel(c, r);
    return image_data;
}

#endif /** TYPES_H_ */
