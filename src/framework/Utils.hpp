#ifndef UTILS_HPP_
#define UTILS_HPP_

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
// #include <png++/png.hpp>

#include "Types.hpp"

/**
 * namespace shortcuts
 */
namespace fs = std::filesystem;
typedef fs::path path_t;

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

// static inline image_t vec_to_image(const image_data_t& image_data, const size_t width, const size_t height)
// {
//     image_t image(width, height);
//     for (size_t r = 0; r < height; r++)
//         for (size_t c = 0; c < width; c++)
//             image.set_pixel(c, r, (pixel_t) image_data[width * r + c]);
//     return image;
// }

// static inline image_data_t image_to_vec(const image_t& image)
// {
//     image_data_t image_data(image.get_width() * image.get_height());
//     for (size_t r = 0; r < image.get_height(); r++)
//         for (size_t c = 0; c < image.get_width(); c++)
//             image_data[image.get_width() * r + c] = (pixel64_t) image.get_pixel(c, r);
//     return image_data;
// }

#endif /** UTILS_HPP_ */
