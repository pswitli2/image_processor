#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <chrono>
#include <filesystem>

#include <magic_enum.hpp>

#include "Types.hpp"

/**
 * Define some common routines used throughout the framework.
 */

/** namespace shortcuts. */
namespace fs = std::filesystem;
typedef fs::path path_t;

/** Parameter names. */
constexpr const char INPUT_DIR_PARAM_NAME[] = "INPUT_DIR";
constexpr const char OUTPUT_DIR_PARAM_NAME[] = "OUTPUT_DIR";
constexpr const char DISPLAY_TYPE_PARAM_NAME[] = "DISPLAY_TYPE";
constexpr const char DELAY_PARAM_NAME[] = "DELAY";
constexpr const char THRESHOLD_TOLERANCE_PARAM_NAME[] = "THRESHOLD_TOLERANCE";
constexpr const char BACKGROUND_HISTORY_LENGTH_PARAM_NAME[] = "BACKGROUND_HISTORY_LENGTH";
constexpr const char BACKGROUND_TOLERANCE_PARAM_NAME[] = "BACKGROUND_TOLERANCE";
constexpr const char ENHANCEMENT_FACTOR_PARAM_NAME[] = "ENHANCEMENT_FACTOR";
constexpr const char LONE_PIXEL_NUM_ADJACENT_PARAM_NAME[] = "LONE_PIXEL_NUM_ADJACENT";

/** Enum types. */
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

/** Convert a string to an enum. */
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

/** Time helper macros. */
#define TIME_NOW() std::chrono::high_resolution_clock::now()
#define DURATION_NS(dt) (double) std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()

#endif /** UTILS_HPP_ */
