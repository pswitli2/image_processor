#ifndef TYPES_H_
#define TYPES_H_

#include <memory>

#include <png++/png.hpp>

typedef png::gray_pixel_16 pixel_t;

enum class DisplayType
{
    NONE,
    FIRST_LAST,
    ALL,
    SIZE
};

typedef png::image<pixel_t> Image;

static inline size_t DisplayTypeInt(const DisplayType& display_type)
{
    return static_cast<size_t>(display_type);
}

static inline std::string DisplayTypeString(const DisplayType& display_type)
{
    switch (display_type)
    {
        case DisplayType::NONE: return "NONE";
        case DisplayType::FIRST_LAST: return "FIRST_LAST";
        case DisplayType::ALL: return "ALL";
        case DisplayType::SIZE: return "SIZE";
        default: return "";
    }
}

#endif /** TYPES_H_ */
